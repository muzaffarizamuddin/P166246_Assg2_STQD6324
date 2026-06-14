# -*- coding: utf-8 -*-
# =============================================================================
# mongodb_analysis.py
# Assignment 2 Extension | STQD6324 | P166246 | Muzaffar Izamuddin
#
# MongoDB Integration — Finding and Storing Least Liked Movies for Removal
#
# Process:
#   1. Find each user's favourite genre (the genre they rated most).
#   2. Find unrated movies in that genre with a minimum of 5 global ratings.
#   3. Sort ascending by global average rating to isolate the LEAST liked films.
#   4. Store the bottom-5 movies per user inside MongoDB document store.
# =============================================================================

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import explode, col, asc, row_number
from pyspark.sql.window import Window
import pymongo
import os
import csv
from collections import namedtuple

# =============================================================================
# CONFIGURATION
# =============================================================================
HDFS_BASE    = "hdfs:///user/maria_dev/ml-100k"
MONGO_HOST   = "localhost"
MONGO_PORT   = 27017
DB_NAME      = "movielens_exclusion_db"
COLL_NAME    = "least_liked_movies"
MIN_RATINGS  = 5    # Minimum global ratings for eligibility
BOTTOM_N     = 5    # Number of negative recommendations to capture

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# =============================================================================
# PARSE FUNCTIONS
# =============================================================================
def parse_rating(line):
    f = line.split("\t")
    return Row(user_id=int(f[0]), movie_id=int(f[1]), rating=float(f[2]))

def parse_movie(line):
    f = line.split("|")
    genre_list = [GENRES[i] for i in range(19) if len(f) > 5 + i and f[5 + i] == "1"]
    return Row(movie_id=int(f[0]), title=f[1], genres=genre_list)

if __name__ == "__main__":
    CSV_PATH = "/home/maria_dev/least_liked_cache.csv"
    RecRow = namedtuple("RecRow", ["user_id", "rn", "movie_id", "title", "avg_rating", "fav_genre"])
    spark = None

    if os.path.exists(CSV_PATH):
        print("Cached negative recommendations found at %s. Skipping Spark phase." % CSV_PATH)
        records_list = []
        with open(CSV_PATH, "r") as f:
            for r in csv.DictReader(f):
                records_list.append(RecRow(
                    user_id=int(r["user_id"]), rn=int(r["rank"]), movie_id=int(r["movie_id"]),
                    title=r["title"], avg_rating=float(r["avg_rating"]), fav_genre=r["fav_genre"]
                ))
    else:
        # Start Spark Pipeline
        spark = (SparkSession.builder
                 .appName("MovieLens_MongoDB_Exclusions_P166246")
                 .config("spark.eventLog.enabled", "false")
                 .getOrCreate())
        sc = spark.sparkContext
        sc.setLogLevel("WARN")

        ratings_rdd = sc.textFile(HDFS_BASE + "/u.data").map(parse_rating)
        movies_rdd  = sc.textFile(HDFS_BASE + "/u.item").map(parse_movie)

        ratings_df = spark.createDataFrame(ratings_rdd)
        movies_df  = spark.createDataFrame(movies_rdd)

        movies_genres_df = movies_df.select("movie_id", "title", explode("genres").alias("genre"))
        ratings_df.createOrReplaceTempView("ratings")
        movies_genres_df.createOrReplaceTempView("movies_genres")

        # Find most watched genre per user
        fav_genre_df = spark.sql("""
            WITH genre_counts AS (
                SELECT r.user_id, mg.genre, COUNT(*) AS cnt
                FROM ratings r
                JOIN movies_genres mg ON r.movie_id = mg.movie_id
                GROUP BY r.user_id, mg.genre
            ),
            ranked AS (
                SELECT user_id, genre, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY cnt DESC) AS rn
                FROM genre_counts
            )
            SELECT user_id, genre AS fav_genre FROM ranked WHERE rn = 1
        """)
        fav_genre_df.createOrReplaceTempView("fav_genres")

        # Global average rating per movie
        metrics_by_genre_df = spark.sql("""
            SELECT mg.genre, mg.movie_id, mg.title, ROUND(AVG(r.rating), 4) AS avg_rating
            FROM ratings r
            JOIN movies_genres mg ON r.movie_id = mg.movie_id
            GROUP BY mg.genre, mg.movie_id, mg.title
            HAVING COUNT(*) >= {min_r}
        """.format(min_r=MIN_RATINGS))
        metrics_by_genre_df.createOrReplaceTempView("metrics_by_genre")

        already_rated_df = spark.sql("SELECT DISTINCT user_id, movie_id FROM ratings")
        already_rated_df.createOrReplaceTempView("already_rated")

        candidates_df = spark.sql("""
            SELECT fg.user_id, mbg.movie_id, mbg.title, mbg.avg_rating, fg.fav_genre
            FROM fav_genres fg
            JOIN metrics_by_genre mbg ON fg.fav_genre = mbg.genre
            LEFT JOIN already_rated ar ON fg.user_id = ar.user_id AND mbg.movie_id = ar.movie_id
            WHERE ar.movie_id IS NULL
        """)
        candidates_df.createOrReplaceTempView("candidates")

        # CRITICAL DIFFERENCE: Ordered by avg_rating ASCENDING to isolate worst-performing movies
        window_spec = Window.partitionBy("user_id").orderBy(asc("avg_rating"))
        exclusions_df = (candidates_df
                         .withColumn("rn", row_number().over(window_spec))
                         .filter(col("rn") <= BOTTOM_N)
                         .orderBy("user_id", "rn"))

        spark_rows = exclusions_df.collect()

        # Save to Local CSV Cache
        with open(CSV_PATH, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "rank", "movie_id", "title", "avg_rating", "fav_genre"])
            for r in spark_rows:
                writer.writerow([r.user_id, r.rn, r.movie_id, r.title, round(r.avg_rating, 4), r.fav_genre])

        records_list = [RecRow(r.user_id, r.rn, r.movie_id, r.title, r.avg_rating, r.fav_genre) for r in spark_rows]
        spark.stop()

    # =========================================================================
    # WRITE TO MONGODB (Document / Nested Array Store Mapping)
    # =========================================================================
    client = pymongo.MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    collection = db[COLL_NAME]

    # Reset Collection to ensure idempotency
    collection.drop()
    print("\nConnected to MongoDB on {}:{}. Dropped existing collections.".format(MONGO_HOST, MONGO_PORT))

    # Reorganize wide data structure into a document array mapping matching MongoDB's model
    user_documents = {}
    for r in records_list:
        if r.user_id not in user_documents:
            user_documents[r.user_id] = {
                "_id": "user_{:06d}".format(r.user_id),
                "user_id": r.user_id,
                "favorite_genre": r.fav_genre,
                "blacklisted_movies": []
            }
        
        user_documents[r.user_id]["blacklisted_movies"].append({
            "avoid_rank": r.rn,
            "movie_id": r.movie_id,
            "title": r.title,
            "global_avg_rating": r.avg_rating
        })

    # Bulk insert documents into MongoDB
    print("Migrating exclusion lists to MongoDB document structures...")
    collection.insert_many(user_documents.values())
    print("Database populate complete! Total stored user profiles: {}".format(collection.count()))
    client.close()