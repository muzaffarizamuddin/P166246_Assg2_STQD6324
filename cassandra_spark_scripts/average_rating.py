# -*- coding: utf-8 -*-
# =============================================================================
# average_rating.py
# Assignment 2 | STQD6324 | P166246 Muzaffar Izamuddin
#
# Tasks:
#   i)  Calculate the average rating for each movie
#   ii) Identify the top 10 movies with the highest average ratings
#
# Run with:
#   spark-submit --packages com.datastax.spark:spark-cassandra-connector_2.11:2.3.0 \
#       cassandra/average_rating.py
#
#   For Spark 3.x (Scala 2.12), replace the package with:
#   com.datastax.spark:spark-cassandra-connector_2.12:3.2.0
# =============================================================================

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import desc, lit, row_number
from pyspark.sql.window import Window
from cassandra.cluster import Cluster

# =============================================================================
# CONFIGURATION  —  change HDFS_BASE to match your cluster username/path
# =============================================================================
CASSANDRA_HOST = "127.0.0.1"
KEYSPACE       = "movielens"
HDFS_BASE      = "hdfs:///user/maria_dev/ml-100k"   # e.g. hdfs:///user/hadoop/ml-100k
MIN_RATINGS    = 10    # minimum number of ratings needed to appear in top-10

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

# =============================================================================
# CASSANDRA SETUP
# Creates the required tables (drops old data so script is safe to re-run)
# =============================================================================
def setup_cassandra():
    cluster = Cluster([CASSANDRA_HOST])
    session = cluster.connect()

    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS movielens
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    """)
    session.set_keyspace(KEYSPACE)

    # Table for Task i  — one row per movie
    session.execute("""
        CREATE TABLE IF NOT EXISTS avg_ratings (
            movie_id     INT PRIMARY KEY,
            title        TEXT,
            avg_rating   DOUBLE,
            rating_count BIGINT
        )
    """)
    session.execute("TRUNCATE avg_ratings")

    # Table for Task ii — top-10 stored under a single partition key for ordering
    session.execute("""
        CREATE TABLE IF NOT EXISTS top_movies (
            partition_key INT,
            rank          INT,
            movie_id      INT,
            title         TEXT,
            avg_rating    DOUBLE,
            rating_count  BIGINT,
            PRIMARY KEY (partition_key, rank)
        ) WITH CLUSTERING ORDER BY (rank ASC)
    """)
    session.execute("TRUNCATE top_movies")

    print("Cassandra keyspace and tables ready.")
    cluster.shutdown()

# =============================================================================
# PARSE FUNCTIONS
# =============================================================================
def parse_rating(line):
    # u.data format: user_id \t movie_id \t rating \t timestamp
    f = line.split("\t")
    return Row(user_id=int(f[0]), movie_id=int(f[1]), rating=int(f[2]), ts=int(f[3]))

def parse_movie(line):
    # u.item format: movie_id | title | release_date | ... | genre_flag x 19
    f = line.split("|")
    genre_list = [GENRES[i] for i in range(19) if len(f) > 5 + i and f[5 + i] == "1"]
    return Row(movie_id=int(f[0]), title=f[1], release_date=f[2], genres=genre_list)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # Step 1: Prepare Cassandra tables
    setup_cassandra()

    # Step 2: Create Spark session with Cassandra connector
    spark = (SparkSession.builder
             .appName("MovieLens_AvgRating_P166246")
             .config("spark.cassandra.connection.host", CASSANDRA_HOST)
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    print("Spark version:", spark.version)

    # -------------------------------------------------------------------------
    # Step 3: Load raw data from HDFS into RDDs
    # -------------------------------------------------------------------------
    ratings_rdd = sc.textFile(HDFS_BASE + "/u.data").map(parse_rating)
    movies_rdd  = sc.textFile(HDFS_BASE + "/u.item").map(parse_movie)
    print("Ratings loaded:", ratings_rdd.count())
    print("Movies loaded :", movies_rdd.count())

    # -------------------------------------------------------------------------
    # Step 4: Convert RDDs to DataFrames and register as SQL views
    # -------------------------------------------------------------------------
    ratings_df = spark.createDataFrame(ratings_rdd)
    movies_df  = spark.createDataFrame(movies_rdd)
    ratings_df.createOrReplaceTempView("ratings")
    movies_df.createOrReplaceTempView("movies")

    # =========================================================================
    # TASK i — Average Rating for Each Movie
    # =========================================================================
    print("\n" + "=" * 62)
    print("TASK i  —  Average Rating for Each Movie")
    print("=" * 62)

    avg_ratings_df = spark.sql("""
        SELECT
            r.movie_id,
            m.title,
            ROUND(AVG(r.rating), 4)  AS avg_rating,
            COUNT(r.rating)          AS rating_count
        FROM ratings r
        JOIN movies  m ON r.movie_id = m.movie_id
        GROUP BY r.movie_id, m.title
        ORDER BY avg_rating DESC
    """)

    print("Top 20 movies by average rating:")
    avg_ratings_df.show(20, truncate=False)
    print("Total movies with at least one rating:", avg_ratings_df.count())

    # Write result to Cassandra
    avg_ratings_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(table="avg_ratings", keyspace=KEYSPACE) \
        .save()
    print("avg_ratings written to Cassandra.")

    # Read back from Cassandra to validate
    print("\nReading avg_ratings back from Cassandra (top 10):")
    spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="avg_ratings", keyspace=KEYSPACE) \
        .load() \
        .orderBy(desc("avg_rating")) \
        .show(10, truncate=False)

    # =========================================================================
    # TASK ii — Top 10 Movies with the Highest Average Ratings
    # =========================================================================
    print("\n" + "=" * 62)
    print("TASK ii  —  Top 10 Movies (min %d ratings)" % MIN_RATINGS)
    print("=" * 62)

    top10_df = spark.sql("""
        SELECT
            r.movie_id,
            m.title,
            ROUND(AVG(r.rating), 4)  AS avg_rating,
            COUNT(r.rating)          AS rating_count
        FROM ratings r
        JOIN movies  m ON r.movie_id = m.movie_id
        GROUP BY r.movie_id, m.title
        HAVING COUNT(r.rating) >= %d
        ORDER BY avg_rating DESC
        LIMIT 10
    """ % MIN_RATINGS)

    print("Top 10 movies:")
    top10_df.show(truncate=False)

    # Add rank (1-10) and partition_key (constant=1) for Cassandra composite key
    window_spec = Window.orderBy(desc("avg_rating"))
    top10_cassandra = (top10_df
        .withColumn("rank",          row_number().over(window_spec))
        .withColumn("partition_key", lit(1)))

    top10_cassandra.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(table="top_movies", keyspace=KEYSPACE) \
        .save()
    print("top_movies written to Cassandra.")

    # Read back from Cassandra to validate
    print("\nReading top_movies back from Cassandra:")
    spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="top_movies", keyspace=KEYSPACE) \
        .load() \
        .orderBy("rank") \
        .show(truncate=False)

    spark.stop()
    print("\nDone.")
