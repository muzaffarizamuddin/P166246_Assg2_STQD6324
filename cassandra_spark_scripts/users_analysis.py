# -*- coding: utf-8 -*-
# =============================================================================
# users_analysis.py
# Assignment 2 | STQD6324 | P166246
#
# Tasks:
#   iii) Users who rated >= 50 movies and their favourite genre
#   iv)  All users younger than 20 years old
#   v)   All scientists aged between 30 and 40
#
# Run with:
#   spark-submit --packages com.datastax.spark:spark-cassandra-connector_2.11:2.3.0 \
#       cassandra/users_analysis.py
#
#   For Spark 3.x (Scala 2.12), replace the package with:
#   com.datastax.spark:spark-cassandra-connector_2.12:3.2.0
# =============================================================================

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import desc
from cassandra.cluster import Cluster

# =============================================================================
# CONFIGURATION  —  change HDFS_BASE to match your cluster username/path
# =============================================================================
CASSANDRA_HOST = "127.0.0.1"
KEYSPACE       = "movielens"
HDFS_BASE      = "hdfs:///user/maria_dev/ml-100k"   # e.g. hdfs:///user/hadoop/ml-100k

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

# =============================================================================
# CASSANDRA SETUP
# =============================================================================
def setup_cassandra():
    cluster = Cluster([CASSANDRA_HOST])
    session = cluster.connect()

    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS movielens
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    """)
    session.set_keyspace(KEYSPACE)

    # Raw users table (mirrors u.user file)
    session.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    INT PRIMARY KEY,
            age        INT,
            gender     TEXT,
            occupation TEXT,
            zip_code   TEXT
        )
    """)
    session.execute("TRUNCATE users")

    # Task iii result — one row per active user
    session.execute("""
        CREATE TABLE IF NOT EXISTS user_fav_genres (
            user_id            INT PRIMARY KEY,
            favorite_genre     TEXT,
            genre_rating_count BIGINT
        )
    """)
    session.execute("TRUNCATE user_fav_genres")

    # Task iv result — users under 20
    session.execute("""
        CREATE TABLE IF NOT EXISTS young_users (
            user_id    INT PRIMARY KEY,
            age        INT,
            gender     TEXT,
            occupation TEXT,
            zip_code   TEXT
        )
    """)
    session.execute("TRUNCATE young_users")

    # Task v result — scientists aged 30-40
    session.execute("""
        CREATE TABLE IF NOT EXISTS scientist_users (
            user_id    INT PRIMARY KEY,
            age        INT,
            gender     TEXT,
            occupation TEXT,
            zip_code   TEXT
        )
    """)
    session.execute("TRUNCATE scientist_users")

    print("Cassandra keyspace and tables ready.")
    cluster.shutdown()

# =============================================================================
# PARSE FUNCTIONS
# =============================================================================
def parse_user(line):
    # u.user format: user_id | age | gender | occupation | zip_code
    f = line.split("|")
    return Row(user_id=int(f[0]), age=int(f[1]), gender=f[2],
               occupation=f[3], zip_code=f[4].strip())

def parse_rating(line):
    # u.data format: user_id \t movie_id \t rating \t timestamp
    f = line.split("\t")
    return Row(user_id=int(f[0]), movie_id=int(f[1]), rating=int(f[2]), ts=int(f[3]))

def parse_movie(line):
    # u.item format: movie_id | title | release_date | ... | genre_flag x 19
    f = line.split("|")
    genre_list = [GENRES[i] for i in range(19) if len(f) > 5 + i and f[5 + i] == "1"]
    return Row(movie_id=int(f[0]), title=f[1], genres=genre_list)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # Step 1: Prepare Cassandra tables
    setup_cassandra()

    # Step 2: Create Spark session with Cassandra connector
    spark = (SparkSession.builder
             .appName("MovieLens_UsersAnalysis_P166246")
             .config("spark.cassandra.connection.host", CASSANDRA_HOST)
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    print("Spark version:", spark.version)

    # -------------------------------------------------------------------------
    # Step 3: Load raw data from HDFS into RDDs
    # -------------------------------------------------------------------------
    users_rdd   = sc.textFile(HDFS_BASE + "/u.user").map(parse_user)
    ratings_rdd = sc.textFile(HDFS_BASE + "/u.data").map(parse_rating)
    movies_rdd  = sc.textFile(HDFS_BASE + "/u.item").map(parse_movie)
    print("Users   loaded:", users_rdd.count())
    print("Ratings loaded:", ratings_rdd.count())
    print("Movies  loaded:", movies_rdd.count())

    # -------------------------------------------------------------------------
    # Step 4: Convert RDDs to DataFrames
    # -------------------------------------------------------------------------
    users_df   = spark.createDataFrame(users_rdd)
    ratings_df = spark.createDataFrame(ratings_rdd)
    movies_df  = spark.createDataFrame(movies_rdd)

    # -------------------------------------------------------------------------
    # Step 5: Write users to Cassandra, then read back (as per assignment flow)
    # -------------------------------------------------------------------------
    users_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(table="users", keyspace=KEYSPACE) \
        .save()
    print("Users written to Cassandra.")

    # Read users back from Cassandra into a new DataFrame
    readUsers = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="users", keyspace=KEYSPACE) \
        .load()
    readUsers.createOrReplaceTempView("users")

    # Register ratings and movies as SQL views (used for Task iii)
    ratings_df.createOrReplaceTempView("ratings")
    movies_df.createOrReplaceTempView("movies")

    # =========================================================================
    # TASK iii — Active Users and Their Favourite Genre
    #
    # Active user = rated at least 50 movies.
    # Favourite genre = the genre they rated most frequently.
    # =========================================================================
    print("\n" + "=" * 62)
    print("TASK iii  —  Active Users' Favourite Genre (>= 50 ratings)")
    print("=" * 62)

    # Explode the genres array so each (movie_id, genre) pair is its own row
    movies_exploded = spark.sql("""
        SELECT movie_id, explode(genres) AS genre
        FROM   movies
        WHERE  genres IS NOT NULL
    """)
    movies_exploded.createOrReplaceTempView("movies_genres")

    # Identify active users
    active_users_df = spark.sql("""
        SELECT user_id
        FROM   ratings
        GROUP  BY user_id
        HAVING COUNT(*) >= 50
    """)
    active_users_df.createOrReplaceTempView("active_users")
    print("Active users (>= 50 ratings):", active_users_df.count())

    # For each active user count how many times they rated each genre,
    # then pick the genre with the highest count using ROW_NUMBER()
    fav_genre_df = spark.sql("""
        WITH genre_counts AS (
            SELECT r.user_id,
                   mg.genre,
                   COUNT(*) AS genre_count
            FROM   ratings       r
            JOIN   active_users  au ON r.user_id  = au.user_id
            JOIN   movies_genres mg ON r.movie_id = mg.movie_id
            GROUP  BY r.user_id, mg.genre
        ),
        ranked AS (
            SELECT user_id,
                   genre,
                   genre_count,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY genre_count DESC) AS rn
            FROM   genre_counts
        )
        SELECT user_id,
               genre      AS favorite_genre,
               genre_count AS genre_rating_count
        FROM   ranked
        WHERE  rn = 1
        ORDER  BY user_id
    """)

    print("Active users with favourite genre determined:", fav_genre_df.count())
    fav_genre_df.show(20, truncate=False)

    # Write to Cassandra
    fav_genre_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(table="user_fav_genres", keyspace=KEYSPACE) \
        .save()
    print("user_fav_genres written to Cassandra.")

    # Read back from Cassandra to validate
    print("\nReading user_fav_genres back from Cassandra (sample):")
    spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="user_fav_genres", keyspace=KEYSPACE) \
        .load() \
        .orderBy("user_id") \
        .show(20, truncate=False)

    # =========================================================================
    # TASK iv — All Users Younger than 20
    # =========================================================================
    print("\n" + "=" * 62)
    print("TASK iv  —  Users Under 20 Years Old")
    print("=" * 62)

    sqlDF_iv = spark.sql("""
        SELECT user_id, age, gender, occupation, zip_code
        FROM   users
        WHERE  age < 20
        ORDER  BY age ASC, user_id ASC
    """)

    sqlDF_iv.show(truncate=False)
    print("Total users under 20:", sqlDF_iv.count())

    # Write to Cassandra
    sqlDF_iv.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(table="young_users", keyspace=KEYSPACE) \
        .save()
    print("young_users written to Cassandra.")

    # Read back from Cassandra to validate
    print("\nReading young_users back from Cassandra:")
    spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="young_users", keyspace=KEYSPACE) \
        .load() \
        .orderBy("age", "user_id") \
        .show(truncate=False)

    # =========================================================================
    # TASK v — Scientists Aged 30 to 40
    # =========================================================================
    print("\n" + "=" * 62)
    print("TASK v  —  Scientists Aged 30 to 40")
    print("=" * 62)

    sqlDF_v = spark.sql("""
        SELECT user_id, age, gender, occupation, zip_code
        FROM   users
        WHERE  occupation = 'scientist'
          AND  age BETWEEN 30 AND 40
        ORDER  BY age ASC, user_id ASC
    """)

    sqlDF_v.show(truncate=False)
    print("Total scientist users aged 30-40:", sqlDF_v.count())

    # Write to Cassandra
    sqlDF_v.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(table="scientist_users", keyspace=KEYSPACE) \
        .save()
    print("scientist_users written to Cassandra.")

    # Read back from Cassandra to validate
    print("\nReading scientist_users back from Cassandra:")
    spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="scientist_users", keyspace=KEYSPACE) \
        .load() \
        .orderBy("age", "user_id") \
        .show(truncate=False)

    spark.stop()
    print("\nDone.")
