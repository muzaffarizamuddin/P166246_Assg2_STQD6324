# -*- coding: utf-8 -*-
# =============================================================================
# hbase_analysis.py
# Assignment 2 | STQD6324 | P166246 | Muzaffar Izamuddin
#
# HBase Optional Extension — Genre-Based Movie Recommendations
#
# For each user:
#   1. Find their favourite genre (genre they rated most)
#   2. Rank unrated movies in that genre by global avg rating (min 5 ratings)
#   3. Store top-5 recommendations in HBase (wide-column row per user)
#   4. Query any user_id to retrieve their personalised recommendation list
#
# Why HBase suits this task:
#   - Wide-column model: one row per user, N columns for N recommendations
#   - O(1) key lookup by user_id — no full table scan needed
#   - Contrasts with Cassandra CQL approach in the main scripts
#
# Prerequisites:
#   1. HBase Master running  : hbase-daemon.sh start master
#   2. Thrift server running : hbase-daemon.sh start thrift
#   3. Python driver         : pip install happybase
#
# Run:
#   spark-submit hbase/hbase_analysis.py
# =============================================================================

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import explode, col, desc, row_number
from pyspark.sql.window import Window
import happybase

# =============================================================================
# CONFIGURATION
# =============================================================================
HDFS_BASE   = "hdfs:///user/maria_dev/ml-100k"
HBASE_HOST  = "localhost"
HBASE_PORT  = 9090
TABLE_NAME  = "movielens_recommendations"
MIN_RATINGS = 5    # minimum global ratings for a movie to qualify
TOP_N       = 5    # recommendations per user

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

# =============================================================================
# HBASE HELPERS
# =============================================================================
def setup_hbase_table(connection):
    existing = [t.decode() for t in connection.tables()]
    if TABLE_NAME in existing:
        print("Dropping existing table:", TABLE_NAME)
        connection.delete_table(TABLE_NAME, disable=True)
    connection.create_table(TABLE_NAME, {"recommendations": dict()})
    print("Created HBase table '%s'  (column family: recommendations)" % TABLE_NAME)


def write_recommendations(connection, recs_list):
    """
    Each HBase row  = one user  (row key: user_000001)
    Each HBase col  = one ranked recommendation:
        recommendations:rank_1_movie_id
        recommendations:rank_1_title
        recommendations:rank_1_avg_rating
        ... up to rank_5
    recs_list is a list of Spark Row objects (from recs_df.collect()).
    """
    table = connection.table(TABLE_NAME)
    with table.batch(batch_size=500) as batch:
        for row in recs_list:
            uid  = int(row.user_id)
            rk   = int(row.rn)
            rkey = "user_{:06d}".format(uid).encode()
            batch.put(rkey, {
                ("recommendations:rank_%d_movie_id"   % rk).encode(): str(int(row.movie_id)).encode(),
                ("recommendations:rank_%d_title"      % rk).encode(): str(row.title).encode(),
                ("recommendations:rank_%d_avg_rating" % rk).encode(): "{:.4f}".format(float(row.avg_rating)).encode(),
            })

    stored = sum(1 for _ in table.scan(columns=[]))
    print("Recommendations stored in HBase for %d users." % stored)


def lookup_user(table, user_id):
    """Return recommendations list for a single user by HBase key lookup."""
    rkey = "user_{:06d}".format(user_id).encode()
    row  = table.row(rkey)
    if not row:
        return None
    recs = []
    for rank in range(1, TOP_N + 1):
        title_key = ("recommendations:rank_%d_title"      % rank).encode()
        rate_key  = ("recommendations:rank_%d_avg_rating" % rank).encode()
        mid_key   = ("recommendations:rank_%d_movie_id"   % rank).encode()
        if title_key in row:
            recs.append({
                "rank":       rank,
                "movie_id":   int(row[mid_key].decode()),
                "title":      row[title_key].decode(),
                "avg_rating": row[rate_key].decode(),
            })
    return recs


def print_recs(user_id, recs):
    print("\nUser %d — Top-%d Recommendations" % (user_id, TOP_N))
    print("-" * 62)
    if not recs:
        print("  No recommendations found.")
        return
    print("  {:<5} {:<42} {}".format("Rank", "Movie Title", "Avg Rating"))
    print("  " + "-" * 58)
    for r in recs:
        print("  {:<5} {:<42} {}".format(r["rank"], r["title"][:41], r["avg_rating"]))

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import csv
    import os
    from collections import namedtuple

    CSV_PATH = "/home/maria_dev/recommendations.csv"

    # RecRow lets CSV-loaded data use the same attribute access as Spark Rows
    RecRow = namedtuple("RecRow", ["user_id", "rn", "movie_id", "title", "avg_rating", "fav_genre"])

    spark = None  # only set if Spark phase runs

    if os.path.exists(CSV_PATH):
        # -------------------------------------------------------------------------
        # Fast path: CSV already exists from a previous run — skip Spark entirely.
        # This avoids memory pressure that causes the HBase Master to crash.
        # -------------------------------------------------------------------------
        print("CSV found at %s — skipping Spark computation." % CSV_PATH)
        recs_list = []
        with open(CSV_PATH, "r") as f:
            for r in csv.DictReader(f):
                recs_list.append(RecRow(
                    user_id=int(r["user_id"]),
                    rn=int(r["rank"]),
                    movie_id=int(r["movie_id"]),
                    title=r["title"],
                    avg_rating=float(r["avg_rating"]),
                    fav_genre=r["fav_genre"]
                ))
        print("Loaded %d recommendation rows from CSV." % len(recs_list))

    else:
        # -------------------------------------------------------------------------
        # Full Spark pipeline (Steps 1-5)
        # Runs only on first execution; results are cached to CSV.
        # -------------------------------------------------------------------------

        # Step 1: Load raw data from HDFS into RDDs
        spark = (SparkSession.builder
                 .appName("MovieLens_HBase_Recommendations_P166246")
                 .config("spark.eventLog.enabled", "false")
                 .getOrCreate())
        sc = spark.sparkContext
        sc.setLogLevel("WARN")
        print("Spark version:", spark.version)

        ratings_rdd = sc.textFile(HDFS_BASE + "/u.data").map(parse_rating)
        movies_rdd  = sc.textFile(HDFS_BASE + "/u.item").map(parse_movie)
        print("Ratings loaded from HDFS:", ratings_rdd.count())
        print("Movies  loaded from HDFS:", movies_rdd.count())

        # Step 2: Convert RDDs to DataFrames
        ratings_df = spark.createDataFrame(ratings_rdd)
        movies_df  = spark.createDataFrame(movies_rdd)

        movies_genres_df = movies_df.select(
            "movie_id", "title", explode("genres").alias("genre")
        )
        ratings_df.createOrReplaceTempView("ratings")
        movies_genres_df.createOrReplaceTempView("movies_genres")

        print("\nMovies DataFrame schema:")
        movies_df.printSchema()

        # Step 3: Each user's favourite genre (most-rated genre)
        fav_genre_df = spark.sql("""
            WITH genre_counts AS (
                SELECT r.user_id, mg.genre, COUNT(*) AS cnt
                FROM ratings r
                JOIN movies_genres mg ON r.movie_id = mg.movie_id
                GROUP BY r.user_id, mg.genre
            ),
            ranked AS (
                SELECT user_id, genre,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY cnt DESC) AS rn
                FROM genre_counts
            )
            SELECT user_id, genre AS fav_genre FROM ranked WHERE rn = 1
        """)
        fav_genre_df.createOrReplaceTempView("fav_genres")
        print("\nFavourite genre computed for", fav_genre_df.count(), "users.")
        fav_genre_df.show(5)

        # Step 4: Global average rating per (genre, movie) with min ratings threshold
        top_by_genre_df = spark.sql("""
            SELECT mg.genre, mg.movie_id, mg.title,
                   ROUND(AVG(r.rating), 4) AS avg_rating,
                   COUNT(*) AS rating_count
            FROM ratings r
            JOIN movies_genres mg ON r.movie_id = mg.movie_id
            GROUP BY mg.genre, mg.movie_id, mg.title
            HAVING COUNT(*) >= {min_r}
        """.format(min_r=MIN_RATINGS))
        top_by_genre_df.createOrReplaceTempView("top_by_genre")

        # Step 5: Top-N unrated movies per user in their favourite genre
        already_rated_df = spark.sql("SELECT DISTINCT user_id, movie_id FROM ratings")
        already_rated_df.createOrReplaceTempView("already_rated")

        candidates_df = spark.sql("""
            SELECT fg.user_id, tbg.movie_id, tbg.title, tbg.avg_rating, fg.fav_genre
            FROM fav_genres fg
            JOIN top_by_genre tbg ON fg.fav_genre = tbg.genre
            LEFT JOIN already_rated ar
                ON fg.user_id = ar.user_id AND tbg.movie_id = ar.movie_id
            WHERE ar.movie_id IS NULL
        """)
        candidates_df.createOrReplaceTempView("candidates")

        window_spec = Window.partitionBy("user_id").orderBy(desc("avg_rating"))
        recs_df = (candidates_df
                   .withColumn("rn", row_number().over(window_spec))
                   .filter(col("rn") <= TOP_N)
                   .orderBy("user_id", "rn"))

        spark_rows = recs_df.collect()
        print("\nTotal recommendation rows computed:", len(spark_rows))
        print("Sample (first 10 rows):")
        print("  {:<8} {:<4} {:<42} {:<8} {}".format("user_id", "rn", "title", "avg_rat", "fav_genre"))
        print("  " + "-" * 72)
        for r in spark_rows[:10]:
            print("  {:<8} {:<4} {:<42} {:<8} {}".format(
                r.user_id, r.rn, r.title[:41], round(r.avg_rating, 4), r.fav_genre))

        # Save to CSV so re-runs skip Spark
        with open(CSV_PATH, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "rank", "movie_id", "title", "avg_rating", "fav_genre"])
            for r in spark_rows:
                writer.writerow([r.user_id, r.rn, r.movie_id, r.title, round(r.avg_rating, 4), r.fav_genre])
        print("Recommendations saved to: %s" % CSV_PATH)

        # Convert to RecRow namedtuples for uniform access below
        recs_list = [RecRow(r.user_id, r.rn, r.movie_id, r.title, r.avg_rating, r.fav_genre)
                     for r in spark_rows]

        spark.stop()
        spark = None

    # =========================================================================
    # Step 6: Write recommendations to HBase
    # =========================================================================
    connection = happybase.Connection(HBASE_HOST, port=HBASE_PORT)
    connection.open()
    print("\nHBase connection: %s:%d" % (HBASE_HOST, HBASE_PORT))

    setup_hbase_table(connection)
    write_recommendations(connection, recs_list)

    # =========================================================================
    # DEMO — Query recommendations for sample users
    # =========================================================================
    table = connection.table(TABLE_NAME)

    print("\n" + "=" * 62)
    print("DEMO — Recommendations for Sample Users")
    print("(HBase key lookup — O(1), no full table scan)")
    print("=" * 62)

    for uid in [1, 50, 100, 200, 500]:
        recs = lookup_user(table, uid)
        print_recs(uid, recs)

    # =========================================================================
    # QUERY A SPECIFIC USER
    # Change QUERY_USER_ID below to look up any user (1 to 943)
    # =========================================================================
    QUERY_USER_ID = 1   # <-- change this to any user_id

    print("\n" + "=" * 62)
    print("DIRECT LOOKUP — User %d" % QUERY_USER_ID)
    print("HBase row key: user_%06d" % QUERY_USER_ID)
    print("=" * 62)

    recs = lookup_user(table, QUERY_USER_ID)
    if recs:
        print("  {:<5} {:<42} {}".format("Rank", "Movie Title", "Avg Rating"))
        print("  " + "-" * 58)
        for r in recs:
            print("  {:<5} {:<42} {}".format(r["rank"], r["title"][:41], r["avg_rating"]))
    else:
        print("  No data for user %d." % QUERY_USER_ID)

    # =========================================================================
    # Spot-check: read one raw HBase row to show column structure
    # =========================================================================
    print("\n" + "=" * 62)
    print("HBase raw row structure — user_000001")
    print("=" * 62)
    raw = table.row(b"user_000001")
    for col_name, val in sorted(raw.items()):
        print("  %-48s = %s" % (col_name.decode(), val.decode()))

    connection.close()
    print("\nDone.")
