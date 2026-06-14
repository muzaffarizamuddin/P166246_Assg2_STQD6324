import csv
import pymongo

client = pymongo.MongoClient("localhost", 27017)
db = client["movielens_exclusion_db"]
collection = db["least_liked_movies"]

# Fetch all records
cursor = collection.find({})

# Open a comprehensive flat CSV file
with open("exported_all_exclusions.csv", "w") as f:
    writer = csv.writer(f)
    # Define clean, flat headers
    writer.writerow(["user_id", "favorite_genre", "avoid_rank", "movie_id", "title", "global_avg_rating"])
    
    for doc in cursor:
        uid = doc.get("user_id")
        genre = doc.get("favorite_genre")
        # Loop through all 5 items nested inside the MongoDB document array
        for movie in doc.get("blacklisted_movies", []):
            writer.writerow([
                uid,
                genre,
                movie.get("avoid_rank"),
                movie.get("movie_id"),
                movie.get("title"),
                movie.get("global_avg_rating")
            ])

print("Successfully exported all 5 records per user to exported_all_exclusions.csv!")
client.close()