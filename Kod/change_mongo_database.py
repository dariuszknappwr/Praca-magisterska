from pymongo import MongoClient

# Create a client connection to your MongoDB server
client = MongoClient('mongodb://localhost:27017/')

# Connect to your database
db = client['PracaMagisterska']

# Connect to your collection
collection = db['Test1']

# Rename the field
collection.update_many({}, {'$rename': {"A* Algorithm v Average Speed": "A* Algorithm Haversine Average Speed"}})