from pymongo import MongoClient
import os

def search_vectors(query_vector, collection_name="vector_data"):
    client = MongoClient(os.getenv("MONGODB_ATLAS_URI"))
    db = client["ai_database"]
    collection = db[collection_name]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 10
            }
        }
    ]

    results = collection.aggregate(pipeline)
    return list(results)

if __name__ == "__main__":
    print("Atlas Vector Search module initialized.")
