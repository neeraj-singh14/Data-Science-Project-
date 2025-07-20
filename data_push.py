
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
import os

client = MongoClient("mongodb://localhost:27017/")

# 📂 Select your database and collection
db = client["test_db"]  # Replace with your database name

table_type = input("s: student_response or d: scoring_DSAT -->")
if table_type == "s":
    collection = db["student_response"]  # Replace with your collection name
elif table_type == "d":
    collection = db["scoring_DSAT"]
else:
    print("None Selected!!!!!!")

path_file = input("Put path or file name: ")

if path_file.endswith(".json"):
    with open(path_file, "r") as f:
        records = json.load(f)
    for record in records:
        record.pop('_id', None)
    # 📨 Insert into MongoDB
    if records:
        collection.insert_many(records)
        print(f"✅ Inserted {len(records)} documents into MongoDB.")
    else:
        print("⚠️ No data found to insert.")

else:
    try: 
        contents = os.listdir(path_file)
        
        for file in contents:
            final_file = f"{path_file}/{file}"
            with open(final_file, "r") as f:
                records = json.load(f)
            for record in records:
                record.pop('_id', None)

            # 📨 Insert into MongoDB
            if records:
                collection.insert_many(records)
                print(f"✅ Inserted {len(records)} documents into MongoDB.")
            else:
                print("⚠️ No data found to insert.")
    except:
        print("Worng path of directory, please put directory path or json file!!!")
    