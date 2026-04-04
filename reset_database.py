import os

files_to_delete = [
    "person_embeddings.pkl",
    "cafe_data.json"
]

for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)
        print(f"✅ Deleted {file}")
    else:
        print(f"⚠️ {file} not found")

print("\n🔄 Database reset complete!")
print("Run 'streamlit run cafe_analytics.py' to start fresh")
