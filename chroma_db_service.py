from sentence_transformers import SentenceTransformer
import chromadb as db

PRESTIGE_DIR = "content/chroma_db" # path where the computer saves the database permanently

# creates client with permanently saving on hardware disk
chroma_client = db.PersistentClient(path=PRESTIGE_DIR)
# Get an existing collection in Prestige_dir or creates a new with the name 'wiki_entries'
collection = chroma_client.get_or_create_collection(name='wiki_entries')

# initializing the model (class) used for embedding the data into the database
model = SentenceTransformer("all-MiniLM-L6-v2")

# checking the health status of the database
print(f"Total chunks: {collection.count()}")
collection.peek()

# semantic search tests
queries = ["student protests", "obama", "mercedes"]
# iterating through the queries (simulating the laters users input)
for q in queries:
    # creating the query
    results = collection.query(
        model.encode(q), # -> encoding the input with the above initialized model
        n_results=3, # -> defining the results per query for one input to not get to much
        include=["documents", "metadatas", "embeddings"] # -> defining what we want to include from the output
    )
    # printing the results
    print(q, results)

# get and print the first 3 database entries
sample = collection.get(limit=3)
print(sample['metadatas'])
