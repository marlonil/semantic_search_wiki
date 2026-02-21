from sentence_transformers import SentenceTransformer
import json
import chromadb as db
from langchain_text_splitters import CharacterTextSplitter

PRESTIGE_DIR = "content/chroma_db"

chroma_client = db.PersistentClient(path=PRESTIGE_DIR)
collection = chroma_client.get_or_create_collection(name='wiki_entries')

def recursive_search(json_data: dict, target_key: str):
    """Recursively search a dict for target_key

    Inputs: dict out of a json with dicts, lists and basic data types.
    Outputs: ["value1", "value2", "value3"] -> The values of the keys we target
    """
    recursive_results = [] # -> initializes the results (values of the targeted key)
    if isinstance(json_data, dict): # -> checks if the json_data is a dict, if so we will become a true and statement is finished
        for key, value in json_data.items():
            if key == target_key: # -> checks if the key is the targeted key, we search for
                recursive_results.append(value) # if so, we will add it to the results
            recursive_results.extend(recursive_search(value, target_key)) # -> if not we just extend the results with a new iteration will calling the function we currently in. Why extend? If we would add, we would create a empty list which would overrwite everything
    elif isinstance(json_data, list): # -> we check if the current json data is a list
        for value in json_data:
            recursive_results.extend(recursive_search(value, target_key)) # -> we add every item to the list and call the function again to add the value finally to the results.
# the recall of the function inside the function will cause that we create like a recursive list which will search deeper in the json and also add everything found in one "tree"
    elif isinstance(json_data, (int, float, bool, str)):
        pass# print(f"3. Check (basis-typ) json_data: {json_data}")

    return recursive_results

model = SentenceTransformer("all-MiniLM-L6-v2")

with open('data/enwiki_namespace_0/enwiki_namespace_0_0.jsonl', 'r', encoding='utf-8') as input_file:
    counter = 0
    for line in input_file:
        counter += 1
        data = json.loads(line)
        raw_recursive = recursive_search(data, "value")

        #print(f"Artikel identifier: {data["identifier"]}")
        #print(f"Artikel name: {data['name']}")
        #print(f"Artikel url: {data['url']}")

        clean_sentences = []
        for sentence in raw_recursive:
            if isinstance(sentence, str):
                clean_sentences.append(sentence)

        text_string = " ".join(clean_sentences)

        if len(text_string) < 300:
            continue
        #print(f'text_string: {text_string}')

        splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_text(text_string)

        #print(f"chunks: {chunks}")

        embeddings = model.encode(chunks)
        #print(f"Embeddings Shape: {embeddings.shape}")

        #print("-" * 30)

        batch_chunk_embeddings = []
        batch_chunk_texts = []
        batch_ids = []
        batch_metadatas = []
        for chunk in range(0, len(chunks)):
            chunk_text = chunks[chunk]
            chunk_embedding = embeddings[chunk]
            chunk_id = f"{data['identifier']}_{chunk}"
            metadata = {"artikel_id": data["identifier"], "title": data["name"], "url": data["url"], "chunk_index": chunk}
            batch_chunk_embeddings.append(chunk_embedding), batch_chunk_texts.append(chunk_text), batch_ids.append(chunk_id), batch_metadatas.append(metadata)
        collection.add(
            embeddings= batch_chunk_embeddings,
            documents= batch_chunk_texts,
            ids= batch_ids,
            metadatas= batch_metadatas,
        )

        #similarities = model.similarity(embeddings, embeddings)
        #print(similarities)

        if counter > 100:
            break
