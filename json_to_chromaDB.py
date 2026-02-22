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

with open('data/enwiki_namespace_0/enwiki_namespace_0_0.jsonl', 'r', encoding='utf-8') as input_file: # file I want to iterate through
    counter = 0 # for printing processed documents or break after 3 for a test / debugging
    for line in input_file:
        counter += 1
        data = json.loads(line) # creating a dict out of the json line in document
        raw_recursive = recursive_search(data, "value")  # extracting the text out of every wikipedia entire

        clean_sentences = [] # -> collection of sentences
        # This loop is needed because there are often integers, bools or float numbers into a entire we dont care about
        for sentence in raw_recursive:
            if isinstance(sentence, str): # -> only getting the strings (real data) to dont put garbage into the database
                clean_sentences.append(sentence) # -> appending the "real sentence"

        text_string = " ".join(clean_sentences) # -> putting all sentences together because we can only chunk on strings and no lists with strings included

        # we skip sentences that aren't relevant like "REDIRECT List of Suits characters#Mike Ross" If we would add something like this the UX would be bad because the database would maybe give the user such a text on asking for Mike Ross
        if len(text_string) < 300: # -> if the text of all sentences is smaller than 300 it will be skipped
            continue

        splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50) # -> initializing the model used for the chunks with a defined chunk size of 512 which is a common size but can be in or decreased
        chunks = splitter.split_text(text_string) # chunking the text. The used chunking framework will provide that sentences get splitted on a way that makes sense for ML models

        embeddings = model.encode(chunks) # encoding the chunks

        # creating lists for loading the batch into the database
        batch_chunk_embeddings = []
        batch_chunk_texts = []
        batch_ids = []
        batch_metadatas = []
        # iterating through the chunks to load them separately into the database which is best practice
        for chunk in range(0, len(chunks)):
            chunk_text = chunks[chunk] # -> getting the text of the chunk
            chunk_embedding = embeddings[chunk] # -> getting the embeddings (float numbers [0.1201, 0.7213, ...] of the chunk
            chunk_id = f"{data['identifier']}_{chunk}" # defining the chunk_id used in the database out of the artikel identifier like e.g. 71273671 with the number of the chunk
            metadata = {"artikel_id": data["identifier"], "title": data["name"], "url": data["url"], "chunk_index": chunk} # -> defining the metadata
            batch_chunk_embeddings.append(chunk_embedding), batch_chunk_texts.append(chunk_text), batch_ids.append(chunk_id), batch_metadatas.append(metadata) # -> adding all data to the respective list
        # adding the 'batch' to the database
        collection.add(
            embeddings= batch_chunk_embeddings,
            documents= batch_chunk_texts,
            ids= batch_ids,
            metadatas= batch_metadatas,
        )
        counter += 1
        data = json.loads(line) # creating a dict out of the json line in document
        raw_recursive = recursive_search(data, "value")  # extracting the text out of every wikipedia entire

        clean_sentences = [] # -> collection of sentences
        # This loop is needed because there are often integers, bools or float numbers into a entire we dont care about
        for sentence in raw_recursive:
            if isinstance(sentence, str): # -> only getting the strings (real data) to dont put garbage into the database
                clean_sentences.append(sentence) # -> appending the "real sentence"

        text_string = " ".join(clean_sentences) # -> putting all sentences together because we can only chunk on strings and no lists with strings included

        # we skip sentences that aren't relevant like "REDIRECT List of Suits characters#Mike Ross" If we would add something like this the UX would be bad because the database would maybe give the user such a text on asking for Mike Ross
        if len(text_string) < 300: # -> if the text of all sentences is smaller than 300 it will be skipped
            continue

        splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50) # -> initializing the model used for the chunks with a defined chunk size of 512 which is a common size but can be in or decreased
        chunks = splitter.split_text(text_string) # chunking the text. The used chunking framework will provide that sentences get splitted on a way that makes sense for ML models

        embeddings = model.encode(chunks) # encoding the chunks

        # creating lists for loading the batch into the database
        batch_chunk_embeddings = []
        batch_chunk_texts = []
        batch_ids = []
        batch_metadatas = []
        # iterating through the chunks to load them separately into the database which is best practice
        for chunk in range(0, len(chunks)):
            chunk_text = chunks[chunk] # -> getting the text of the chunk
            chunk_embedding = embeddings[chunk] # -> getting the embeddings (float numbers [0.1201, 0.7213, ...] of the chunk
            chunk_id = f"{data['identifier']}_{chunk}" # defining the chunk_id used in the database out of the artikel identifier like e.g. 71273671 with the number of the chunk
            metadata = {"artikel_id": data["identifier"], "title": data["name"], "url": data["url"], "chunk_index": chunk} # -> defining the metadata
            batch_chunk_embeddings.append(chunk_embedding), batch_chunk_texts.append(chunk_text), batch_ids.append(chunk_id), batch_metadatas.append(metadata) # -> adding all data to the respective list
        # adding the 'batch' to the database
        collection.add(
            embeddings= batch_chunk_embeddings,
            documents= batch_chunk_texts,
            ids= batch_ids,
            metadatas= batch_metadatas,
        )

        # will print out the processed articles and the counting of the collection chunks for tracking
        if counter % 1000 == 0:
            print(f"Processed {counter} articles")
            print(f"Total chunks in DB: {collection.count()}")

        #similarities = model.similarity(embeddings, embeddings)
        #print(similarities)

        #if counter > 100:
            #break

        if counter % 1000 == 0:
            print(f"Processed {counter} articles")
            print(f"Total chunks in DB: {collection.count()}")

        #similarities = model.similarity(embeddings, embeddings)
        #print(similarities)

        #if counter > 100:
            #break