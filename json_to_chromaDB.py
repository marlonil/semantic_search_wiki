from sentence_transformers import SentenceTransformer
import json
import ijson

"""Needs!

-> Helperfunction: The function will go through every json, every time it findes a field names value, it saves the text. 

-> memory crash fixing (json.load will crash my memory with 70gb jsons
    -> Streaming (ijson): Opens file, Takes everything out of one artikel and extrects the text, 
    chunks it embed it and saves it in db. So all in one loop

Reminder: ijson walkes through the hole json file and just takes one object. 

-> I need to load 3 sentences at once in the place of one index
"""

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
        print(f"3. Check (basis-typ) json_data: {json_data}")

    return recursive_results

def process_an_object(v1, v2):
    print(f"v1 = {v1}, \n v2 = {v2}")
model = SentenceTransformer("all-MiniLM-L6-v2")

with open('test.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=4))
y = json.loads(json.dumps(data, indent=4))

raw_recursive = recursive_search(y, "value")

for sentence in raw_recursive:
    if isinstance(sentence, (int, float)):
        raw_recursive.remove(sentence)

results = []
for sentence in range(0, len(raw_recursive), 3):
    # searches every sentence from 0 to the length of all sentences in raw_recursive, but only 3 at once.
    block = raw_recursive[sentence:sentence+3]
    joined_block = ". ".join(block)
    results.append(joined_block)
print(f"Results: {results}")

embeddings = model.encode(results)
print(f"Embeddings Shape: {embeddings.shape}")

print("-"*30)

similarities = model.similarity(embeddings, embeddings)
print(similarities)

