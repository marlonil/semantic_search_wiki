from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

with open('test.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=4))
y = json.loads(json.dumps(data, indent=4))
text = y["sections"][2]["has_parts"][1]["value"]

lenght = len(text)
print(f"lenght: {lenght}")

word = (i.strip() for i in text.split("."))

sentence = []
for w in word:
    sentence.append(w)

print(f"sentence: {sentence}")

embeddings = model.encode(sentence)
print(f"Embeddings Shape: {embeddings.shape}")

print("-"*30)

similarities = model.similarity(embeddings, embeddings)
print(similarities)

