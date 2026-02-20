# Architecture of the project

---

#### On highlevel the project got like **4 main** system "parts". 

> 1. Frontend
> 2. Backend + Deep Learning Architecture
> 3. Database
> 4. Deep Learning Part (which is I know in the Backend but for me a seprate system part.)

## Deep dive Frontend: 

---

> ### Content for frontend here...

## Deep dive backend:

---

> ### Content here...

## Deep dive Database:

---

> ### Content here...

## Deep dive AI:

---

> #### In this part we need to separate between the semantic search and the Tiny-LLM archi.
>
> 
> ### 1. Semantic search
> - The semantic search itself is mainly part of the Database. I guess Chroma DB comes with a already implemented search. That means after the embedding there is just the mapping from user input to a embedding with that you can search through the database. If this is not the case I implement a **KNN**.
> - So that means for the semantic search I need to build a data pipeline that transforms the JSON from the real dataset into a vector embedding ready for the chroma db database. 
>   - The information will be extracted out of the nested json with a recursive key search
>    - For the embedding I use the all-MiniLM-L6-v2 out of the sentence-transformer framework in Python. The embedding model is designed to map sentences and paragraph's into a 384-dimensional dens vector space. **Key Features:** {"Dimensionality": "Outputs 383-dimensional embeddings", "Application": "Ideal for tasks like information retrieval, clustering, and semantic similarity", "input limit": "Text longer than 256 word pieces is truncated by default", "Training": "Fine-tuned using datasets like Reddit comments, WikiAnswers, and Stack Exchange, among others"}
> 
> ### 2. Tiny-LLM
> - An LLM will not be trained on embeddings, therefore we separate those both parts. The outputs of the semantic search are others than we need for an LLM. **Why?** Because the sentence transformer will represent the semantic meaning of the text in numbers. **What does the LLM need?** A LLM learns his own embeddings (nn.Embeddings - PyTorch) out of the raw data. Those are two complete different things. 
> #### How does the LLM works internally?
> ``` 
> "Hallo World"
>    ↓  Tokenizer
>[15496, 995]          ← Integer Token-IDs
>    ↓  nn.Embedding (Lookup-Tabel)
>[[0.1, 0.3, -0.2, ...],
> [0.5, -0.1, 0.8, ...]]  ← Dense Vektoren per token
>    ↓  Transformer Layers
>    ↓  Output Logits → next token guessing
>```
>  - For tokenization, we want to use **subword-tokenization like BPE - Byte Pair Encoding**. BPE starts with a simple byte/char as basis vocabulary and merges iterative the most char-pairs to the new tokens
> 
>``` 
> "training"  →  ["t", "r", "a", "i", "n", "i", "n", "g"]
>              →  ["tr", "a", "in", "in", "g"]
>              →  ["train", "ing"]          ← two tokens
> ```