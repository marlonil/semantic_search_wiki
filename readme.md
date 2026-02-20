# Semantic Search + Tiny-LLM

---

## What do I build?

---

> Simply a search as you may know from Google search or something like that. 
> **But what is behind a search?** With the development of the current AI movement, there are many AI components in a simply looking search. We have the keywords combined with the semantic search that will help the user finding more complex asked questions and at latest there is this cool AI summary to the users question. 
As you can see, there is more behind a simple search than just a database
 
## Reason for the project:

---

> **Personal reason:** I want to understand Deep Learning and LLM's on a deeper level
- I learn how a system architecture is build!
- I build an understanding of different ways to search in a lets say embedded room full of vektors
- It teaches me a understanding of how a LLM is build and what the magic Transformer really is
- I understand Vektor Databases like ChromaDB which I will use
- I learn more deeper how Forntend and Backend need to work together

## The goal of the project!

---

> #### A MVP that can search through WIKIPEDIA articels and understand the users intention and give a summary of what the Users question was out of the Data stored in the DB (Metadata like links and the summary).  

### How to I achive this goal?
> - [] I collect data (Wikipedia artikels as a JSON)
> - [] Embedding of the articels into the database (Metadata, and content as vectors)
> - [] Build up the frontend 
> - [] Initizilize the backend and build up a semantic search only
> - [] Training of the Tiny-LLM based on Wikipedia artikels
> - [] Implemeting of the Tiny-LLM into Backend and build a answer for user in markdown format


## Techstack!

---

> - **Fronend:** HTML, CSS, JavaScript (React)
> - **Backend:** Python with FastAPI - maybe extend to C++ for optimization
> - **Database:** Chroma DB - Vekotdatabase made for large language models
> - **Data Science & Deep Learning Frameworks:** Pandas, PyTorch, Sentance Transformer