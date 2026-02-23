# The transformer
> Let me introduce you to the transformer. The transformer as you may know was first introduces in the paper 
> "Attention is all you need" from Google: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

Given by the name of the paper it's not surprising that transformers are using the attention
mechanism to process sequences in parallel.
They are Efficient in NLP (Natural language processing) like: 
- text generation
- language translation 
- sentiment analysis

> With that in mind I guess it's clear why we get to use the transformer. But let me break it down in few sentences real quick.
> Our goal is to give the user a summary, based on the wikipedia entries the model knows. We search semantically in the database, and 
> after that, the app will give us a short summary like the Google search. **Declaimer:** It will surely not that good 
> as the model from Google :)

---

# The attention mechanism
### Key Facts: 
- Technique that allows models to focus on the most important parts of input data when making preds.
- Helps model to prioritize relevant information instead of treating all inputs equally

**What it means in case of an input in real usage:**
- It improves how models handle long sequences in data.
- It helps capture relationships between distant elements in a sequence.
- Enhances interpretability by showing which parts of input influenced the output.
- Widely applied in translation, summarization, image captioning and speech processing.

---

## Types of Attention Mechanisms
- Soft Attention: Differentiable mechanism using softmax and is widely used in NLP and transformers.
- Hard Attention: Non-differentiable and uses sampling to select specific parts. It is trained using reinforcement learning.
- Self-Attention: Enables each input element to attend to other aspects in the same sequence.
- Multi-Head Attention: Uses multiple attention heads to capture diverse features from different representation subspaces.
- Additive Attention: Uses a feed-forward neural network to calculate attention scores instead of dot products.

> We use the Multi-Head Attention which is a extension of the self-attention where multiple attention
> mechanisms i.e. heads are applied in parallel. Each head learns different aspects of the input data allowing the model 
> to capture various dependencies at different levels of abstraction.
> 
> **Multi-Head Attention is the reason for the success of transformer models like BERT, GPT an others.**

---

## How does the attention work? 
There are simply 8 steps we can split that in: 

1. Input Encoding: Input sequence is encoded using an encoder like RNN, LSTM, GRU or Transformer to generate 
hidden states representing the input context.
2. **Query, Key and Value Vectors:** 
   - **Query (Q):** Represents what we're looking for.
   - **Key (K):** Represents what information each input contains.
   - **Value (V):** Contains that actual information for each input.

    There are linear transforming of the input embeddings (activation comes later on.)

---

## Links
> **Attention Mechanism in ML:** https://www.geeksforgeeks.org/artificial-intelligence/ml-attention-mechanism/
> 
> **Transformer using PyTorch:** https://www.geeksforgeeks.org/deep-learning/transformer-using-pytorch/
> 
> **build-nanopgt:** https://github.com/karpathy/build-nanogpt/tree/master
> 
> **Video to build-nanogpt:** https://youtu.be/kCc8FmEb1nY