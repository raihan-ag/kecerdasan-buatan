from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import numpy as np
from scipy.spatial.distance import cosine
import re

# Load embedding model (sentence-transformers/all-MiniLM-L6-v2)
embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Load generation model (gpt2 for demo, replace with better LLM if available)
generator = pipeline('text-generation', model='gpt2')

# Load knowledge base from file
with open('router_manual.txt', 'r') as f:
    text = f.read()

# Chunking: Split into paragraphs/sentences
chunks = re.split(r'\n\n|\.\s+', text)  # Simple chunking by paragraph or sentence
chunks = [chunk.strip() for chunk in chunks if len(chunk) > 50]  # Filter short chunks

# Embedding: Create vector for each chunk
def get_embedding(text):
    emb = embedder(text)[0]
    return np.mean(emb, axis=0)  # Mean pooling

chunk_embeddings = np.array([get_embedding(chunk) for chunk in chunks])

# Retrieval function
def retrieve(query, top_k=3):
    query_emb = get_embedding(query)
    similarities = [1 - cosine(query_emb, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:]
    return [chunks[i] for i in top_indices]

# Generation with RAG
def rag_query(query):
    retrieved = retrieve(query)
    context = "\n".join(retrieved)
    prompt = f"Based on this manual: {context}\nQuestion: {query}\nAnswer accurately without hallucination:"
    response = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
    return response.split("Answer:")[-1].strip()  # Simple parse

# Example usage
print(rag_query("Bagaimana cara reset konfigurasi pabrik pada router ini?"))
print(rag_query("Mengapa lampu indikator merah pada router saya berkedip?"))