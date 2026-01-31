import streamlit as st
from transformers import pipeline
import numpy as np
from scipy.spatial.distance import cosine
import re

# Load embedding model
embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Load generation model (gpt2 for demo; ganti dengan OpenAI jika ada API key)
generator = pipeline('text-generation', model='gpt2')

# Load knowledge base
with open('router_manual.txt', 'r') as f:
    text = f.read()

# Chunking
chunks = re.split(r'\n\n|\.\s+', text)
chunks = [chunk.strip() for chunk in chunks if len(chunk) > 50]

# Embeddings
def get_embedding(text):
    emb = embedder(text)[0]
    return np.mean(emb, axis=0)

chunk_embeddings = np.array([get_embedding(chunk) for chunk in chunks])

# Retrieval
def retrieve(query, top_k=3):
    query_emb = get_embedding(query)
    similarities = [1 - cosine(query_emb, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:]
    return [chunks[i] for i in top_indices]

# RAG Query
def rag_query(query):
    retrieved = retrieve(query)
    context = "\n".join(retrieved)
    prompt = f"Based on this manual: {context}\nQuestion: {query}\nAnswer accurately without hallucination:"
    response = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
    return response.split("Answer:")[-1].strip()

# Streamlit Dashboard
st.title("Smart Manual Support: TP-Link Archer C80")
st.write("Masukkan pertanyaan tentang user manual router, dan sistem RAG akan jawab berdasarkan dokumen referensi.")

query = st.text_input("Pertanyaan Anda:")
if st.button("Tanya"):
    if query:
        with st.spinner("Memproses..."):
            answer = rag_query(query)
        st.success("Jawaban:")
        st.write(answer)
        st.info("Context yang Diretrieve:")
        st.write("\n".join(retrieve(query)))  # Tampilkan chunks relevan untuk demo transparansi
    else:
        st.warning("Masukkan pertanyaan terlebih dahulu.")