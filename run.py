from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm

# Step 1: Load embedding model (placeholder for Qwen3)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")  # Replace with actual model path if local
#model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")  # Replace with actual model path if local

# Step 2: Load textbook documents from EPUB/Markdown/Text
docs = SimpleDirectoryReader(input_dir="/app/textbook").load_data()
print("✅ Loaded document count:", len(docs))

# Step 3: Chunk the text (approx. 512 tokens per chunk)
print("✂️ Chunking documents into manageable pieces...")
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(docs)
chunks = [n.text for n in nodes]
print(f"📚 Total chunks generated: {len(chunks)}")

# Step 4: Batch embed chunks to avoid memory overload
def batch_encode(texts, model, batch_size=8):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔄 Embedding chunks"):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)

print("🚀 Starting embedding...")
embeddings = batch_encode(chunks, model)
dimension = embeddings.shape[1]

# Step 5: Build FAISS index
print("🗂️ Building FAISS index...")
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings))
print("✅ FAISS index built with", index.ntotal, "vectors.")

# Step 6: Interactive query loop
print("\n💬 Enter your query below. Type 'exit' to quit. Type 'vllm' to export final context for your LLM prompt.\n")

retained_context = []

while True:
    query = input("🔎 Your query: ").strip()
    if query.lower() == "exit":
        break
    elif query.lower() == "vllm":
        print("\n📤 Final context for vLLM prompt:\n")
        print("\n--- CONTEXT START ---\n")
        print("\n".join(retained_context))
        print("\n--- CONTEXT END ---\n")
        continue

    query_embedding = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_embedding, k=3)

    results = [chunks[i] for i in I[0]]
    retained_context.extend(results)

    print("\n📚 Retrieved Chunks:")
    for i, chunk in enumerate(results):
        print(f"--- Chunk #{i+1} ---")
        print(chunk[:1000], "\n")


