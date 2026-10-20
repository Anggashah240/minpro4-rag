import os
import re
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. INISIALISASI FASTAPI
app = FastAPI(title="Minpro4 RAG Assistant API")

# Model request JSON untuk endpoint
class QueryRequest(BaseModel):
    question: str

# 2. INISIALISASI DIREKTORI & DATASET
DATA_DIR = Path("data")
POLICIES_DIR = DATA_DIR / "policies"

if not POLICIES_DIR.exists():
    raise FileNotFoundError(f"Folder {POLICIES_DIR.resolve()} tidak ditemukan! Pastikan folder 'data/policies' sudah dibuat.")

loaded_files = list(POLICIES_DIR.glob("*.md"))
print("=" * 65)
print("Step 1 Selesai: DATASET YANG DIPILIH")
print("=" * 65)
print(f"Lokasi Direktori : {POLICIES_DIR.resolve()}")
print(f"Jumlah Dokumen   : {len(loaded_files)} file kebijakan (.md):\n")
for f in sorted(loaded_files):
    print(f"  - {f.name} ({len(f.read_text(encoding='utf-8'))} karakter)")
print("=" * 65)

# 3. CHUNKING (MEMOTONG DOKUMEN)
chunks = []
for f in sorted(POLICIES_DIR.glob("*.md")):
    text = f.read_text(encoding="utf-8")
    title = f.stem.replace("-", " ").title()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    body = " ".join([l for l in lines if not l.startswith("#")])
    if body:
        chunks.append({
            "chunk_id": f"{f.stem}-c0",
            "title": title,
            "source": f.name,
            "content": body
        })

print(f"\nStep 2 Selesai: Dihasilkan {len(chunks)} potongan chunk terstruktur.")

# 4. LOCAL VECTOR STORE & INDEXING
class LocalVectorStore:
    def __init__(self, documents):
        self.documents = documents
        vocab = sorted(list(set(" ".join([d["content"].lower() for d in self.documents]).split())))
        self.vocab = {w: i for i, w in enumerate(vocab)}
        vectors = []
        for d in self.documents:
            vec = np.zeros(len(vocab), dtype=np.float32)
            for w in d["content"].lower().split():
                if w in self.vocab:
                    vec[self.vocab[w]] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        self.vectors = np.array(vectors)

    def search(self, query: str, top_k: int = 2):
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        for w in query.lower().split():
            if w in self.vocab:
                q_vec[self.vocab[w]] += 1.0
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm
        scores = np.dot(self.vectors, q_vec)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [{"chunk": self.documents[i], "score": float(scores[i])} for i in top_idx]

vector_store = LocalVectorStore(chunks)

# 5. LOGIKA GROUNDED RAG ASSISTANT
def process_rag_query(question: str):
    top_k = 2
    results = vector_store.search(question, top_k=top_k)
    
    top_chunk = results[0]["chunk"]
    top_score = results[0]["score"]
    
    # Threshold grounding
    if top_score < 0.15:
        answer = "The answer cannot be determined from the available documents."
        source = "Tidak ada (Bukti tidak cukup)"
    else:
        answer = f"According to company policy: {top_chunk['content']}"
        source = top_chunk['source']
        
    return {
        "question": question,
        "answer": answer,
        "source": source,
        "top_score": top_score
    }

# 6. FASTAPI ENDPOINTS
@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Assistant Service is running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    return process_rag_query(request.question)

# 7. PENGUJIAN LOKAL (HANYA DIEKSEKUSI JIKA DI-RUN MANUAL `python app.py`)
if __name__ == "__main__":
    print("\n--- RUNNING LOCAL TESTS ---")
    test_questions = [
        "How many annual leave days do employees receive?",
        "How many leave days per year can employees get?",
        "What is the replacement cycle for standard company laptop?",
        "gym membership subsidy allowance"
    ]
    for q in test_questions:
        res = process_rag_query(q)
        print(f"\nQ: {res['question']}\nA: {res['answer']}\nSource: {res['source']}")