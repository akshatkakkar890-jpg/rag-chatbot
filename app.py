import os
import glob
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import chromadb
from openai import OpenAI

# =========================
# CONFIG
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

DOC_FOLDER = "docs"
CHUNK_SIZE = 800
MAX_CONTEXT_DOCS = 3

# =========================
# LOAD PDFs
# =========================

def load_pdfs(folder):
    texts = []
    for file in glob.glob(f"{folder}/*.pdf"):
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return texts

documents = load_pdfs(DOC_FOLDER)

# =========================
# SIMPLE CHUNKING
# =========================

chunks = []
for doc in documents:
    for i in range(0, len(doc), CHUNK_SIZE):
        chunks.append(doc[i:i+CHUNK_SIZE])

# =========================
# VECTOR DB (OpenAI embeddings)
# =========================

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("rag")

if len(collection.get()["ids"]) == 0:
    for i, chunk in enumerate(chunks):
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        collection.add(
            ids=[str(i)],
            embeddings=[emb],
            documents=[chunk]
        )

# =========================
# BM25
# =========================

bm25 = BM25Okapi([c.split() for c in chunks])

# =========================
# FASTAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/chat")
def chat(q: str):

    # Vector search
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=q
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=MAX_CONTEXT_DOCS
    )

    vector_docs = results["documents"][0]

    # BM25 search
    bm25_results = bm25.get_top_n(q.split(), chunks, n=MAX_CONTEXT_DOCS)

    # Combine & deduplicate
    combined = list(set(vector_docs + bm25_results))
    context = "\n\n".join(combined[:MAX_CONTEXT_DOCS])

    prompt = f"""
Answer using ONLY the given context.
If not found, say "Not found".

Context:
{context}

Question:
{q}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content

    return {"answer": answer}


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Hybrid RAG Chatbot</title>
<style>
body { font-family: Arial; background:#111; color:white; }
#chat { height:70vh; overflow-y:auto; }
.msg { padding:8px; margin:5px; border-radius:6px; }
.user { background:#2563eb; }
.bot { background:#333; }
</style>
</head>
<body>
<h2>Hybrid RAG Chatbot</h2>
<div id="chat"></div>
<input id="q" placeholder="Ask something..." />
<button onclick="ask()">Send</button>
<script>
const chat=document.getElementById("chat");
function add(t,c){let d=document.createElement("div");
d.className="msg "+c; d.innerText=t; chat.appendChild(d);}
async function ask(){
let q=document.getElementById("q").value;
add(q,"user");
let res=await fetch("/chat?q="+encodeURIComponent(q));
let data=await res.json();
add(data.answer,"bot");
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
