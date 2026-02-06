# ==========================================================
# 🚀 HYBRID RAG CHATBOT (HF API DEPLOYABLE VERSION)
# ==========================================================

import os
import glob
import requests

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==========================================================
# SETTINGS
# ==========================================================

DOC_FOLDER = "docs"
DB_DIR = "chroma_db"

TOP_K_VECTOR = 6
TOP_K_BM25 = 6
MAX_CONTEXT_DOCS = 3

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# ==========================================================
# LOAD PDFs
# ==========================================================

def load_all_pdfs(folder):
    docs = []
    for file in glob.glob(f"{folder}/*.pdf"):
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file, "page": i}
                    )
                )
    return docs


print("📄 Loading PDFs...")
documents = load_all_pdfs(DOC_FOLDER)
print("✅ Pages loaded:", len(documents))


# ==========================================================
# CHUNKING
# ==========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)


# ==========================================================
# EMBEDDINGS + VECTOR DB
# ==========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

if os.path.exists(DB_DIR):
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
else:
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )
    vectordb.persist()

vector_retriever = vectordb.as_retriever(
    search_kwargs={"k": TOP_K_VECTOR}
)


# ==========================================================
# BM25 SEARCH
# ==========================================================

bm25_corpus = [c.page_content.split() for c in chunks]
bm25 = BM25Okapi(bm25_corpus)

def bm25_search(query):
    scores = bm25.get_scores(query.split())
    top_idx = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K_BM25]
    return [chunks[i] for i in top_idx]


# ==========================================================
# HYBRID RETRIEVAL
# ==========================================================

def hybrid_retrieve(query):
    v_docs = vector_retriever.invoke(query)
    b_docs = bm25_search(query)
    unique = {d.page_content: d for d in v_docs + b_docs}
    return list(unique.values())[:MAX_CONTEXT_DOCS]


def format_context(docs):
    return "\n\n".join(
        f"[Page {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )


# ==========================================================
# FASTAPI APP
# ==========================================================

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
    docs = hybrid_retrieve(q)
    context = format_context(docs)

    prompt = f"""
Answer using ONLY the given context.
If not found, say "Not found".

Context:
{context}

Question:
{q}

Answer:
"""

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0
            }
        }
    )

    result = response.json()

    if isinstance(result, list):
        answer = result[0]["generated_text"]
    else:
        answer = str(result)

    return {"answer": answer}


# ==========================================================
# WEB UI
# ==========================================================

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>🚀 Hybrid RAG Chatbot</title>
<style>
body { font-family: Arial; background:#0f0f0f; color:white; }
#chat { height:80vh; overflow-y:auto; margin-bottom:10px; }
.msg { padding:10px; margin:8px; border-radius:8px; max-width:80%; }
.user { background:#2563eb; margin-left:auto; }
.bot { background:#333; }
input { width:80%; padding:10px; }
button { padding:10px; }
</style>
</head>

<body>
<h2>🚀 Hybrid RAG Chatbot</h2>

<div id="chat"></div>

<input id="q" placeholder="Ask something..." />
<button onclick="ask()">Send</button>

<script>
const chat = document.getElementById("chat");

function add(text, cls) {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.innerText = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

async function ask() {
  const qInput = document.getElementById("q");
  const q = qInput.value.trim();
  if (!q) return;

  add(q, "user");
  qInput.value = "";

  const res = await fetch("/chat?q=" + encodeURIComponent(q));
  const data = await res.json();

  add(data.answer, "bot");
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
