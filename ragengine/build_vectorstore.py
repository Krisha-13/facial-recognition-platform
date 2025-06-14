import os
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Disable TensorFlow/Keras usage ===
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# === Paths ===
DB_PATH = os.path.join(os.path.dirname(__file__), "../database/faces.db")
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

# === Load local embedding model wrapper ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def fetch_face_logs():
    """Fetch name and timestamp entries from SQLite DB and format as LangChain documents."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, timestamp FROM faces")
    rows = cursor.fetchall()
    conn.close()

    docs = []
    for name, timestamp in rows:
        content = f"{name} was registered on {timestamp}"
        docs.append(Document(page_content=content, metadata={"name": name, "timestamp": timestamp}))
    return docs

def build_and_save_vectorstore():
    docs = fetch_face_logs()
    if not docs:
        print("[INFO] No data found in DB.")
        return

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    # Build FAISS vector store using wrapped embedding model
    vectorstore = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)

    # Save vector index
    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"[INFO] Vector store saved to {VECTOR_DB_PATH}")

if __name__ == "__main__":
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    build_and_save_vectorstore()
