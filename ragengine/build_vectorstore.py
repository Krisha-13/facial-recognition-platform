import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Disable TensorFlow/Keras usage ===
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# === Paths ===
ENCODING_PATH = os.path.join(os.path.dirname(__file__), "../face-recognition/known_faces.npy")
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

# === Load local embedding model wrapper ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def fetch_face_logs():
    """Fetch name and timestamp from known_faces.npy and format as LangChain documents."""
    if not os.path.exists(ENCODING_PATH):
        print("[INFO] No face encodings found.")
        return []

    data = np.load(ENCODING_PATH, allow_pickle=True).tolist()
    if not isinstance(data, list):
        data = []

    docs = []
    for entry in data:
        name = entry.get("name")
        timestamp = entry.get("timestamp")
        if name and timestamp:
            content = f"{name} was registered on {timestamp}"
            docs.append(Document(page_content=content, metadata={"name": name, "timestamp": timestamp}))
    return docs

def build_and_save_vectorstore():
    docs = fetch_face_logs()
    if not docs:
        print("[INFO] No face data found in known_faces.npy.")
        return

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    # Build FAISS vector store
    vectorstore = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)

    # Save vector store to disk
    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"[INFO] Vector store saved to {VECTOR_DB_PATH}")

if __name__ == "__main__":
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    build_and_save_vectorstore()
