from fastapi import APIRouter
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
import os

router = APIRouter()

# === Request schema ===
class ChatRequest(BaseModel):
    message: str

# === Chat endpoint ===
@router.post("/chat")
def chat(request: ChatRequest):
    try:
        # Define vectorstore path
        vector_path = os.path.join("ragengine", "faiss_index")

        # Debug: confirm file structure
        print("📂 Current working directory:", os.getcwd())
        print("🔍 Checking vectorstore at:", vector_path)
        
        if not os.path.exists(vector_path):
            return {"error": f"❌ Vectorstore folder not found at '{vector_path}'"}
        
        files = os.listdir(vector_path)
        print("📄 Files in vectorstore folder:", files)

        if "index.faiss" not in files:
            return {"error": "❌ 'index.faiss' not found in vectorstore folder"}

        # Load vectorstore with embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(vector_path, embedding_model, allow_dangerous_deserialization=True)

        # Load local model (Ollama must be running)
        llm = ChatOllama(model="llama3", base_url="http://localhost:11434")

        # RAG pipeline
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
        response = qa.run(request.message)

        return {"answer": response}

    except Exception as e:
        print("⚠️ Exception during chat:", str(e))
        return {"error": str(e)}
