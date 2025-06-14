from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Use local MiniLM model for embeddings
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS index with local embeddings
vectorstore = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)

# Set up LLM (still using OpenAI for answering)
llm = ChatOpenAI(temperature=0)  # Make sure OPENAI_API_KEY is still set

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest):
    result = qa_chain.run(request.query)
    return {"response": result}
