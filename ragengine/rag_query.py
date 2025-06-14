import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# === Paths ===
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

# === Load embedding model ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load FAISS VectorStore ===
vectorstore = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# === Load Ollama LLM ===
llm = Ollama(model="llama3")

# === Setup Retriever + QA Chain ===
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

# === Ask in loop ===
print("Ask a question (or type 'exit'):\n")
while True:
    query = input(">>> ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.invoke(query)
    print("\n[ANSWER] " + response + "\n")
