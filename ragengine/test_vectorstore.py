from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set the path to your FAISS index
VECTOR_DB_PATH = "./faiss_index"

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS vectorstore
vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Ask a test query
query = "When did Krisha register?"
results = vectorstore.similarity_search(query, k=3)

print(f"🔍 Top results for: {query}")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
