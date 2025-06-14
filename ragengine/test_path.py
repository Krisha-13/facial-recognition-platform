# test_path.py
from pathlib import Path

faiss_path = Path("C:/Users/krish/facial-recognition-platform/ragengine/faiss_index/index.faiss")
print("Exists:", faiss_path.exists())
