from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
import os
import pickle

class LangChainIRSystem:
    def __init__(self, processed_dir="processed/"):
        self.processed_dir = Path(processed_dir)
        self.documents = []
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.faiss_index = None
        self.bm25_retriever = None

    def load_documents(self):
        print("Loading documents...")
        for file_path in self.processed_dir.glob("*.txt"):
            loader = TextLoader(str(file_path), encoding="utf-8")
            loaded_docs = loader.load()
            self.documents.extend(loaded_docs)

    def build_vector_db(self, save_path="indexes/faiss_index"):
        print("Building FAISS vector store...")
        self.faiss_index = FAISS.from_documents(self.documents, self.embedding_model)
        self.faiss_index.save_local(save_path)

    def build_sparse_index(self, save_path="indexes/bm25.pkl"):
        print("Building BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.bm25_retriever, f)

    def save_all(self):
        self.build_vector_db()
        self.build_sparse_index()

    def load_all(self):
        print("Loading indexes...")
        self.faiss_index = FAISS.load_local(
            "indexes/faiss_index",
            self.embedding_model,
            allow_dangerous_deserialization=True  # ✅ Add this line
        )
        with open("indexes/bm25.pkl", 'rb') as f:
            self.bm25_retriever = pickle.load(f)

if __name__ == "__main__":
    ir = LangChainIRSystem()
    ir.load_documents()
    ir.save_all()
