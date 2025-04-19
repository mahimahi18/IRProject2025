import os
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
import pickle
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import json

class IRSystem:
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = {}

        # 🔥 UPGRADE: Better encoder (optional, or keep your MiniLM)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_dim = 384

        self.index = None
        self.doc_embeddings = None

        self.inverted_index = defaultdict(list)

        # 🔥 UPGRADE: 1-gram to 3-gram for better matching
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            max_features=10000  # Optional: limit if needed
        )

        self.load_documents()

    def _extract_title(self, content):
        return content.split("\n")[0] if content else "Untitled"

    def load_documents(self):
        for file_path in self.processed_dir.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Skip empty
                    doc_id = file_path.stem

                    # ⬇ NEW: Prepend filename to the content
                    content = doc_id.replace("_", " ").replace("-", " ") + "\n\n" + content

                    self.documents.append(content)
                    self.doc_ids.append(doc_id)
                    self.doc_metadata[doc_id] = {
                        'source': str(file_path),
                        'length': len(content.split()),
                        'title': self._extract_title(content)
                    }

        # Load table CSVs too (merged already during preprocessing)
        for file_path in self.processed_dir.glob('*_table*.csv'):
            try:
                df = pd.read_csv(file_path)
                if df.dropna(how='all').shape[0] > 1:
                    content = df.to_string(index=False)
                    doc_id = file_path.stem

                    # ⬇ NEW: Prepend filename to the content for tables
                    content = doc_id.replace("_", " ").replace("-", " ") + "\n\n" + content

                    self.documents.append(content)
                    self.doc_ids.append(doc_id)
                    self.doc_metadata[doc_id] = {
                        'source': str(file_path),
                        'length': len(content.split()),
                        'title': self._extract_title(content)
                    }
            except Exception as e:
                print(f"Warning: skipping broken table {file_path}: {e}")


    def build_vector_db(self, index_type: str = 'ivf'):
        print("Computing dense embeddings...")
        self.doc_embeddings = self.encoder.encode(
            self.documents,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # 🔥 UPGRADE: Normalize for better FAISS behavior
        )

        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.vector_dim)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(self.vector_dim)
            nlist = min(len(self.documents) // 10, 100)
            self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist)
            self.index.train(np.array(self.doc_embeddings, dtype='float32'))
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        self.index.add(np.array(self.doc_embeddings, dtype='float32'))

    def build_inverted_index(self):
        print("Building sparse TF-IDF index...")
        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        terms = self.vectorizer.get_feature_names_out()

        for idx, row in enumerate(tfidf_matrix):
            for term_idx in row.nonzero()[1]:
                term = terms[term_idx]
                score = row[0, term_idx]
                self.inverted_index[term].append({
                    'doc_id': self.doc_ids[idx],
                    'tfidf_score': score
                })

    def vector_search(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None:
            raise ValueError("Vector DB not built")

        query_vec = self.encoder.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(np.array(query_vec, dtype='float32'), k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.doc_ids):
                results.append({
                    'doc_id': self.doc_ids[idx],
                    'similarity_score': float(distance),
                    'metadata': self.doc_metadata[self.doc_ids[idx]]
                })
        return results

    def boolean_search(self, query: str) -> List[Dict]:
        query_vector = self.vectorizer.transform([query])
        query_terms = self.vectorizer.inverse_transform(query_vector)[0]

        matching_docs = defaultdict(float)
        for term in query_terms:
            for doc_info in self.inverted_index.get(term, []):
                matching_docs[doc_info['doc_id']] += doc_info['tfidf_score']

        return [
            {'doc_id': doc_id, 'score': score}
            for doc_id, score in sorted(matching_docs.items(), key=lambda x: x[1], reverse=True)
        ]

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict]:
        dense_results = self.vector_search(query, k=k)
        sparse_results = self.boolean_search(query)

        combined_scores = defaultdict(float)

        for r in dense_results:
            combined_scores[r['doc_id']] += alpha * r['similarity_score']

        max_sparse = max((r['score'] for r in sparse_results), default=1)
        for r in sparse_results:
            combined_scores[r['doc_id']] += (1 - alpha) * (r['score'] / max_sparse)

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                'doc_id': doc_id,
                'combined_score': score,
                'metadata': self.doc_metadata[doc_id]
            }
            for doc_id, score in sorted_results[:k]
        ]

    def save_indexes(self, base_path: str):
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        faiss.write_index(self.index, f"{base_path}_faiss.index")
        np.save(f"{base_path}_embeddings.npy", self.doc_embeddings)

        with open(f"{base_path}_inverted.pkl", 'wb') as f:
            pickle.dump(self.inverted_index, f)

        with open(f"{base_path}_metadata.json", 'w') as f:
            json.dump(self.doc_metadata, f)

        with open(f"{base_path}_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_indexes(self, base_path: str):
        self.index = faiss.read_index(f"{base_path}_faiss.index")
        self.doc_embeddings = np.load(f"{base_path}_embeddings.npy")

        with open(f"{base_path}_inverted.pkl", 'rb') as f:
            self.inverted_index = pickle.load(f)

        with open(f"{base_path}_metadata.json", 'r') as f:
            self.doc_metadata = json.load(f)

        with open(f"{base_path}_vectorizer.pkl", 'rb') as f:
            self.vectorizer = pickle.load(f)

if __name__ == "__main__":
    ir_system = IRSystem("processed/")
    ir_system.build_vector_db(index_type='ivf')
    ir_system.build_inverted_index()
    ir_system.save_indexes("indexes/ir_system")

    query = "What are the proposed sub-areas for PhD Qualifying examination for the CS&IS department?"
    vector_results = ir_system.vector_search(query)
    hybrid_results = ir_system.hybrid_search(query)

    print("\nVector Search Results:")
    for res in vector_results:
        print(f"{res['doc_id']} -> {res['metadata']['source']}")

    print("\nHybrid Search Results:")
    for res in hybrid_results:
        print(f"{res['doc_id']} -> {res['metadata']['source']}")
