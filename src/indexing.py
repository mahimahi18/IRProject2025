import os
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
import pickle
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import json

class IRSystem:
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = {}  # Store source info, titles, etc.
        
        # For vector search
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_dim = 384  # dimensionality of chosen model
        self.index = None
        self.doc_embeddings = None  # Store embeddings for potential reuse
        
        # For inverted index
        self.inverted_index = defaultdict(list)
        self.vectorizer = TfidfVectorizer(
            lowercase=True, 
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        
        self.load_documents()
    
    def _extract_title(self, content):
    # Example: Extract the first line as the title
        return content.split("\n")[0] if content else "Untitled"

    def load_documents(self):
        """Load all processed documents and their metadata"""
        for file_path in self.processed_dir.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc_id = file_path.stem
                
                # Store document and ID
                self.documents.append(content)
                self.doc_ids.append(doc_id)
                
                # Store metadata
                self.doc_metadata[doc_id] = {
                    'source': str(file_path),
                    'length': len(content.split()),
                    'title': self._extract_title(content)  # You can implement this based on your docs
                }
    
    def build_vector_db(self, index_type: str = 'flat'):
        """Build FAISS index with specified type"""
        # Compute embeddings
        print("Computing document embeddings...")
        self.doc_embeddings = self.encoder.encode(
            self.documents, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Choose index type
        if index_type == 'flat':
            self.index = faiss.IndexFlatL2(self.vector_dim)
        elif index_type == 'ivf':
            # IVF index for faster search with slight accuracy trade-off
            quantizer = faiss.IndexFlatL2(self.vector_dim)
            nlist = min(len(self.documents), 100)  # number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist)
            self.index.train(self.doc_embeddings)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add vectors to index
        self.index.add(np.array(self.doc_embeddings).astype('float32'))
        
        return self.index
    
    def build_inverted_index(self):
        """Build inverted index with TF-IDF scores"""
        print("Building inverted index...")
        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        terms = self.vectorizer.get_feature_names_out()
        
        # Build inverted index
        for idx, doc in enumerate(tfidf_matrix):
            doc_vector = doc.toarray().flatten()
            for term_idx, value in enumerate(doc_vector):
                if value > 0:
                    self.inverted_index[terms[term_idx]].append({
                        'doc_id': self.doc_ids[idx],
                        'tfidf_score': value
                    })
        
        return self.inverted_index
    
    def vector_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search using vector similarity"""
        if self.index is None:
            raise ValueError("Vector DB not built! Call build_vector_db() first.")
        
        # Encode query
        query_vector = self.encoder.encode([query])
        
        # Search
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), k
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.doc_ids):  # Safety check
                doc_id = self.doc_ids[idx]
                results.append({
                    'doc_id': doc_id,
                    'content': self.documents[idx],
                    'similarity_score': 1 / (1 + distance),
                    'metadata': self.doc_metadata[doc_id]
                })
        
        return results
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """Combine vector and boolean search results"""
        vector_results = self.vector_search(query, k=k)
        boolean_results = self.boolean_search(query)
        
        # Combine scores
        combined_scores = defaultdict(float)
        
        # Add vector scores
        for result in vector_results:
            combined_scores[result['doc_id']] += alpha * result['similarity_score']
            
        # Add boolean scores
        max_boolean_score = max(r['score'] for r in boolean_results) if boolean_results else 1
        for result in boolean_results:
            combined_scores[result['doc_id']] += (1 - alpha) * (result['score'] / max_boolean_score)
        
        # Sort and format results
        results = []
        for doc_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            results.append({
                'doc_id': doc_id,
                'combined_score': score,
                'metadata': self.doc_metadata[doc_id]
            })
        
        return results
    
    def boolean_search(self, query: str) -> List[Dict]:
        """Boolean search using inverted index"""
        query_vector = self.vectorizer.transform([query])
        query_terms = self.vectorizer.inverse_transform(query_vector)[0]
        
        matching_docs = defaultdict(float)
        for term in query_terms:
            if term in self.inverted_index:
                for doc_info in self.inverted_index[term]:
                    matching_docs[doc_info['doc_id']] += doc_info['tfidf_score']
        
        return [
            {'doc_id': doc_id, 'score': score}
            for doc_id, score in sorted(
                matching_docs.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
    
    def save_indexes(self, base_path: str):
        """Save all indexes and metadata, ensuring the directory exists"""
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{base_path}_faiss.index")
        
        # Save document embeddings
        np.save(f"{base_path}_embeddings.npy", self.doc_embeddings)
        
        # Save inverted index
        with open(f"{base_path}_inverted.pkl", 'wb') as f:
            pickle.dump(self.inverted_index, f)
        
        # Save metadata
        with open(f"{base_path}_metadata.json", 'w') as f:
            json.dump(self.doc_metadata, f)
    
    def load_indexes(self, base_path: str):
        """Load all indexes and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(f"{base_path}_faiss.index")
        
        # Load embeddings
        self.doc_embeddings = np.load(f"{base_path}_embeddings.npy")
        
        # Load inverted index
        with open(f"{base_path}_inverted.pkl", 'rb') as f:
            self.inverted_index = pickle.load(f)
        
        # Load metadata
        with open(f"{base_path}_metadata.json", 'r') as f:
            self.doc_metadata = json.load(f)

# Example usage
if __name__ == "__main__":
    # Initialize system
    ir_system = IRSystem("processed/")
    
    # Build indexes
    ir_system.build_vector_db(index_type='ivf')  # Use IVF index for better performance
    ir_system.build_inverted_index()
    
    # Save indexes
    ir_system.save_indexes("indexes/ir_system")
    
    # Example searches
    query = "How do I apply for a travel grant as a PhD student?"
    
    # Vector search
    vector_results = ir_system.vector_search(query, k=3)
    
    # Hybrid search (combines vector and boolean)
    hybrid_results = ir_system.hybrid_search(query, k=3, alpha=0.7)
    
    # Print results
    print("\nVector Search Results:")
    for result in vector_results:
        print(f"Doc: {result['doc_id']}")
        print(f"Score: {result['similarity_score']:.3f}")
        print(f"Source: {result['metadata']['source']}\n")
    
    print("\nHybrid Search Results:")
    for result in hybrid_results:
        print(f"Doc: {result['doc_id']}")
        print(f"Score: {result['combined_score']:.3f}")
        print(f"Source: {result['metadata']['source']}\n")