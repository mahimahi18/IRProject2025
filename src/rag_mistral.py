import os
from mistral_api import query_mistral
from indexing import IRSystem

class MistralRAG:
    def __init__(self, ir_system: IRSystem):
        self.ir_system = ir_system

    def answer_question(self, question: str, top_k: int = 3, alpha: float = 0.5) -> str:
        # Retrieve top-k relevant documents using hybrid search
        retrieved_docs = self.ir_system.hybrid_search(question, k=top_k, alpha=alpha)

        # Build context from the top documents
        context = ""
        for doc in retrieved_docs:
            try:
                with open(doc['metadata']['source'], 'r', encoding='utf-8') as f:
                    context += f"\n\n{doc['metadata']['title']}:\n{f.read()}"
            except FileNotFoundError:
                context += f"\n\n{doc['metadata']['title']}:\n[Could not load document]"

        # Format prompt for Mistral
        prompt = f"""Use the following context to answer the question concisely and clearly.

Context:
{context}

Question:
{question}
"""

        return query_mistral(prompt)
