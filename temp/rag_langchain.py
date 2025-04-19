from langchain.schema import Document
from temp.ir_system_langchain import LangChainIRSystem
from mistral_api import query_mistral

class RAGLangChain:
    def __init__(self):
        self.ir_system = LangChainIRSystem()
        self.ir_system.load_all()

    def answer_question(self, query, top_k=5, alpha=0.5, context_char_limit=3000):
        dense_results = self.ir_system.faiss_index.similarity_search_with_score(query, k=top_k)
        sparse_results = self.ir_system.bm25_retriever.get_relevant_documents(query)

        scores = {}
        for doc, score in dense_results:
            scores[doc.page_content] = alpha * (1 / (1 + score))
        for doc in sparse_results:
            scores[doc.page_content] = scores.get(doc.page_content, 0) + (1 - alpha)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=False)

        context = ""
        for doc_content, _ in sorted_docs:
            if len(context) + len(doc_content) < context_char_limit:
                context += doc_content + "\n\n"
            else:
                break

        prompt = f"""Use the following context to answer the question as concisely and clearly as possible.

Context:
{context}

Question:
{query}
"""
        return query_mistral(prompt)
