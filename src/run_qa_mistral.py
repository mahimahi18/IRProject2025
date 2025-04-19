from indexing import IRSystem
from rag_mistral import MistralRAG

def main():
    # Load retrieval system and indexes
    ir_system = IRSystem("processed/")
    ir_system.load_indexes("indexes/ir_system")

    # Initialize RAG pipeline with Mistral API
    rag = MistralRAG(ir_system)

    print("💬 Ask a question about BITS research regulations!")
    while True:
        query = input("\nYour Question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break

        answer = rag.answer_question(query)
        print("\n🧠 Answer:\n", answer)

if __name__ == "__main__":
    main()
