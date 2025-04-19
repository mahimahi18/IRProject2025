from temp.rag_langchain import RAGLangChain

def main():
    rag = RAGLangChain()

    while True:
        query = input("\nYour Question (or type 'exit'): ")
        if query.lower() == 'exit':
            break

        answer = rag.answer_question(query)
        print(f"\n📝 Answer:\n{answer}\n")

if __name__ == "__main__":
    main()
