import pandas as pd
import time
from temp.rag_langchain import RAGLangChain

# Load RAG system
rag = RAGLangChain()

# Load questions
questions_file = "src/questions.csv"  # where your query + reference are
questions_df = pd.read_csv(questions_file)

# Create a new DataFrame to store generated answers
generated_answers = []

for idx, row in questions_df.iterrows():
    query = row['query']
    print(f"Processing query {idx+1}/{len(questions_df)}: {query}")
    
    try:
        answer = rag.answer_question(query)
    except Exception as e:
        print(f"⚠️ Failed to generate answer for: {query} -- Error: {e}")
        answer = ""
    
    generated_answers.append({
        "query": query,
        "answer": answer
    })

    time.sleep(1.1)  # 🚨 Add delay between queries to avoid rate limit

# Save generated answers
output_file = "src/batch_answers.csv"
pd.DataFrame(generated_answers).to_csv(output_file, index=False)
print(f"\n✅ All answers generated and saved to {output_file}")
