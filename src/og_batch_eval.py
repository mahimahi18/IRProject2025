import os
import time
import pandas as pd
from indexing import IRSystem
from rag_mistral import MistralRAG
from tqdm import tqdm

# Paths
QUESTIONS_FILE = "temp/questions.csv"
OUTPUT_FILE = "og_questions_with_answers.csv"

# Load questions
questions_df = pd.read_csv(QUESTIONS_FILE)

# Load IR system and RAG
ir_system = IRSystem("processed/")
ir_system.load_indexes("indexes/ir_system")
rag = MistralRAG(ir_system)

# Generate answers
answers = []
print(f"Generating answers for {len(questions_df)} queries...")
for query in tqdm(questions_df['query']):
    try:
        answer = rag.answer_question(query)
    except Exception as e:
        print(f"First attempt failed for query '{query}': {e}")
        time.sleep(2)  # wait a bit and retry
        try:
            answer = rag.answer_question(query)
        except Exception as e2:
            print(f"Second attempt also failed for query '{query}': {e2}")
            answer = "[ERROR]"
    
    answers.append(answer)
    
    # Sleep after every request to avoid 60 RPM limit
    time.sleep(1.1)  # sleep for slightly more than 1 sec

# Add answers to dataframe
questions_df['answer'] = answers

# Save output
questions_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved completed file to {OUTPUT_FILE}")
