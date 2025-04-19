import pandas as pd

# Paths
questions_with_refs = "src/questions.csv"  # your queries + references
new_answers = "src/batch_answers.csv"      # new answers
output_file = "src/logs/final_log_for_evaluation.csv"

# Load both files
ref_df = pd.read_csv(questions_with_refs)
ans_df = pd.read_csv(new_answers)

# Merge by 'query'
merged_df = pd.merge(ans_df, ref_df, on="query", how="left")

# Save the final merged CSV
merged_df.to_csv(output_file, index=False)
print("✅ Final merged file ready:", output_file)
