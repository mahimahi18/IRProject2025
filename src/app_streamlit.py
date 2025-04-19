import os
import streamlit as st
import pandas as pd
from indexing import IRSystem
from rag_mistral import MistralRAG

# Load everything
ir_system = IRSystem("processed/")
ir_system.load_indexes("indexes/ir_system")
rag = MistralRAG(ir_system)

# Log file
LOG_FILE = "src/logs/query_answer_log.csv"

# Streamlit app
st.set_page_config(page_title="BITS QA System", page_icon="🧠", layout="wide")
st.title("🧠 BITS Research Regulations QA System")

query = st.text_input("Enter your question:", "")

if query:
    with st.spinner("Retrieving and generating answer..."):
        answer = rag.answer_question(query)
    
    st.subheader("📝 Answer:")
    st.write(answer)

    # Save to logs
    log_entry = pd.DataFrame([{
        "query": query,
        "answer": answer
    }])

    if not os.path.exists(LOG_FILE):
        log_entry.to_csv(LOG_FILE, index=False)
    else:
        log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)

    st.success("Answer generated and logged!")

    with st.expander("📜 View retrieved context and documents?"):
        retrieved_docs = ir_system.hybrid_search(query, k=3)
        for idx, doc in enumerate(retrieved_docs):
            st.markdown(f"**Document {idx+1}: {doc['metadata']['title']}**")
            with open(doc['metadata']['source'], 'r', encoding='utf-8') as f:
                st.text(f.read())

# Option to download logs
st.sidebar.header("Download Logs")
if st.sidebar.button("Download Query Log"):
    st.sidebar.download_button(
        label="Download log as CSV",
        data=open(LOG_FILE, 'rb'),
        file_name='query_answer_log.csv',
        mime='text/csv'
    )