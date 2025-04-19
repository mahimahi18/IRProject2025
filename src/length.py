import pdfplumber
import os

pdf_dir = "pdfs/"
doc_lengths = []

for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        
        word_count = len(text.split())  # Count words
        doc_lengths.append(word_count)

average_length = sum(doc_lengths) / len(doc_lengths)
print(f"Average document length (in words): {average_length}")
