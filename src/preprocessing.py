import os
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import numpy as np
import cv2  # OpenCV
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import shutil
import pandas as pd  # Added for better table merging

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

INPUT_DIR = "pdfs/"
OUTPUT_DIR = "processed/"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

os.makedirs(OUTPUT_DIR, exist_ok=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return preprocess_text(text)

def extract_tables_from_pdf(pdf_path, output_csv_base_path):
    table_paths = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            table = page.extract_table()
            if table:
                # Check if table has more than just header, or non-empty cells
                non_empty_rows = [row for row in table if any(cell and cell.strip() for cell in row)]
                if len(non_empty_rows) > 1:  # More than just header
                    table_path = f"{output_csv_base_path}_table{i}.csv"
                    with open(table_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(non_empty_rows)
                    table_paths.append(table_path)
    return table_paths

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return preprocess_text(text)

def merge_text_and_tables(text_content, table_paths):
    merged_content = text_content + "\n\n"
    for table_path in table_paths:
        try:
            df = pd.read_csv(table_path)
            # Skip tables that are mostly empty
            if df.dropna(how='all').shape[0] > 1:  # At least 2 non-empty rows
                table_text = df.to_string(index=False)
                merged_content += "Extracted Table:\n" + table_text + "\n\n"
            else:
                print(f"Skipping empty or trivial table: {table_path}")
        except Exception as e:
            print(f"Warning: Could not process table {table_path}: {e}")
    return merged_content


def process_file(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            images = fitz.open(file_path)
            text = ""
            for i, img in enumerate(images):
                pix = img.get_pixmap()
                img_path = os.path.join(OUTPUT_DIR, f"{file_name}_page{i}.png")
                pix.save(img_path)
                text += extract_text_from_image(img_path) + "\n"
        # Extract tables and get their paths
        table_paths = extract_tables_from_pdf(file_path, os.path.join(OUTPUT_DIR, file_name))
        # Merge text and tables
        merged_content = merge_text_and_tables(text, table_paths)
        # Save merged content
        with open(os.path.join(OUTPUT_DIR, f"{file_name}.txt"), "w", encoding="utf-8") as f:
            f.write(merged_content)
    elif file_path.endswith((".jpg", ".jpeg", ".png")):
        text = extract_text_from_image(file_path)
        with open(os.path.join(OUTPUT_DIR, f"{file_name}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

for file in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR, file)
    process_file(file_path)

print("Processing complete.")
