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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Paths
INPUT_DIR = "pdfs/"
OUTPUT_DIR = "processed/"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize NLP tools
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

def extract_tables_from_pdf(pdf_path, output_csv_path):
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            table = page.extract_table()
            if table:
                with open(f"{output_csv_path}_table{i}.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(table)

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return preprocess_text(text)

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
        with open(os.path.join(OUTPUT_DIR, f"{file_name}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        extract_tables_from_pdf(file_path, os.path.join(OUTPUT_DIR, file_name))
    elif file_path.endswith((".jpg", ".jpeg", ".png")):
        text = extract_text_from_image(file_path)
        with open(os.path.join(OUTPUT_DIR, f"{file_name}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

# Process all files
for file in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR, file)
    process_file(file_path)

print("Processing complete. Extracted text and tables saved in 'processed/' directory.")
