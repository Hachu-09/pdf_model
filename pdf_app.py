import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os
import re
import requests
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import difflib
import tempfile
from pdf2image import convert_from_path
import google.generativeai as genai
from io import BytesIO

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Set your API key for Gemini LLM
api_key = "AIzaSyBdfcvlqp0PD2HWq1IzwXSMBcKah84W1_Q"
genai.configure(api_key=api_key)

# Gemini model setup
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Sentence transformer model setup
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate response using Gemini
def generate_response_gemini(prompt, max_length=8192):
    response = model.generate_content(contents=[prompt])
    generated_text = response.text.strip()
    
    if len(generated_text) > max_length:
        generated_text = generated_text[:max_length]
    
    return generated_text

def extract_images_from_pdf(pdf_bytes):
    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes.getbuffer())  # Write BytesIO content to the temporary file
        temp_pdf.flush()  # Ensure all data is written to disk
        temp_pdf_path = temp_pdf.name  # Get the path of the temporary file

    # Now use the file path with convert_from_path
    images = convert_from_path(temp_pdf_path)
    
    return images
# Function to perform OCR on extracted images
def perform_ocr(images):
    ocr_text = ""
    for idx, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='eng')
        ocr_text += f"Text from Image {idx + 1}:\n{text}\n\n"
    return ocr_text

# Function to extract text using PyPDF2
def extract_text_pypdf2(file_path):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text using pdfplumber
def extract_text_pdfplumber(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Preprocessing text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(w.lower()) for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    return set(tokens)

# Extract numerical values from text
def extract_numerical_values(text):
    pattern = r"\b\d+(\.\d+)?\b"
    return re.findall(pattern, text)

# Split text into chunks
def split_text_into_chunks(text, max_chunk_size=1000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Processing PDFs
def process_pdf(pdf_path):
    text_pypdf2 = extract_text_pypdf2(pdf_path)
    text_pdfplumber = extract_text_pdfplumber(pdf_path)
    images = extract_images_from_pdf(pdf_path)
    ocr_text = perform_ocr(images)
    
    combined_text = text_pypdf2 + "\n" + text_pdfplumber + "\n" + ocr_text
    
    numerical_values = extract_numerical_values(combined_text)
    chunks = split_text_into_chunks(combined_text)
    document_embeddings = encoder.encode(chunks)
    
    return combined_text, numerical_values, chunks, document_embeddings

# Extract risk and problem factors
def extract_risk_problem_factors(text, numerical_values):
    prompt = (
        f"Based on the following medical report content and numerical values:\n\n"
        f"Medical Report: {text}\n\n"
        f"Numerical Values: {numerical_values}\n\n"
        f"Please extract the following information:\n\n"
        f"1. Risk Domain (such as cardiovascular, kidney, respiratory, etc.)\n"
        f"2. Specific Disease Problem\n"
        f"3. Stage of Risk (e.g., low, medium, high)\n\n"
        f"Provide the output in a structured format, preferably JSON."
    )
    
    # Generate response from the model
    response = generate_response_gemini(prompt)
    
    # Attempt to parse the response as JSON
    try:
        risk_problem_factors = json.loads(response)
    except json.JSONDecodeError:
        print("Error: The response is not in valid JSON format. Raw response returned.")
        print(response)
        # If the response is not JSON, handle it as plain text
        risk_problem_factors = {
            "Risk Domain": "Unknown",
            "Specific Disease Problem": "Unknown",
            "Stage of Risk": "Unknown"
        }
    
    return risk_problem_factors

# Generating a prevention report
def generate_prevention_report(risk, disease, age):
    if not risk and not disease:
        return "No significant risks or problems detected. You're safe and healthy. Keep up the good work!"
    
    prompt = f"""
    Provide a detailed wellness report with the following sections:

    1. *Introduction*
       - Purpose of the report
       - Context of general health and wellness

    2. *Risk Description*
       - General description of the identified risk
       - Common factors associated with the risk

    3. *Stage of Risk*
       - General information about the risk stage
       - Any typical considerations

    4. *Risk Assessment*
       - General overview of the risk's impact on health

    5. *Findings*
       - General wellness observations
       - Supporting information

    6. *Recommendations*
       - General wellness tips and lifestyle changes
       - Actions to promote well-being

    7. *Way Forward*
       - Suggested next steps for maintaining health
       - General follow-up actions

    8. *Conclusion*
       - Summary of overall wellness advice
       - General support resources

    Generate the report based on the following details:

    Risk Domain: {risk}
    Disease Problem: {disease}
    User Age: {age}
    """
    response = generate_response_gemini(prompt)
    return response

# Main Streamlit App
st.title("Medical Report Analysis and Prevention Report Generator")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Input Query and Age
query = st.text_input("Enter the query for analysis")
age = st.number_input("Enter your age", min_value=0, max_value=120)

# Process the PDF and Generate Report
if st.button("Generate Report"):
    if uploaded_file is not None:
        # Process PDF file
        with BytesIO(uploaded_file.read()) as pdf_data:
            combined_text, numerical_values, chunks, document_embeddings = process_pdf(pdf_data)
            
        # Extract risk and problem factors
        risk_problem_factors = extract_risk_problem_factors(combined_text, numerical_values)
        
        # Generate prevention report
        prevention_report = generate_prevention_report(
            risk_problem_factors['Risk Domain'],
            risk_problem_factors['Specific Disease Problem'],
            age
        )
        
        # Display the generated report
        st.subheader("Generated Prevention Report")
        st.text(prevention_report)
    else:
        st.error("Please upload a PDF file before generating the report.")
