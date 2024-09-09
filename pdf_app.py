import streamlit as st
import time
from io import BytesIO
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
import PyPDF2
import pdfplumber
import pytesseract
import tempfile
import json
import google.generativeai as genai
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk

# Download necessary NLTK data
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

# Streamlit page configuration
st.set_page_config(page_title="HL-PS Medical Report", layout="centered")

# Add title and subtitle
st.markdown("<h1 style='text-align: center;'>HL-PS Medical Report Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Get a comprehensive wellness report</h2>", unsafe_allow_html=True)

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# User input for query and age
query = st.text_input("Enter the query for analysis")
age = st.number_input("Enter your age", min_value=0, max_value=120)

# Extract images from PDF
def extract_images_from_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes.getbuffer())
        temp_pdf.flush()
        temp_pdf_path = temp_pdf.name
    images = convert_from_path(temp_pdf_path)
    return images

# Perform OCR on images
def perform_ocr(images):
    ocr_text = ""
    for idx, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='eng')
        ocr_text += f"Text from Image {idx + 1}:\n{text}\n\n"
    return ocr_text

# Extract text using PyPDF2 and pdfplumber
def extract_text_pypdf2(file_path):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_pdfplumber(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(w.lower()) for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    return set(tokens)

# Extract numerical values
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

# Process PDF
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
    response = model.generate_content(contents=[prompt])
    try:
        risk_problem_factors = json.loads(response.text.strip())
    except json.JSONDecodeError:
        risk_problem_factors = {
            "Risk Domain": "Unknown",
            "Specific Disease Problem": "Unknown",
            "Stage of Risk": "Unknown"
        }
    return risk_problem_factors

# Generate prevention report
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a detailed wellness report with the following sections:

    1. Introduction
       - Purpose of the report
       - Context of general health and wellness

    2. Risk Description
       - General description of the identified risk
       - Common factors associated with the risk

    3. Stage of Risk
       - General information about the risk stage
       - Any typical considerations

    4. Risk Assessment
       - General overview of the risk's impact on health

    5. Findings
       - General wellness observations
       - Supporting information

    6. Recommendations
       - General wellness tips and lifestyle changes
       - Actions to promote well-being

    7. Way Forward
       - Suggested next steps for maintaining health
       - General follow-up actions

    8. Conclusion
       - Summary of overall wellness advice
       - General support resources

    Generate the report based on the following details:

    Risk Domain: {risk}
    Disease Problem: {disease}
    User Age: {age}
    """
    response = model.generate_content(contents=[prompt])
    return response.text.strip()

# Generate report button
if st.button("Generate Report"):
    if uploaded_file is not None:
        with BytesIO(uploaded_file.read()) as pdf_data:
            combined_text, numerical_values, chunks, document_embeddings = process_pdf(pdf_data)
        risk_problem_factors = extract_risk_problem_factors(combined_text, numerical_values)
        prevention_report = generate_prevention_report(
            risk_problem_factors['Risk Domain'],
            risk_problem_factors['Specific Disease Problem'],
            age
        )
        st.markdown("<h3>Generated Prevention Report</h3>", unsafe_allow_html=True)
        st.text(prevention_report)
    else:
        st.error("Please upload a PDF file.")
