import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import os

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

# Preprocessing (optional: you can add more)
def preprocess(text):
    return text.lower().strip()

# UI
st.title("AI-Powered Resume Screening System")

# Job Description Input
job_description = st.text_area("Paste the Job Description here")
uploaded_resumes = st.file_uploader("Upload Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

if st.button("Score & Rank Resumes") and job_description and uploaded_resumes:
    jd_embedding = model.encode(preprocess(job_description), convert_to_tensor=True)

    results = []
    for file in uploaded_resumes:
        resume_text = extract_text_from_pdf(file)
        resume_embedding = model.encode(preprocess(resume_text), convert_to_tensor=True)
        score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
        results.append((file.name, score))

    # Sort results
    ranked = sorted(results, key=lambda x: x[1], reverse=True)

    st.subheader("Ranked Resumes:")
    for i, (name, score) in enumerate(ranked, 1):
        st.write(f"{i}. **{name}** – Score: {score:.2f}")
