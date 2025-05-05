import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import pandas as pd

# Load model on CPU
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

model = load_model()

# Extract text from PDF
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

# Preprocess text (basic cleaning)
def preprocess(text):
    return text.lower().strip()

# Streamlit UI
st.title("📄 AI-Powered Resume Screening System")

# Job Description Input
job_description = st.text_area("Paste the Job Description:")

# Resume Upload
uploaded_files = st.file_uploader("Upload Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

if st.button("Score & Rank Resumes") and job_description and uploaded_files:
    jd_embedding = model.encode(preprocess(job_description), convert_to_tensor=True)
    results = []

    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        resume_embedding = model.encode(preprocess(resume_text), convert_to_tensor=True)
        score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
        results.append((file.name, score))

    # Show ranked results
    ranked = sorted(results, key=lambda x: x[1], reverse=True)
    st.subheader("🏆 Ranked Resumes:")
    for i, (name, score) in enumerate(ranked, 1):
        st.write(f"{i}. **{name}** — Score: `{score:.2f}`")
