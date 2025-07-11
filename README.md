# AI-Powered Resume Screening System 🧠📄  

## 🔍 Project Overview

The **AI-Powered Resume Screening System** is a machine learning-based tool that automates the initial screening of job applicants by analyzing and ranking resumes based on job descriptions. It reduces manual effort and helps recruiters quickly identify the most relevant candidates.

---

## 🚀 Features

- 📄 Extracts text from PDF resumes  
- 🧠 Uses NLP to match resumes with job descriptions  
- 📊 Assigns relevance scores to each candidate  
- ✅ Filters top candidates automatically  
- 📁 Supports multiple job roles and industries  

---

## 🧰 Tech Stack

- **Python**
- **scikit-learn** (for ML modeling)
- **NLTK / spaCy** (for text processing)
- **Tkinter / Streamlit** (optional: for GUI)
- **PyMuPDF** or **pdfminer** (for PDF text extraction)
- **Pandas & NumPy** (for data handling)

---

## 🏗️ Project Structure


resumes/ # Folder with sample resumes (PDF)
├── job_description.txt # Job description file
├── model/ # Trained model files
├── main.py # Main script to run screening
├── utils.py # Helper functions
├── requirements.txt # Python dependencies


---

## 📈 How It Works

1. Upload resumes (in PDF format).
2. Provide a job description.
3. The system preprocesses both and converts them into numerical vectors using TF-IDF.
4. Cosine similarity is calculated between the job description and each resume.
5. Resumes are ranked based on similarity scores.

---

## 🧪 Example Use Case

You’re a recruiter looking to hire a Data Analyst. You upload 50 resumes and your job description. The system will return the top 10 most relevant candidates in a few seconds — saving hours of manual work.

---

## 🙌 Acknowledgements

• Inspired by real HR challenges
• Built by Abhay Saini as part of a practical AI project

