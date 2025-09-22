# 📌 Resume Matcher App

This is a Streamlit app that recommends the best matching resume(s) for a given job description using OpenAI embeddings and FAISS similarity search.

## 🚀 Features
- Upload resumes (PDF format) into a folder
- Paste a job description
- Get top 3 best-matching resumes with similarity scores

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/)
- [OpenAI API](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyPDF2](https://pypi.org/project/pypdf2/)
- Numpy

## ▶️ Run locally
```bash
pip install -r requirements.txt
streamlit run resume_matcher.py
