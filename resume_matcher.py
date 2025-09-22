import os
import numpy as np
import faiss
import streamlit as st
from PyPDF2 import PdfReader
import time
import random
from openai import OpenAI
from openai import RateLimitError

# ---- Initialize OpenAI client ----
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---- Helper functions ----
def load_resume_texts(folder="resumes"):
    resumes = {}
    for fname in os.listdir(folder):
        if fname.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, fname))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            resumes[fname] = text
    return resumes

def get_embedding(text, model="text-embedding-3-small", max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=text)
            return np.array(resp.data[0].embedding)
        except RateLimitError:
            sleep_time = (2 ** attempt) + random.random()
            st.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
    st.error("Failed after multiple retries due to rate limits.")
    return None

@st.cache_resource
def build_index():
    resumes = load_resume_texts("resumes")
    index = faiss.IndexFlatL2(1536)  # correct embedding size
    resume_embeddings = []
    resume_names = list(resumes.keys())

    for text in resumes.values():
        emb = get_embedding(text)
        resume_embeddings.append(emb)

    resume_embeddings = np.array(resume_embeddings).astype("float32")
    index.add(resume_embeddings)
    return index, resume_names, resume_embeddings, resumes

# ---- Streamlit App ----
st.title("Resume Recommender")
st.write("Paste a Job Description and get the best matching resumes.")

index, resume_names, resume_embeddings, resumes = build_index()

jd = st.text_area("Paste the Job Description here")

if st.button("Find Best Resume") and jd:
    jd_emb = get_embedding(jd).astype("float32")
    D, I = index.search(np.array([jd_emb]), k=3)

    st.subheader("Top Matching Resumes")
    for rank, idx in enumerate(I[0]):
        st.write(f"**Rank {rank+1}: {resume_names[idx]}** (score: {D[0][rank]:.2f})")
