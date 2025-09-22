import os
import numpy as np
import json
from PyPDF2 import PdfReader
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_KEY")  # put your key here temporarily

def load_resume_texts(folder="resumes"):
    resumes = {}
    for fname in os.listdir(folder):
        if fname.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, folder, fname))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            resumes[fname] = text
    return resumes

def get_embedding(text, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding)

resumes = load_resume_texts("resumes")
resume_names = list(resumes.keys())
resume_embeddings = []

for name, text in resumes.items():
    emb = get_embedding(text)
    resume_embeddings.append(emb)

resume_embeddings = np.array(resume_embeddings).astype("float32")

# Save for later
np.save("resume_embeddings.npy", resume_embeddings)
with open("resume_names.json", "w") as f:
    json.dump(resume_names, f)

print("✅ Saved embeddings and names!")
