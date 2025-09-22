import os
import time
import json
import random
import numpy as np
from PyPDF2 import PdfReader
from openai import OpenAI, RateLimitError

# ✅ Load API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def get_embedding_with_retry(text, model="text-embedding-3-small", max_retries=5):
    """Get embeddings with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=text)
            return np.array(resp.data[0].embedding)
        except RateLimitError:
            wait_time = (2 ** attempt) + random.random()  # exponential backoff
            print(f"⚠️ Rate limit hit, retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    raise Exception("❌ Failed after multiple retries due to rate limits.")

# ---- Run ----
resumes = load_resume_texts("resumes")
resume_names = list(resumes.keys())
resume_embeddings = []

for name, text in resumes.items():
    print(f"📄 Processing {name}...")
    emb = get_embedding_with_retry(text)
    resume_embeddings.append(emb)
    time.sleep(1)  # small pause between calls to avoid rate-limit

resume_embeddings = np.array(resume_embeddings).astype("float32")

# Save results
np.save("resume_embeddings.npy", resume_embeddings)
with open("resume_names.json", "w") as f:
    json.dump(resume_names, f)

print("✅ All embeddings computed and saved!")
