# ==========================================
# AI Career Architect
# RAG Career Recommendation System
# ==========================================

# First-time installation:
# pip install -r requirements.txt

# ==========================================
# IMPORTS
# ==========================================

import os
import re
import requests
import numpy as np
import faiss
import gradio as gr

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ==========================================
# LOAD ENV VARIABLES
# ==========================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "❌ OPENROUTER_API_KEY not found.\n"
        "Create a .env file and add:\n"
        "OPENROUTER_API_KEY=your_api_key"
    )

# ==========================================
# CONFIG
# ==========================================

PDF_PATH = "data/E-youth Courses.pdf"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OPENROUTER_MODEL = "deepseek/deepseek-chat"

TOP_K_RESULTS = 3

# ==========================================
# LOAD PDF
# ==========================================

def load_pdf(path):
    """
    Extract text from PDF.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ PDF not found: {path}")

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    if not text.strip():
        raise ValueError("❌ PDF extraction failed")

    return text.strip()


# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):
    """
    Clean extracted PDF text.
    """

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"[^\w\u0600-\u06FF0-9.,،:()\-\s]",
        "",
        text
    )

    return text.strip()


# ==========================================
# CHUNKING
# ==========================================

def chunk_text(text, size=450, overlap=100):
    """
    Split text into overlapping chunks.
    """

    words = text.split()

    chunks = []

    i = 0

    while i < len(words):

        chunk = " ".join(words[i:i + size])

        if len(chunk) > 100:
            chunks.append(chunk)

        i += size - overlap

    return chunks


# ==========================================
# LOAD & PREPARE DATA
# ==========================================

print("📄 Loading PDF...")

raw_text = load_pdf(PDF_PATH)

cleaned_text = clean_text(raw_text)

chunks = chunk_text(cleaned_text)

if len(chunks) == 0:
    raise ValueError("❌ No valid chunks created")

print(f"✅ Chunks created: {len(chunks)}")


# ==========================================
# EMBEDDING MODEL
# ==========================================

print("🧠 Loading embedding model...")

embedding_model = SentenceTransformer(EMBED_MODEL)

embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

# Safety reshape
if len(embeddings.shape) == 1:
    embeddings = embeddings.reshape(1, -1)

print("✅ Embeddings generated")


# ==========================================
# FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("✅ FAISS index ready")


# ==========================================
# RETRIEVAL
# ==========================================

def retrieve(query, k=TOP_K_RESULTS):
    """
    Retrieve most relevant chunks.
    """

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:

        if 0 <= idx < len(chunks):

            chunk = chunks[idx].strip()

            if len(chunk) > 50:
                results.append(chunk)

    if not results:
        return "No relevant courses found in the dataset."

    return "\n\n".join(results)


# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are the eYouth Career Path Architect.

STRICT RULES:
- Use ONLY the provided context
- Answer ONLY in Arabic
- Recommend ONLY ONE best course
- If multiple courses exist:
    - Select the BEST one
    - Explain WHY it is the best
    - Briefly explain why others are weaker
- Be structured and concise

You MUST also generate a CAREER PATH.

CAREER PATH FORMAT:
- Beginner Level Course
- Intermediate Level Course
- Advanced Level Course

RESPONSE FORMAT:

🎯 Recommended Course (BEST ONLY)
📚 Why This Is The Best Choice
⚖️ Why Other Courses Are Less Suitable
🧩 Skills You Will Learn
🚀 Career Path (Beginner → Intermediate → Advanced)
"""


# ==========================================
# OPENROUTER CHAT
# ==========================================

def generate_response(question):
    """
    Generate AI response using retrieved context.
    """

    context = retrieve(question)

    prompt = f"""
{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly in Arabic.
"""

    try:

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 400
            },
            timeout=60
        )

        data = response.json()

        # API error handling
        if "error" in data:

            return (
                f"❌ API ERROR:\n"
                f"{data['error'].get('message', data['error'])}"
            )

        # Invalid response handling
        if "choices" not in data:

            return f"❌ INVALID RESPONSE:\n{data}"

        return data["choices"][0]["message"]["content"].strip()

    except Exception as error:

        return f"❌ REQUEST FAILED:\n{str(error)}"


# ==========================================
# GRADIO CHAT FUNCTION
# ==========================================

def respond(message, history):

    if not message.strip():
        return "اكتب سؤال من فضلك"

    return generate_response(message)


# ==========================================
# GRADIO UI
# ==========================================

ui = gr.ChatInterface(
    fn=respond,
    title="🎓 AI Career Advisor",
    description="RAG chatbot using OpenRouter + FAISS + PDF",
    theme="soft"
)

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    ui.launch()
