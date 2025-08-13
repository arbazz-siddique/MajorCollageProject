import requests
import pymupdf
import streamlit as st
from sentence_transformers import SentenceTransformer
from config import Config

# Initialize embedding model for title generation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

def chunk_text(text, chunk_size=300, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def generate_chunk_title(text):
    try:
        if len(text.split()) < 5:  # Don't generate title for very short chunks
            return text[:80] + "..."
            
        # First try to extract a meaningful sentence
        sentences = [s.strip() for s in text.split('.') if len(s.split()) > 3]
        if sentences:
            return sentences[0][:80] + ("..." if len(sentences[0]) > 80 else "")
            
        # Only call LLM if necessary
        prompt = (
            "Summarize the following content into a short, descriptive title (5-10 words max).\n\n"
            f"Content:\n{text}\n\n"
            "Title:"
        )
        
        response = requests.post(
            Config.OLLAMA_URL + "/api/generate",
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": 0.3,
                "max_tokens": 20,
                "stream": False,
            },
            timeout=10,  # Reduced timeout
        )
        response.raise_for_status()
        return response.json()["response"].strip().replace('"', '')
    except Exception:
        # Fallback to first meaningful words
        return ' '.join(text.split()[:7]) + ("..." if len(text.split()) > 7 else "")