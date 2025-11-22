import pymupdf
import streamlit as st
from sentence_transformers import SentenceTransformer

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
            
        # Fallback to first meaningful words
        return ' '.join(text.split()[:7]) + ("..." if len(text.split()) > 7 else "")
    except Exception as e:
        st.error(f"Title generation error: {e}")
        return ' '.join(text.split()[:5]) + "..."