class Config:
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3"  # or "mistral", "gemma"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_TOKENS = 500
    TEMPERATURE = 0.3
    FAISS_INDEX_DIM = 384
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50