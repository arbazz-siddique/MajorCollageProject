class Config:
    GEMINI_MODEL = "models/gemini-2.0-flash"  # Use available model
    MAX_TOKENS = 1000
    TEMPERATURE = 0.2
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    FAISS_INDEX_DIM = 384
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"