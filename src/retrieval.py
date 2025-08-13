import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import streamlit as st
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedding_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        st.success("Embedding model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        raise

class AIDocumentStore:
    def __init__(self, dataset_path, index_path, chunk_size=500):
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.documents = []
        self.document_metadata = []
        self.index_path = index_path
        self._embedding_model = None
        self._index = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def load_and_split(self):
        """Load dataset and split into chunks with progress bar"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
            
        df = pd.read_csv(self.dataset_path)
        with st.spinner("Processing documents..."):
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Splitting documents"):
                chunks = self.chunk_text(row['abstract'], self.chunk_size)
                for chunk in chunks:
                    self.documents.append(chunk)
                    self.document_metadata.append({
                        'title': row['title'],
                        'url': row['url']
                    })

    def chunk_text(self, text, size):
        """Split text into chunks with word boundaries"""
        words = text.split()
        return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

    def embed_documents(self):
        """Embed documents with batch processing"""
        embeddings = []
        batch_size = 32  # Process documents in batches
        
        with st.spinner("Creating embeddings..."):
            for i in tqdm(range(0, len(self.documents), batch_size), desc="Embedding"):
                batch = self.documents[i:i + batch_size]
                embeddings.extend(self.embedding_model.encode(
                    batch, 
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=batch_size
                ))
        
        return np.array(embeddings).astype("float32")

    def build_index(self):
        """Build and save FAISS index"""
        self.load_and_split()
        embeddings = self.embed_documents()
        
        # Create optimized FAISS index
        dim = embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(100, len(self.documents)//4))
        index.nprobe = 5 
        # Train and add embeddings
        index.train(embeddings)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)
        
        st.success(f"Index built with {len(self.documents)} documents")
        return index

    def load_index(self):
        """Load existing FAISS index with validation"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}\n"
                "Run build_index() first to generate it."
            )
            
        index = faiss.read_index(self.index_path)
        if index.ntotal == 0:
            raise ValueError("Loaded index is empty")
            
        return index

    def get_metadata(self):
        return self.document_metadata

    def get_documents(self):
        return self.documents

def embed_query(query):
    """Embed a query with error handling"""
    try:
        model = get_embedding_model()
        return model.encode(query, convert_to_numpy=True).astype("float32")
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return np.zeros(384)  # Return zero vector as fallback