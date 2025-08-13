import faiss
import pytest
from src.retrieval import AIDocumentStore
import numpy as np
from unittest.mock import patch
import pandas as pd
import os


@pytest.fixture
def mock_encoder(monkeypatch):
    def mock_encode(text, convert_to_numpy=True):
        return np.random.rand(384).astype("float32")
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_encode)


def test_build_index(mock_encoder, tmp_path):
    dataset_path = tmp_path / "test.csv"
    index_path = tmp_path / "test.index"
    
    # Create test dataset
    pd.DataFrame({
        'title': ['Test Paper'],
        'abstract': ['word ' * 1000],  # 1000 word abstract
        'url': ['http://test.com']
    }).to_csv(dataset_path, index=False)
    
    store = AIDocumentStore(dataset_path, index_path)
    index = store.build_index()
    assert os.path.exists(index_path)
    assert index.ntotal > 0

def test_initialize_with_existing_index(mock_encoder, tmp_path):
    dataset_path = tmp_path / "test.csv" 
    index_path = tmp_path / "test.index"
    
    # Create dummy index
    dummy_index = faiss.IndexFlatL2(384)
    faiss.write_index(dummy_index, index_path)
    
    store = AIDocumentStore(dataset_path, index_path)
    index = store.load_index()  # Should not rebuild
    assert index.ntotal == 0  # Our dummy index was empty

def test_embed_query_fallback(monkeypatch):
    from src.retrieval import embed_query
    
    def mock_encode(*args, **kwargs):
        raise Exception("Embedding failed")
    
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_encode)
    
    embedding = embed_query("test query")
    assert embedding.shape == (384,)
    assert np.all(embedding == 0) 

# Test that the constructor correctly sets initial attributes
def test_aidocumentstore_initialization(mock_encoder):
    store = AIDocumentStore(dataset_path="dummy.csv", index_path="dummy.index", chunk_size=100)
    assert store.dataset_path == "dummy.csv"
    assert store.index_path == "dummy.index"
    assert store.chunk_size == 100
    assert isinstance(store.documents, list)
    assert isinstance(store.document_metadata, list)



def test_embed_documents(mock_encoder):
    store = AIDocumentStore("dummy.csv", "dummy.index")
    store.documents = ["doc1", "doc2"]
    embeddings = store.embed_documents()
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)


def test_embed_query(mock_encoder):
    from src.retrieval import embed_query
    embedding = embed_query("test query")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)

    
# Test that load_and_split() raises an error if file doesn't exist
def test_load_and_split_file_not_found():
    with pytest.raises(FileNotFoundError) as excinfo:
        store = AIDocumentStore(dataset_path="nonexistent_file.csv", index_path="dummy.index")
        store.load_and_split()
    assert "Dataset not found" in str(excinfo.value)

# Test that chunk_text() properly splits a large string into expected chunks
def test_chunk_text_basic_split():
    text = "word " * 1050
    store = AIDocumentStore(dataset_path=None, index_path=None)
    chunks = store.chunk_text(text, size=500)
    assert len(chunks) == 3
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert sum(len(chunk.split()) for chunk in chunks) == 1050

# Test chunk_text() handles shorter input correctly
def test_chunk_text_small_text():
    text = "tiny text"
    store = AIDocumentStore(dataset_path=None, index_path=None)
    chunks = store.chunk_text(text, size=500)
    assert len(chunks) == 1
    assert chunks[0] == text

# Test chunk_text() returns an empty list on empty input
def test_chunk_text_empty_string():
    text = ""
    store = AIDocumentStore(dataset_path=None, index_path=None)
    chunks = store.chunk_text(text, size=500)
    assert chunks == []

# Test chunk_text() creates expected word groupings
def test_chunk_text_words_are_correct():
    text = "one two three four five six seven eight nine ten"
    store = AIDocumentStore(dataset_path=None, index_path=None)
    chunks = store.chunk_text(text, size=3)
    assert chunks == ["one two three", "four five six", "seven eight nine", "ten"]

# Test get_documents() returns a list of document chunks
def test_get_documents_returns_list():
    store = AIDocumentStore(dataset_path=None, index_path=None)
    store.documents = ["doc1", "doc2", "doc3"]
    docs = store.get_documents()
    assert isinstance(docs, list)
    assert docs == ["doc1", "doc2", "doc3"]

# Test get_metadata() returns a list of metadata entries
def test_get_metadata_returns_list():
    store = AIDocumentStore(dataset_path=None, index_path=None)
    store.document_metadata = [{"title": "Paper 1", "url": "http://example.com/1"}]
    metadata = store.get_metadata()
    assert isinstance(metadata, list)
    assert metadata[0]["title"] == "Paper 1"

# Test that load_index() raises an error if the FAISS index is missing
def test_load_index_raises_error_if_not_found():
    store = AIDocumentStore("dummy.csv", "nonexistent.index")
    with pytest.raises(FileNotFoundError):
        store.load_index()