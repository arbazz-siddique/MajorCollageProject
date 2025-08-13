from src.upload_utils import chunk_text, extract_text_from_txt, extract_text_from_pdf
import io
import pymupdf
from unittest.mock import patch


def test_generate_chunk_title(monkeypatch):
    from src.upload_utils import generate_chunk_title
    
    # Mock the Ollama API response
    def mock_post(*args, **kwargs):
        class MockResponse:
            def json(self):
                return {"response": "Mock Title"}
            def raise_for_status(self):
                pass
        return MockResponse()
    
    monkeypatch.setattr("requests.post", mock_post)
    
    title = generate_chunk_title("Some sample text content")
    assert title == "Mock Title"

def test_generate_chunk_title_fallback(monkeypatch):
    from src.upload_utils import generate_chunk_title
    
    # Mock a failed API call
    def mock_post(*args, **kwargs):
        raise Exception("API error")
    
    monkeypatch.setattr("requests.post", mock_post)
    
    long_text = " ".join(["word"] * 100)
    title = generate_chunk_title(long_text)
    assert "word" in title or "..." in title

    
# Test that long text gets chunked correctly with overlap
def test_chunk_text_length():
    txt = "word " * 1000
    chunks = chunk_text(txt, chunk_size=300, overlap=50)
    
    assert len(chunks) >= 3
    assert all(len(chunk.strip()) > 0 for chunk in chunks)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert any("word" in chunk for chunk in chunks)

# Test extracting plain text from a .txt file
def test_extract_text_from_txt():
    fake_txt = io.BytesIO(b"This is a simple test text file.")
    extracted_text = extract_text_from_txt(fake_txt)
    assert extracted_text == "This is a simple test text file."

# Test extracting text from a PDF using mocked PyMuPDF behavior
def test_extract_text_from_pdf(monkeypatch):
    class DummyPage:
        def get_text(self):
            return "Dummy page text."

    class DummyDocument:
        def __init__(self, stream, filetype):
            self.pages = [DummyPage(), DummyPage()]
        
        def __iter__(self):
            return iter(self.pages)
    
    def dummy_open(stream, filetype):
        return DummyDocument(stream, filetype)

    # Replace actual open() with dummy
    monkeypatch.setattr(pymupdf, "open", dummy_open)

    fake_pdf = io.BytesIO(b"%PDF-1.4 fake pdf content here")
    text = extract_text_from_pdf(fake_pdf)

    assert "Dummy page text." in text
    assert text.count("Dummy page text.") == 2