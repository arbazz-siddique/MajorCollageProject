# test_ollama_integration.py
import pytest
from unittest.mock import patch
import requests
from src.generator import generate_answer

def test_ollama_connection(monkeypatch):
    def mock_post(*args, **kwargs):
        class MockResponse:
            status_code = 200
            def json(self):
                return {"response": "Test response"}
            def raise_for_status(self):
                pass
        return MockResponse()
    
    monkeypatch.setattr("requests.post", mock_post)
    
    response = generate_answer("Test prompt")
    assert response == "Test response"

def test_ollama_connection_failure(monkeypatch):
    def mock_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Connection failed")
    
    monkeypatch.setattr("requests.post", mock_post)
    
    with pytest.raises(Exception):
        generate_answer("Test prompt")