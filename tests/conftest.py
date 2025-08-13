# tests/conftest.py
import pytest
import numpy as np
from unittest.mock import patch

@pytest.fixture
def mock_encoder():
    with patch('sentence_transformers.SentenceTransformer.encode') as mock:
        mock.return_value = np.random.rand(384).astype("float32")
        yield mock

@pytest.fixture
def mock_ollama():
    with patch('requests.post') as mock:
        mock.return_value.json.return_value = {"response": "Mock response"}
        mock.return_value.raise_for_status.return_value = None
        yield mock 