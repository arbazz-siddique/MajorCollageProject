# AI Study Buddy 🤖📚

Your intelligent research assistant that understands PDFs! Upload academic papers, select sections, and get AI-powered explanations with sources.

## 🚀 Features

- **Document Processing**: Upload and process PDF/TXT documents
- **Section-based Q&A**: Ask questions about specific document sections
- **Multi-language Support**: Text-to-speech (English, Hindi, Japanese, etc.)
- **Conversation Memory**: Remembers previous questions and answers
- **Cited Answers**: Provides sources for generated answers
- **Customizable AI**: Adjust temperature, max tokens, and prompt styles

## 🛠️ Tech Stack

**Core Technologies**:

- Python 3.11+
- Streamlit (Web Interface)
- Ollama (LLM Backend)
- FAISS (Vector Search)

**Key Libraries**:

- PyMuPDF (PDF processing)
- Sentence Transformers (Embeddings)
- gTTS/pyttsx3 (Text-to-speech)
- Pandas (Data handling)

## 🏁 Getting Started

### Prerequisites

- Python 3.11+
- Ollama installed and running
- Git (for cloning)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/arbazz-siddique/MajorCollageProject.git
cd MajorCollageProject
```

### Create and activate virtual environment:

```
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Install dependencies:

pip install -r requirements.txt

### Set up Ollama (in separate terminal):

```
ollama pull llama3
ollama serve
```

## 🖥️ Running the Application:

```
streamlit run app.py
```

## 📂 Project Structure:

MajorCollageProject/
├── app.py                # Main application
├── config.py             # Configuration
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── data/                 # Data storage
│   ├── arxiv_dataset.csv # Sample dataset
│   ├── faiss.index       # Vector index
│   └── history.csv       # Conversation history
├── src/                  # Core modules
│   ├── generator.py      # Answer generation
│   ├── memory.py         # Conversation memory
│   ├── retrieval.py      # Document search
│   ├── tts.py            # Text-to-speech
│   └── upload_utils.py   # File processing
└── tests/                # Test cases
    ├── test_chunking.py
    ├── test_memory.py
    └── test_retrieval.py


## 📜 License

MIT License - See [LICENSE](https://license/) file for details.

## 👨‍💻 About the Creator

**Mohammad Arbazz Siddique**
📧 Email: [arbazzsiddique104@gmail.com](https://mailto:arbazzsiddique104@gmail.com/)
🔗 GitHub: [arbazz-siddique](https://github.com/arbazz-siddique)

Feel free to reach out for:

* Feature requests
* Bug reports
* Collaboration opportunities
* General questions about the project

  Made with ❤️ using Python and Streamlit 

  ```
  Key improvements:
  1. Added emoji headers for better visual scanning
  2. Organized into clear, logical sections
  3. Included detailed tech stack information
  4. Added prerequisites section
  5. Improved project structure visualization
  6. Added specific contact information
  7. Made installation steps more detailed
  8. Included both production and development instructions

  Would you like me to:
  1. Add a troubleshooting section?
  2. Include specific configuration examples?
  3. Add screenshots of the interface?
  4. Include demo gifs/videos?
  ```
