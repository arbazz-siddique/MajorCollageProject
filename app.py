import os
import pandas as pd
import datetime
import time
import requests
import streamlit as st
import asyncio
import sys
import json
import base64
from io import BytesIO
import pyttsx3
from gtts import gTTS
from typing import Optional
from googletrans import Translator

translator = Translator()
# Fix for Windows event loop
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize TTS engine
_tts_active = False
_current_engine = None

# Initialize document store
def initialize_document_store():
    """Initialize and cache the document store"""
    store = AIDocumentStore("data/arxiv_dataset.csv", "data/faiss.index")
    
    if not os.path.exists("data/faiss.index"):
        with st.spinner("Building search index (this may take several minutes)..."):
            store.build_index()
    
    return store

def check_ollama_health():
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        return response.status_code == 200
    except:
        return False

if not check_ollama_health():
    st.error("Ollama server is not running. Please start it first.")
    st.stop()

# Import other modules after fixing the event loop
from src.retrieval import AIDocumentStore, embed_query
from src.generator import build_prompt, generate_answer
from src.memory import add_to_memory, format_memory_prompt
from src.upload_utils import extract_text_from_pdf, extract_text_from_txt, chunk_text, generate_chunk_title

# Initialize history file
if not os.path.exists("data/history.csv"):
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(columns=["timestamp", "question", "answer"]).to_csv("data/history.csv", index=False)

# Initialize session state
def initialize_session_state():
    st.session_state.setdefault("answer", "")
    st.session_state.setdefault("matched_docs", [])
    st.session_state.setdefault("tts_active", False)
    st.session_state.setdefault("qa_memory", [])
    st.session_state.setdefault("explanation", "")
    st.session_state.setdefault("prompt", "")
    st.session_state.setdefault("model_loaded", False)
    st.session_state.setdefault("tts_language", "en")
    st.session_state.setdefault("streaming", False)
    st.session_state.setdefault("stop_generation", False)

initialize_session_state()

# --- UI Components ---
st.title("AI Study Buddy")
st.markdown("Ask an AI/ML related question via text or upload. Get answers with memory, reasoning, and sources!")

# Sidebar settings
with st.sidebar:
    st.header("Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 1000, 300, 50)
    prompt_style = st.selectbox("Prompt Style", [
        "Default", "Concise", "Beginner-Friendly", "Explain Step-by-Step", "With Citations Only"
    ])
    cot_enabled = st.toggle("Chain-of-Thought", value=False,
                          help="Helps the model think through the problem before answering.")
    
    st.header("Local Model Status")
    if st.button("Check Ollama Connection"):
        try:
            response = requests.get("http://localhost:11434")
            if response.status_code == 200:
                st.success("✅ Ollama is running!")
                st.session_state.model_loaded = True
            else:
                st.error("Ollama is not responding")
        except Exception as e:
            st.error(f"Failed to connect to Ollama: {str(e)}")



# Load FAISS index
@st.cache_resource(show_spinner="Loading document store...")
def load_ai_knower():
    try:
        store = initialize_document_store()
        index = store.load_index()
        documents = store.get_documents()
        metadata = store.get_metadata()
        return store, index, documents, metadata
    except Exception as e:
        st.error(f"Failed to initialize document store: {e}")
        st.stop()

try:
    store, index, documents, metadata = load_ai_knower()
except Exception as e:
    st.error(f"Critical error: {e}")
    st.stop()

# --- File Upload Section ---
uploaded_file = st.file_uploader("Optional: Upload a PDF or TXT file", type=["pdf", "txt"])
uploaded_chunks = []
selected_chunk = ""

if uploaded_file:
    try:
        with st.spinner("Processing document..."):
            full_text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_txt(uploaded_file)
            uploaded_chunks = chunk_text(full_text, chunk_size=300, overlap=20)

            # Only generate titles for the first 10 chunks initially
            preview_chunks = min(10, len(uploaded_chunks))
            chunk_titles = [f"Section {i+1}" for i in range(len(uploaded_chunks))]
            
            generate_titles = st.checkbox("Generate descriptive section titles (may take longer)", value=False)
            
            if generate_titles:
                with st.spinner("Generating section titles..."):
                    progress_bar = st.progress(0)
                    for i, chunk in enumerate(uploaded_chunks[:preview_chunks]):
                        chunk_titles[i] = generate_chunk_title(chunk)
                        progress_bar.progress((i + 1) / preview_chunks)
            
            selected_idx = st.selectbox(
                "Choose a section to include with your question:",
                range(len(uploaded_chunks)),
                format_func=lambda i: chunk_titles[i]
            )
            selected_chunk = uploaded_chunks[selected_idx]
            st.success("Document loaded successfully.")
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")

# --- Question Input ---
with st.form("question_form"):
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What is the difference between supervised and unsupervised learning?",
        key="user_query"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        submit = st.form_submit_button("Submit")
    with col2:
        if st.session_state.get("streaming", False):
            stop_gen = st.form_submit_button("Stop Generation")

# --- TTS Functions ---
# Update the TTS functions at the top of your file
def text_to_speech(text: str, language: str = "en") -> Optional[bytes]:
    """Convert text to speech audio with proper language support"""
    try:
        # Always use gTTS for non-English languages
        if language != "en":
            tts = gTTS(text=text, lang=language, slow=False)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            return audio_bytes.getvalue()
        else:  # Use pyttsx3 only for English
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.save_to_file(text, 'temp_audio.mp3')
            engine.runAndWait()
            with open('temp_audio.mp3', 'rb') as f:
                audio_bytes = f.read()
            os.remove('temp_audio.mp3')
            return audio_bytes
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def translate_text(text, target_lang="hi"):
    """Translate text to target language"""
    try:
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Fallback to original text


# Update the toggle_speech function
def toggle_speech(text: str, language: str = "en"):
    """Translate text if needed, then speak"""
    global _tts_active
    
    if _tts_active:
        _tts_active = False
        return False
    
    try:
        # Translate if not English
        if language != "en":
            translated_text = translate_text(text, language)
            st.toast(f"Translated to {language.upper()}")
        else:
            translated_text = text
        
        # Generate speech
        audio_bytes = text_to_speech(translated_text, language)
        if audio_bytes:
            audio_str = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
                <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_str}" type="audio/mp3">
                </audio>
            """
            st.components.v1.html(audio_html, height=0)
            _tts_active = True
            return True
        return False
    except Exception as e:
        st.error(f"Speech error: {str(e)}")
        return False

# Update the language mapping in your sidebar
with st.sidebar:
    st.header("Speech Settings")
    st.session_state.tts_language = st.selectbox(
    "Speech Language",
    options=["en", "hi", "kn", "ta", "te", "ml", "fr", "es","ja"],  # Add more as needed
    format_func=lambda x: {
        "en": "English",
        "hi": "Hindi",
        "ja": "日本語 (Japanese)" ,
        "kn": "Kannada", 
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "fr": "French",
        "es": "Spanish"

    }[x]
)
# --- Answer Generation ---
def generate_answer_stream(prompt: str, temperature: float = 0.2, max_tokens: int = 300):
    """Generate answer with streaming support"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=30
        )
        response.raise_for_status()
        
        answer_container = st.empty()
        full_response = ""
        
        for line in response.iter_lines():
            if st.session_state.stop_generation:
                answer_container.markdown(full_response + " (stopped)")
                st.session_state.stop_generation = False
                return full_response
                
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.strip():
                    try:
                        response_json = json.loads(decoded_line)
                        token = response_json.get("response", "")
                        full_response += token
                        answer_container.markdown(full_response + "▌")
                    except json.JSONDecodeError:
                        continue
        
        answer_container.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"Generation Error: {str(e)}")
        return "⚠️ Error generating answer"

# --- Answer Pipeline ---
# --- Answer Pipeline ---
if submit and query:
    st.session_state.stop_generation = False
    
    if any(greet in query.lower() for greet in ["hello", "hi", "hey", "how are you"]):
        with st.spinner("Thinking..."):
            st.session_state.answer = "Hello! I'm your AI study buddy. How can I help you with your studies today?"
            st.session_state.matched_docs = []
            st.session_state.explanation = "This is a greeting response."
            st.session_state.prompt = "Greeting response"
            time.sleep(0.5)  # Small delay to show spinner briefly
    else:
        input_context = selected_chunk if selected_chunk else ""
        full_input = query if not input_context else f"{input_context}\n{query}"

        # Use a single spinner context for the entire operation
        with st.spinner("Thinking..."):
            try:
                memory_context = format_memory_prompt(st.session_state.qa_memory)
                q_emb = embed_query(full_input).reshape(1, -1)
                _, I = index.search(q_emb, k=3)
                matched_docs = [
                    (doc[:1000], meta)
                    for doc, meta in [(documents[i], metadata[i]) for i in I[0] if i < len(documents)]
                ] if len(documents) > 0 else []

                st.session_state.prompt = build_prompt(
                    question=full_input,
                    docs_metadata=matched_docs,
                    style=prompt_style,
                    memory_block=memory_context,
                    cot=cot_enabled
                )

                # Use streaming for better UX
                st.session_state.streaming = True
                answer = generate_answer_stream(
                    st.session_state.prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Generate explanation only if not stopped
                if not st.session_state.stop_generation:
                    explanation_prompt = (
                        "You are a helpful AI tutor. Briefly explain why the following answer is accurate, "
                        "based on the context it was built from.\n\n"
                        f"Answer:\n{answer}\n\n"
                        f"Context:\n{st.session_state.prompt}\n\n"
                        "Explain why this answer makes sense:"
                    )
                    explanation = generate_answer(explanation_prompt, temperature=0.3, max_tokens=200)
                else:
                    explanation = "Generation was stopped by user"

                st.session_state.answer = answer
                st.session_state.matched_docs = matched_docs
                st.session_state.explanation = explanation
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
            finally:
                # Ensure these are always set when done
                st.session_state.streaming = False
                st.session_state.stop_generation = False

    # Common processing
    st.session_state.qa_memory = add_to_memory(st.session_state.qa_memory, query, st.session_state.answer)
    st.session_state.tts_active = False

    # Save to history
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([{"timestamp": ts, "question": query, "answer": st.session_state.answer}])
    df.to_csv("data/history.csv", mode="a", header=False, index=False)

# --- Display Answer ---
if st.session_state.answer:
    st.subheader("📚 Answer")
    
    if st.session_state.streaming:
        st.markdown(st.session_state.answer + "▌")
    else:
        st.markdown(st.session_state.answer)
    
    # TTS controls
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔊 Read Aloud"):
            st.session_state.tts_active = toggle_speech(
                st.session_state.answer,
                language=st.session_state.tts_language
            )
    with col2:
        st.caption(f"Language: {st.session_state.tts_language.upper()}")

    # Sources
    with st.expander("🔗 Sources"):
        if st.session_state.matched_docs:
            for i, (chunk, meta) in enumerate(st.session_state.matched_docs, start=1):
                st.markdown(f"**{i}. [{meta['title']}]({meta['url']})**")
                st.caption(chunk[:200] + "...")
        else:
            st.info("No sources referenced for this answer")

    # Previous Q&A
    with st.expander("🧠 Previous Q&A Context"):
        if st.session_state.qa_memory:
            for i, (q, a) in enumerate(st.session_state.qa_memory):
                st.markdown(f"**Q{i+1}:** {q}  \n**A:** {a[:200]}...")
        else:
            st.info("No previous conversation history")

    # Explanation
    with st.expander("🤔 Explanation (Why This Answer?)"):
        st.write(st.session_state.explanation)

    # Full Prompt
    with st.expander("🧩 Reasoning Trace (Full Prompt)"):
        st.code(st.session_state.prompt)

    # History
    with st.expander("📜 History"):
        num_history_to_show = st.number_input("Show last N Q&As", min_value=1, max_value=20, value=5, step=1)
        if os.path.exists("data/history.csv"):
            hist = pd.read_csv("data/history.csv")
            hist_to_show = hist.tail(num_history_to_show).iloc[::-1]

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download as CSV",
                    data=hist_to_show.to_csv(index=False).encode("utf-8"),
                    file_name="ai_study_buddy_history.csv",
                    mime="text/csv"
                )
            with col2:
                txt_log = "\n\n".join(
                    f"[{row['timestamp']}]\nQ: {row['question']}\nA: {row['answer']}"
                    for _, row in hist_to_show.iterrows()
                )
                st.download_button(
                    label="Download as TXT",
                    data=txt_log,
                    file_name="ai_study_buddy_history.txt",
                    mime="text/plain"
                )

            for idx, row in hist_to_show.iterrows():
                st.markdown(f"**{row['timestamp']}**  \n**Q:** {row['question']}  \n**A:** {row['answer'][:200]}...")
                if st.button(f"🔊 Read Answer {idx+1}", key=f"tts_history_{idx}"):
                    toggle_speech(row['answer'], language=st.session_state.tts_language)

# Footer
st.markdown("---")
st.markdown("""
📧 Questions or feedback? Reach out at [)  
💼 Connect with me on [LinkedIn]()
""")