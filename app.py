import os
import pandas as pd
import datetime
import time
import streamlit as st
import asyncio
import sys
import json
import base64
from io import BytesIO
# REMOVED pyttsx3
from gtts import gTTS
from typing import Optional
# REMOVED googletrans
import re
from deep_translator import GoogleTranslator



# Import all modules at the top
from src.gemini_utils import GeminiHandler, check_gemini_health
from src.retrieval import AIDocumentStore, embed_query
from src.generator import build_prompt, generate_answer, generate_answer_stream
from src.memory import add_to_memory, format_memory_prompt
from src.upload_utils import extract_text_from_pdf, extract_text_from_txt, chunk_text, generate_chunk_title

translator = Translator()

# Fix for Windows event loop
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Docker-specific configurations
if os.path.exists('/.dockerenv'):
    st.sidebar.info("🐳 Running in Docker container")

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
# In your sidebar section, add this:
with st.sidebar:
    st.header("Model Settings")
    
    # Gemini API Key input
    google_api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get from https://aistudio.google.com/",
        value=st.session_state.get("google_api_key", "")
    )
    st.session_state.google_api_key = google_api_key
    
    if st.button("Check Gemini Connection"):
        if google_api_key:
            if check_gemini_health(google_api_key):
                st.success("✅ Gemini is connected!")
            else:
                st.error("❌ Gemini connection failed")
        else:
            st.warning("Please enter Gemini API key")
    
    # Add debug info
    st.header("🔧 Debug Info")
    if st.session_state.get("google_api_key"):
        st.write(f"API Key Length: {len(st.session_state.google_api_key)}")
        st.write(f"Starts with: {st.session_state.google_api_key[:6]}...")
    
    # Test with simple prompt
    if st.button("Test Simple Prompt"):
        if st.session_state.get("google_api_key"):
            try:
                from src.gemini_utils import GeminiHandler
                gemini = GeminiHandler(st.session_state.google_api_key)
                test_response = gemini.generate_answer("What is 2+2? Answer in one word.")
                st.success(f"Test Response: {test_response}")
            except Exception as e:
                st.error(f"Test Failed: {e}")
        else:
            st.warning("Please enter API key first")
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 50)
    prompt_style = st.selectbox("Prompt Style", [
        "Default", "Concise", "Beginner-Friendly", "Explain Step-by-Step", "With Citations Only"
    ])
    cot_enabled = st.toggle("Chain-of-Thought", value=False)
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
def text_to_speech(text: str, language: str = "en") -> Optional[bytes]:
    """Convert text to speech audio using gTTS (Cloud Compatible)"""
    try:
        # We FORCE gTTS for all languages to prevent cloud crashes
        # pyttsx3 does not work reliably on Streamlit Cloud
        tts = gTTS(text=text, lang=language, slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        return audio_bytes.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def translate_text(text, target_lang="hi"):
    """Translate text using deep-translator (More stable)"""
    try:
        # deep-translator handles the client automatically
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Fallback to original

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
        options=["en", "hi", "kn", "ta", "te", "ml", "fr", "es","ja"],
        format_func=lambda x: {
            "en": "English",
            "hi": "Hindi",
            "ja": "日本語 (Japanese)",
            "kn": "Kannada", 
            "ta": "Tamil",
            "te": "Telugu",
            "ml": "Malayalam",
            "fr": "French",
            "es": "Spanish"
        }[x]
    )

# --- Answer Pipeline ---
if submit and query:
    st.session_state.stop_generation = False
    
    # --- 🔴 FIXED GREETING LOGIC START ---
    # We use Regex (\b) to ensure we match "hi" as a whole word, 
    # not inside "machine", "history", or "within".
    greeting_patterns = [
        r"\bhello\b", r"\bhi\b", r"\bhey\b", 
        r"\bhow are you\b", r"\bgood morning\b", 
        r"\bgood afternoon\b", r"\bgood evening\b"
    ]
    
    # Check if any pattern exists in the query
    is_greeting = any(re.search(p, query.lower()) for p in greeting_patterns)
    # --- 🔴 FIXED GREETING LOGIC END ---
    
    if is_greeting:
        with st.spinner("Thinking..."):
            st.session_state.answer = "Hello! I'm your AI study buddy. How can I help you with your studies today?"
            st.session_state.matched_docs = []
            st.session_state.explanation = "This is a greeting response."
            st.session_state.prompt = "Greeting response"
    else:
        # Process regular question
        input_context = selected_chunk if selected_chunk else ""
        full_input = query if not input_context else f"{input_context}\n{query}"

        with st.spinner("Searching documents and generating answer..."):
            try:
                # Get memory context
                memory_context = format_memory_prompt(st.session_state.qa_memory)
                
                # Search for relevant documents
                q_emb = embed_query(full_input).reshape(1, -1)
                _, I = index.search(q_emb, k=3)
                matched_docs = [
                    (documents[i][:1000], metadata[i])
                    for i in I[0] if i < len(documents)
                ] if len(documents) > 0 else []

                # Build prompt
                st.session_state.prompt = build_prompt(
                    question=full_input,
                    docs_metadata=matched_docs,
                    style=prompt_style,
                    memory_block=memory_context,
                    cot=cot_enabled
                )

                # DEBUG: Show prompt in sidebar
                with st.sidebar.expander("🔍 Debug: Generated Prompt"):
                    st.text_area("Prompt", st.session_state.prompt, height=200)

                # Generate answer using Gemini
                st.session_state.streaming = True
                answer_container = st.empty()
                full_response = ""

                # Use the streaming function from generator.py
                # st.info(f"🔍 Using model: {st.session_state.get('current_model', 'models/gemini-2.0-flash')}")
                
                for chunk in generate_answer_stream(
                    st.session_state.prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=st.session_state.get("google_api_key")
                ):
                    if st.session_state.stop_generation:
                        break
                    full_response += chunk
                    answer_container.markdown(full_response + "▌")
                    time.sleep(0.01)

                st.session_state.answer = full_response
                st.session_state.matched_docs = matched_docs

                # Generate explanation if we got a response
                if not st.session_state.stop_generation and full_response.strip():
                    explanation_prompt = (
                        "You are a helpful AI tutor. Briefly explain why the following answer is accurate, "
                        "based on the context it was built from.\n\n"
                        f"Answer:\n{st.session_state.answer}\n\n"
                        f"Context:\n{st.session_state.prompt}\n\n"
                        "Explain why this answer makes sense:"
                    )
                    st.session_state.explanation = generate_answer(
                        explanation_prompt, 
                        temperature=0.3, 
                        max_tokens=200,
                        api_key=st.session_state.get("google_api_key")
                    )
                else:
                    st.session_state.explanation = "Generation was stopped by user or no response generated"

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                st.session_state.answer = f"Error: {str(e)}"
                st.session_state.explanation = f"An error occurred: {str(e)}"
            finally:
                st.session_state.streaming = False
                st.session_state.stop_generation = False

    # Common processing for both greeting and regular questions
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
📧 Questions or feedback? Reach out at [your-email@example.com]  
💼 Connect with me on [LinkedIn](https://linkedin.com/in/your-profile)
""")