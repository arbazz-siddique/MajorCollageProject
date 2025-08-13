@echo off
REM Start Ollama in background
start "Ollama Server" /MIN ollama serve

REM Wait 5 seconds to ensure Ollama is ready
timeout /t 5 > nul

REM Start Streamlit
streamlit run app.py --server.fileWatcherType none
