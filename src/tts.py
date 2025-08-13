import pyttsx3
import streamlit as st
import threading

# Global variables to track TTS state and thread
_active = False
_current_engine = None
_current_thread = None

def toggle_speech(text):
    global _active, _current_engine, _current_thread

    # If already speaking, stop the engine and clean up
    if _active:
        _active = False
        if _current_engine:
            try:
                _current_engine.stop()
            except Exception as e:
                st.error(f"Error stopping TTS engine: {e}")
        
        if _current_thread and _current_thread.is_alive():
            _current_thread.join(timeout=0.5)
        
        _current_engine = None
        return False

    # Otherwise, start speaking in a new thread
    _active = True

    def speak_text():
        global _active, _current_engine
        try:
            # Initialize new TTS engine
            _current_engine = pyttsx3.init()
            
            # Queue the text
            _current_engine.say(text)
            if _active:
                # Play the speech
                _current_engine.runAndWait()
        except Exception as e:
            st.error(f"TTS error: {e}")
        finally:
            # Reset state after speaking
            _active = False
            _current_engine = None

    _current_thread = threading.Thread(target=speak_text, daemon=True)
    _current_thread.start()

    return True