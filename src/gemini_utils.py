import google.generativeai as genai
import streamlit as st
import os
from typing import Optional, Generator
import time

class GeminiHandler:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.available_models = [
            "gemini-2.0-flash",       # Latest
            "gemini-1.5-flash",       # Extremely stable, high quota
            "gemini-1.5-flash-8b",    # Highest quota, faster for simple tasks
            "gemini-1.5-pro",         # Smarter, but lower quota
        ]
        self.current_model_index = 0
        self.model_name = self.available_models[self.current_model_index]
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
            except Exception as e:
                st.error(f"❌ Gemini configuration failed: {e}")
    
    def _detect_available_model(self):
        """Detect which model is actually available"""
        try:
            # Try the default model first
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content("Test", request_options={"timeout": 5})
            return True
        except:
            # Try other models
            for model_name in self.available_models[1:]:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content("Test", request_options={"timeout": 5})
                    self.model_name = model_name
                    st.info(f"🔁 Switched to model: {model_name}")
                    return True
                except:
                    continue
            # If all fail, use the latest flash as fallback
            self.model_name = "models/gemini-2.0-flash"
            return False
    
    def generate_answer(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> str:
        """Generate answer using Google Gemini"""
        if not self.api_key:
            return "⚠️ Error: Google API key not configured. Please add GOOGLE_API_KEY to your environment variables."
        
        try:
            model = genai.GenerativeModel(self.model_name)
            
            # Configure generation parameters
            generation_config = {
                "temperature": temperature,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=False
            )
            
            return response.text
            
        except Exception as e:
            error_msg = f"Gemini API Error: {str(e)}"
            st.error(error_msg)
            return f"⚠️ {error_msg}"
    
    def generate_answer_stream(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> Generator[str, None, None]:
        """Stream response from Gemini"""
        if not self.api_key:
            yield "⚠️ Error: Google API key not configured."
            return
        
        try:
            model = genai.GenerativeModel(self.model_name)
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            yield f"⚠️ Error: {str(e)}"

def check_gemini_health(api_key: str) -> bool:
    """Check if Gemini API is accessible with detailed error reporting"""
    if not api_key:
        st.error("❌ No API key provided")
        return False
        
    if not (api_key.startswith('AI') or api_key.startswith('AIza')):
        st.error("❌ Invalid API key format.")
        return False
        
    try:
        genai.configure(api_key=api_key)
        
        # Try multiple model options
        model_options = [
            "gemini-2.0-flash",       # Latest
            "gemini-1.5-flash",       # Extremely stable, high quota
            "gemini-1.5-flash-8b",    # Highest quota, faster for simple tasks
            "gemini-1.5-pro",         # Smarter, but lower quota
        ]
        
        working_model = None
        for model_name in model_options:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    "Say 'Hello' in one word.",
                    request_options={"timeout": 10}
                )
                if response.text:
                    working_model = model_name
                    break
            except:
                continue
        
        if working_model:
            st.success(f"✅ Gemini connection successful! Model: {working_model}")
            return True
        else:
            st.error("❌ No compatible model found")
            return False
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Gemini connection failed: {error_msg}")
        
        # Provide specific guidance based on error type
        if "API_KEY_INVALID" in error_msg:
            st.error("🔑 Your API key is invalid. Please check and regenerate it.")
        elif "quota" in error_msg.lower():
            st.error("📊 API quota exceeded. Check your Google AI Studio usage.")
        elif "location" in error_msg.lower() or "region" in error_msg.lower():
            st.error("🌍 API not available in your region. Try with VPN.")
        elif "permission" in error_msg.lower():
            st.error("🔒 API permission denied. Ensure Gemini API is enabled.")
            
        return False