import requests
import streamlit as st
import time 

def build_prompt(question, docs_metadata, style="Default", memory_block="", cot=False):
    # Compile the document chunks into a referenceable context block
    context = ""
    for i, (chunk, meta) in enumerate(docs_metadata, start=1):
        context += f"\n[Source {i}] Title: {meta['title']}\n{chunk}\nURL: {meta['url']}\n"

    # Define available prompt styles
    instructions = {
        "Default": "Answer the question using the context provided.",
        "Concise": "Answer briefly and directly using only relevant context.",
        "Beginner-Friendly": "Explain clearly as if to someone new to AI. Use simple language.",
        "Explain Step-by-Step": "Answer the question by explaining your reasoning step by step.",
        "With Citations Only": "Answer using only the information provided in the sources below. Cite them clearly throughout your response."
    }

    style_instruction = instructions.get(style, instructions["Default"])
    cot_instruction = "\nRespond by thinking through the answer step by step." if cot else ""

    prompt = (
        "You are an expert AI assistant helping users understand concepts in artificial intelligence.\n\n"
        f"{style_instruction}{cot_instruction}\n"
        f"{memory_block}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    return prompt

def generate_answer(prompt, temperature=0.2, max_tokens=300, retries=3):
    for attempt in range(retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                timeout=30  # Increased timeout
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.Timeout:
            if attempt == retries - 1:
                st.error("LLM server timed out after multiple retries. Please check if Ollama is running.")
                return "⚠️ Error: LLM server unavailable"
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            st.error(f"LLM Error: {str(e)}")
            return "⚠️ Error generating answer"