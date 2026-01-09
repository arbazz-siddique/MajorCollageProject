import google.generativeai as genai

def test_gemini_fixed():
    api_key = "AIzaSyCB_tVoY5Ipqktok2NdVH6LIjEM7vamhlg"
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        print("✅ API configured successfully")
        
        # Test with available models
        model_options = [
            "gemini-2.0-flash",       # Latest
            "gemini-1.5-flash",       # Extremely stable, high quota
            "gemini-1.5-flash-8b",    # Highest quota, faster for simple tasks
            "gemini-1.5-pro",         # Smarter, but lower quota
        ]
        
        for model_name in model_options:
            try:
                print(f"🧪 Testing model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("What is 2+2? Answer in one word.")
                print(f"✅ SUCCESS with {model_name}: {response.text}")
                return True
            except Exception as e:
                print(f"❌ Failed with {model_name}: {e}")
                continue
                
        print("❌ All models failed")
        return False
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    test_gemini_fixed()