import google.generativeai as genai
import os

# Configure the API key manually or from environment variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

print("Available Gemini Models (Legacy SDK):\n" + "="*50)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Name:        {model.name}")
        print(f"Description: {model.description}")
        print("-" * 50)