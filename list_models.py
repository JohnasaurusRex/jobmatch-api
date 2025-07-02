import google.generativeai as genai
import os

api_key = os.getenv('GENAI_API_KEY')
if not api_key:
    # Fallback to checking if the user provided it in previous context or just ask the user
    print("GENAI_API_KEY not found in environment.")
else:
    genai.configure(api_key=api_key)
    print("List of available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
