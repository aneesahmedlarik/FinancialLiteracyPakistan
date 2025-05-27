import google.generativeai as genai
import os

# Option 1: Set API key directly (for quick testing only)
genai.configure(api_key="AIzaSyDG6JXCsAKNZ-vANcCBGb0NmdTm76Ehv6Y")

# Option 2: (Better) Set API key as environment variable first
# os.environ["GOOGLE_API_KEY"] = "your_key_here"
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Create model
model = genai.GenerativeModel("gemini-2.0-flash")

# Try a basic prompt
try:
    response = model.generate_content("Hello Gemini! Can you explain what budgeting is in simple terms?")
    print("✅ API is working!")
    print("Gemini Response:\n", response.text)
except Exception as e:
    print("❌ Error calling Gemini API:")
    print(e)
