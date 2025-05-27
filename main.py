import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = "AIzaSyDKxR-3TMZNrL0Zjh50cufqHtGIJLIN6qA"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3002", "http://127.0.0.1:3002"])


# Gemini helper function
def generate_gemini_response(prompt, model="gemini-2.0-flash"):
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Financial Literacy API for Pakistan"})

@app.route("/generate-content", methods=["POST"])
def generate_content():
    data = request.get_json()
    topic = data.get("topic")
    language = data.get("language_preference", "mixed")

    if not topic:
        return jsonify({"error": "Missing topic"}), 400

    if language == "urdu":
        language_modifier = "Explain in Urdu using urdu/arabic script."
    elif language == "mixed":
        language_modifier = "Explain in simple Urdu-English mix."
    else:
        language_modifier = "Explain in simple English."

    prompt = f"""
    Act like a financial literacy coach for Pakistani adults.
    {language_modifier}
    Explain the basics of {topic} clearly for someone with limited financial knowledge.
    Give real-life examples from Pakistan's context.
    Format your response with appropriate headings and bullet points.
    Keep the tone friendly, encouraging, and clear.
    """

    content = generate_gemini_response(prompt)
    if content:
        return jsonify({"content": content, "topic": topic})
    else:
        return jsonify({"error": "Failed to generate content"}), 500

import re  # Add this at the top if not already imported

@app.route("/quiz", methods=["POST"])
def generate_quiz():
    data = request.get_json()
    topic = data.get("topic")
    num_questions = data.get("num_questions", 5)

    prompt = f"""
    Create a multiple-choice quiz on the topic of {topic} for Pakistani adults learning financial literacy.
    Generate exactly {num_questions} questions.
    Each question should have 4 options with exactly one correct answer.
    Make the questions straightforward and educational.
    Format your response as a JSON array with the following structure:
    [
      {{
        "question": "Question text here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": "Option X"
      }}
    ]
    Do not include any explanations, just the JSON array.
    """

    response = generate_gemini_response(prompt)

    # Remove triple backtick code block if present
    cleaned_response = re.sub(r"```(?:json)?\n", "", response).strip().rstrip("```").strip()

    try:
        quiz_data = json.loads(cleaned_response)
        return jsonify({"quiz": quiz_data, "topic": topic})
    except Exception as e:
        return jsonify({"error": "Failed to parse quiz data", "raw_response": response}), 500

@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    data = request.get_json()
    answers = data.get("answers", [])
    topic = data.get("topic")

    if not answers or not topic:
        return jsonify({"error": "Missing answers or topic"}), 400

    answers_text = "\n".join([
        f"Question: {a['question']}\nSelected answer: {a['selected_answer']}"
        for a in answers
    ])

    prompt = f"""
    Evaluate these answers for a financial literacy quiz on {topic}.

    {answers_text}

    For each question, check if the selected answer is correct.
    Then provide feedback on each answer, explaining why it's correct or incorrect.
    Also calculate an overall score (percentage correct).
    Format your response as a JSON object with this structure:
    {{
      "score": 80,
      "feedback": [
        {{
          "question": "Question text",
          "selected_answer": "User's answer",
          "is_correct": true/false,
          "explanation": "Explanation of why it's right/wrong"
        }}
      ],
      "overall_feedback": "Overall feedback on the quiz performance"
    }}
    """

    response = generate_gemini_response(prompt)
    try:
        evaluation_data = json.loads(response)
        return jsonify(evaluation_data)
    except:
        return jsonify({"error": "Failed to parse evaluation response", "raw_response": response}), 500

@app.route("/financial-plan", methods=["POST"])
def financial_plan():
    data = request.get_json()
    income = data.get("monthly_income")
    expenses = data.get("monthly_expenses", {})
    savings_goal = data.get("savings_goal")
    timeframe = data.get("timeframe_months", 12)

    if income is None or not isinstance(expenses, dict):
        return jsonify({"error": "Missing or invalid income/expenses"}), 400

    total_expenses = sum(expenses.values())
    surplus = income - total_expenses

    expense_text = "\n".join([f"- {k}: PKR {v}" for k, v in expenses.items()])
    goal_text = f"They want to save PKR {savings_goal} in {timeframe} months." if savings_goal else ""

    prompt = f"""
    Act as a financial advisor for a Pakistani individual with the following financial situation:

    Monthly income: PKR {income}
    Monthly expenses:
    {expense_text}

    Total expenses: PKR {total_expenses}
    Monthly surplus: PKR {surplus}

    {goal_text}

    Based on this information:
    1. Analyze their current spending patterns
    2. Suggest specific areas where they could reduce expenses
    3. Provide a realistic monthly budget plan with percentages for essential expenses, savings, and discretionary spending
    4. Give advice on how to achieve their financial goals within the Pakistani context
    5. Suggest local investment or savings options appropriate for their situation

    Format your advice in a clear, structured manner with specific financial recommendations.
    """

    advice = generate_gemini_response(prompt)
    if advice:
        return jsonify({
            "financial_advice": advice,
            "monthly_income": income,
            "total_expenses": total_expenses,
            "surplus": surplus
        })
    else:
        return jsonify({"error": "Failed to generate financial plan"}), 500

@app.route("/ask-content-question", methods=["POST"])
def ask_content_question():
    data = request.get_json()
    content = data.get("content")
    question = data.get("question")
    if not content or not question:
        return jsonify({"error": "Missing content or question"}), 400

    # 1. Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([content])

    # 2. Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()  # Uses your OPENAI_API_KEY from env
    db = FAISS.from_documents(docs, embeddings)

    # 3. Set up the retriever and QA chain
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),  # Uses your OPENAI_API_KEY
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    # 4. Run the QA chain
    result = qa({"query": question})
    answer = result["result"]

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
