# Financial Literacy App Backend

This is the backend API for the Financial Literacy application targeted at Pakistani users. It uses FastAPI and the Google Gemini API to generate educational content, quizzes, and personalized financial advice.

## Setup Instructions

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Set up environment variables**

Create a `.env` file in the backend directory with the following content:

```
GOOGLE_API_KEY="your-api-key-here"
```

Replace `your-api-key-here` with your actual Google Gemini API key. You can get one from the [Google AI Studio](https://ai.google.dev/).

3. **Run the server**

```bash
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000

## API Endpoints

- `GET /`: Welcome message
- `POST /generate-content`: Generate educational content on a financial topic
- `POST /quiz`: Generate a quiz on a financial topic
- `POST /evaluate-answer`: Evaluate quiz answers and provide feedback
- `POST /financial-plan`: Generate personalized financial planning advice

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc 