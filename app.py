"""
Flask Web Application & REST API for FAQ Chatbot
================================================
Endpoints:
- GET  /               : Interactive Web Chat UI
- POST /api/chat       : FAQ Query endpoint
- GET  /api/categories : List of FAQ categories with counts
- GET  /api/faqs       : Full knowledge base retrieval
- GET  /api/stats      : System and dataset statistics
"""

import os
from flask import Flask, jsonify, render_template, request
from faq_engine import FAQChatbotEngine

# Initialize Flask app
app = Flask(__name__)

# Load FAQ Chatbot Engine
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "faqs.json")
engine = FAQChatbotEngine(DATA_PATH)


@app.route("/")
def index():
    """Render the main chat interface."""
    categories = engine.get_all_categories()
    faq_count = len(engine.faqs)
    popular_questions = engine.get_popular_questions(count=4)
    return render_template(
        "index.html",
        categories=categories,
        faq_count=faq_count,
        popular_questions=popular_questions
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Process user query and return best-matching FAQ answer,
    confidence score, and related questions.
    """
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "status": "error",
            "message": "Empty query provided.",
            "response": {
                "type": "fallback",
                "answer": "Please ask a question so I can help you!",
                "suggestions": engine.get_popular_questions(4)
            }
        }), 400

    match_result = engine.find_best_match(user_message)

    return jsonify({
        "status": "success",
        "query": user_message,
        "response": match_result
    })


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """Return all categories with their question counts."""
    categories = engine.get_all_categories()
    details = []
    for cat in categories:
        items = engine.get_faqs_by_category(cat)
        details.append({
            "name": cat,
            "count": len(items),
            "sample_questions": [f["question"] for f in items[:2]]
        })
    return jsonify({
        "status": "success",
        "total_categories": len(categories),
        "categories": details
    })


@app.route("/api/faqs", methods=["GET"])
def get_faqs():
    """Return all FAQs or filter by category query param."""
    category = request.args.get("category")
    if category:
        faqs = engine.get_faqs_by_category(category)
    else:
        faqs = engine.faqs
    return jsonify({
        "status": "success",
        "total": len(faqs),
        "faqs": faqs
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return system and knowledge base metrics."""
    vocab_size = len(engine.vectorizer.vocabulary_) if engine.vectorizer else 0
    return jsonify({
        "status": "success",
        "total_faqs": len(engine.faqs),
        "total_indexed_documents": len(engine.indexed_docs),
        "vocabulary_size": vocab_size,
        "categories_count": len(engine.get_all_categories())
    })


if __name__ == "__main__":
    # Host on 127.0.0.1:5000
    print("Starting NovaTech FAQ Chatbot Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
