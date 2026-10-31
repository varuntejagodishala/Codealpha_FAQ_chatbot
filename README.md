# NovaTech FAQ AI Chatbot 🤖

An intelligent, NLP-driven Question Answering chatbot built with **Python**, **NLTK**, **Scikit-Learn (TF-IDF & Cosine Similarity)**, and a modern, responsive **Flask Web UI** and **CLI**.

---

## 🚀 Features

- **Rich FAQ Knowledge Base**: Pre-loaded with 24+ structured questions, answers, tags, and query paraphrases across 7 real-world customer support categories (Orders, Shipping, Returns, Payments, Warranty, Account, Discounts).
- **NLP Preprocessing Pipeline (`NLTK`)**:
  - Contraction expansion (e.g., *can't* $\rightarrow$ *cannot*, *what's* $\rightarrow$ *what is*).
  - Text normalization, punctuation removal, and lowercasing.
  - Word tokenization (`nltk.word_tokenize` with robust fallback).
  - English stopword removal (preserving interrogative pronouns like *how*, *what*, *where*).
  - Morphological lemmatization (`nltk.WordNetLemmatizer`).
- **Semantic Matching with TF-IDF & Cosine Similarity**:
  - Vector space model representation with unigram + bigram n-grams.
  - Sublinear TF scaling to handle repetitive query terms.
  - Alternate question indexing for high-recall question retrieval.
  - Cosine similarity matching between the vectorized user query and the FAQ knowledge base.
- **Rule-based Conversational Intent Recognition**:
  - Greetings (*hi*, *hello*, *good morning*).
  - Gratitude (*thank you*, *thanks*).
  - Farewells (*bye*, *see you*).
  - Identity & Help inquiries (*who are you*, *what can you do*).
- **Confidence Scoring & Fallback System**:
  - **High Confidence ($\ge 40\%$)**: Direct FAQ answer with match confidence badge.
  - **Moderate Confidence ($20\% - 40\%$)**: *"Did you mean: ...?"* confirmation with related question suggestions.
  - **Low Confidence ($< 20\%$)**: Graceful fallback with top related topics and category shortcuts.
- **Modern User Interfaces**:
  - 🌐 **Web UI**: Modern glassmorphic interface, dark/light mode toggle, dynamic typing animation, suggestion chips, confidence badges, and an interactive FAQ Knowledge Explorer modal.
  - 💻 **Interactive CLI**: Color-coded terminal chat with `/categories`, `/faqs <cat>`, and `/help` commands.
  - 🔌 **REST API**: Clean JSON endpoints (`/api/chat`, `/api/faqs`, `/api/categories`, `/api/stats`).

---

## 📁 Project Structure

```
faq_chatbot/
├── data/
│   └── faqs.json          # FAQ dataset with questions, alternates, tags, and categories
├── faq_engine.py          # NLP Preprocessing, TF-IDF vectorizer & Cosine Similarity engine
├── app.py                 # Flask Web application & REST API
├── cli.py                 # Interactive Command-Line Interface (terminal chat)
├── test_chatbot.py        # Automated test suite (13 unit tests)
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Web UI chat interface template
├── static/
│   ├── css/
│   │   └── style.css      # Modern styling with Dark/Light themes and animations
│   └── js/
│       └── chat.js        # Chat frontend controller and real-time API communication
└── README.md              # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Navigate to the project directory:**
   ```powershell
   cd C:\Users\SIDDU\.gemini\antigravity\scratch\faq_chatbot
   ```

2. **Install requirements:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify NLTK data:**
   The required NLTK corpora (`punkt`, `stopwords`, `wordnet`) are automatically downloaded on first run.

---

## 💻 Running the Chatbot

### Option 1: Web Interface (Recommended)

Start the Flask server:
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Option 2: Command-Line Interface (CLI)

For an interactive terminal chat:
```powershell
python cli.py
```
**CLI Commands:**
- Type any question directly: `Where is my order?`, `Can I return an opened item?`
- `/categories` : List all available FAQ categories
- `/faqs <category>` : View questions within a specific category
- `/help` : Display help menu
- `/quit` : Exit the chatbot

---

## 🧪 Running Unit Tests

Run the comprehensive unit test suite:
```powershell
python test_chatbot.py
```
Runs 13 test cases covering:
- Contraction expansion and text cleaning
- Tokenization and WordNet lemmatization
- Greeting, gratitude, farewell intent handling
- High-confidence cosine similarity matching
- Ambiguous query and gibberish fallback behavior

---

## ⚙️ How to Add or Customize FAQs

Open `data/faqs.json` and append a new JSON object:

```json
{
  "id": "custom-id-1",
  "category": "Your Category",
  "question": "What is the primary question?",
  "alternate_questions": [
    "Alternative phrasing 1",
    "Alternative phrasing 2"
  ],
  "answer": "This is the answer that the chatbot will display.",
  "tags": ["keyword1", "keyword2", "keyword3"]
}
```
The TF-IDF matrix will re-index automatically on application restart!
