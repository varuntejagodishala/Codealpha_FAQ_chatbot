"""
FAQ Chatbot Engine
==================
Core Natural Language Processing (NLP) and Similarity Matching Module.
Features:
- Text Cleaning, Normalization, Tokenization, Lemmatization using NLTK
- TF-IDF Vector Space representation with n-grams
- Cosine Similarity matching with confidence thresholding
- Rule-based intent recognition for greetings, identity, and pleasantries
- Alternate query indexation for high-accuracy retrieval
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Ensure NLTK resources are available
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


# Common English contractions dictionary
CONTRACTIONS: Dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "how's": "how is",
    "who's": "who is",
    "there's": "there is",
    "that's": "that is",
}


def expand_contractions(text: str) -> str:
    """Expand contractions in english text (e.g. can't -> cannot)."""
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text_lower)
    return text_lower


class NLPPreprocessor:
    """Handles text cleaning, tokenization, stopword removal, and lemmatization."""

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            # Fallback stopwords if NLTK corpus cannot be read
            self.stop_words = {
                "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
                "of", "is", "am", "are", "was", "were", "be", "been", "being", "have",
                "has", "had", "do", "does", "did", "it", "its", "this", "that", "these",
                "those", "my", "your", "his", "her", "their", "our", "me", "you", "him"
            }
        # Keep interrogative pronouns to retain question intent
        self.stop_words -= {"what", "when", "where", "how", "why", "who", "which"}

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Map POS tag to first character lemmatize() accepts."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def clean_text(self, text: str) -> str:
        """Strip HTML tags, URLs, special characters and lowercase."""
        if not text:
            return ""
        # Lowercase and expand contractions
        text = expand_contractions(text)
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove special characters / punctuation except alphanumeric
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words with fallback to regex."""
        try:
            return word_tokenize(text)
        except Exception:
            return re.findall(r"\b[a-zA-Z0-9]+\b", text)

    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline:
        1. Clean & lower
        2. Tokenize
        3. Remove stopwords
        4. Lemmatize tokens
        Returns preprocessed string joined by spaces.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        processed_tokens: List[str] = []
        for word in tokens:
            word_lower = word.lower()
            if word_lower not in self.stop_words and len(word_lower) > 1:
                try:
                    lemma = self.lemmatizer.lemmatize(word_lower)
                except Exception:
                    lemma = word_lower
                processed_tokens.append(lemma)

        return " ".join(processed_tokens)


class FAQChatbotEngine:
    """
    Core engine managing FAQ knowledge base, TF-IDF vectorization,
    cosine similarity matching, and conversational intent handling.
    """

    def __init__(self, faq_file_path: Optional[str] = None) -> None:
        self.preprocessor = NLPPreprocessor()
        self.faqs: List[Dict[str, Any]] = []
        # Documents indexed for vector space:
        # Each index entry: { "faq_idx": int, "text": str, "preprocessed": str, "source": str }
        self.indexed_docs: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        if faq_file_path and os.path.exists(faq_file_path):
            self.load_faqs(faq_file_path)

    def load_faqs(self, faq_file_path: str) -> int:
        """Load FAQs from JSON file and rebuild TF-IDF search index."""
        with open(faq_file_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self._build_index()
        return len(self.faqs)

    def _build_index(self) -> None:
        """
        Build search index mapping all primary questions, alternate phrasings,
        and tags back to the parent FAQ.
        """
        self.indexed_docs = []

        for idx, faq in enumerate(self.faqs):
            # 1. Primary question
            primary_q = faq.get("question", "")
            self.indexed_docs.append({
                "faq_idx": idx,
                "text": primary_q,
                "preprocessed": self.preprocessor.preprocess(primary_q),
                "source": "primary_question",
            })

            # 2. Alternate questions
            for alt_q in faq.get("alternate_questions", []):
                self.indexed_docs.append({
                    "faq_idx": idx,
                    "text": alt_q,
                    "preprocessed": self.preprocessor.preprocess(alt_q),
                    "source": "alternate_question",
                })

            # 3. Tags and Category combination
            tags = " ".join(faq.get("tags", []))
            category = faq.get("category", "")
            combined = f"{category} {tags}"
            self.indexed_docs.append({
                "faq_idx": idx,
                "text": combined,
                "preprocessed": self.preprocessor.preprocess(combined),
                "source": "tags_category",
            })

        # Fit TF-IDF Vectorizer
        corpus = [doc["preprocessed"] for doc in self.indexed_docs]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def check_conversational_intents(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Detect common conversational intents (greetings, identity, thanks, bye)."""
        cleaned = re.sub(r"[^a-zA-Z\s]", "", user_query.strip().lower())
        tokens = cleaned.split()
        full_text = " " + cleaned + " "

        # Greetings
        greeting_words = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"}
        if any(g == cleaned or f" {g} " in full_text for g in greeting_words):
            return {
                "type": "greeting",
                "answer": "Hello! 👋 Welcome to NovaTech Support. How can I help you today? You can ask me about tracking orders, return policies, shipping times, warranties, or payment methods.",
                "confidence": 1.0,
                "suggestions": [
                    "How can I track my order?",
                    "What is your return policy?",
                    "What are your delivery times?",
                    "What payment methods do you accept?"
                ]
            }

        # Gratitude
        thanks_words = {"thanks", "thank you", "thx", "appreciate it", "thank you so much"}
        if any(t == cleaned or f" {t} " in full_text for t in thanks_words):
            return {
                "type": "gratitude",
                "answer": "You're very welcome! 😊 Feel free to ask if there's anything else I can help you with.",
                "confidence": 1.0,
                "suggestions": []
            }

        # Farewell
        farewell_words = {"bye", "goodbye", "see you", "exit", "quit", "cya"}
        if any(b == cleaned or f" {b} " in full_text for b in farewell_words):
            return {
                "type": "farewell",
                "answer": "Goodbye! Have a wonderful day, and thank you for contacting NovaTech Support! 🚀",
                "confidence": 1.0,
                "suggestions": []
            }

        # Bot identity / Help
        help_words = {"who are you", "what can you do", "help me", "help", "what is your name"}
        if any(h in cleaned for h in help_words):
            return {
                "type": "help",
                "answer": "I am the **NovaTech FAQ Assistant** 🤖. I use NLP and semantic similarity to answer questions about orders, shipping, returns, warranties, discounts, and customer support. Simply type your question!",
                "confidence": 1.0,
                "suggestions": [
                    "Where is my package?",
                    "How do I initiate a return?",
                    "Do you offer student discounts?",
                    "How do I claim warranty?"
                ]
            }

        return None

    def find_best_match(
        self,
        user_query: str,
        high_confidence_threshold: float = 0.40,
        low_confidence_threshold: float = 0.20,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Match user query with FAQs using TF-IDF representation and Cosine Similarity.
        
        Returns:
            Dictionary containing matched FAQ, confidence score, status, and related suggestions.
        """
        # 1. First check rule-based conversational intents
        intent_response = self.check_conversational_intents(user_query)
        if intent_response:
            return intent_response

        # 2. Check if index is initialized
        if self.vectorizer is None or self.tfidf_matrix is None or not self.indexed_docs:
            return {
                "type": "error",
                "answer": "The FAQ knowledge base has not been loaded yet.",
                "confidence": 0.0,
                "suggestions": []
            }

        # 3. Preprocess user query
        preprocessed_query = self.preprocessor.preprocess(user_query)
        if not preprocessed_query.strip():
            return {
                "type": "fallback",
                "answer": "I didn't quite catch that. Could you please rephrase your question?",
                "confidence": 0.0,
                "suggestions": self.get_popular_questions(4)
            }

        # 4. Transform query into TF-IDF vector & calculate Cosine Similarities
        query_vec = self.vectorizer.transform([preprocessed_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Aggregate similarity scores per FAQ (since multiple doc entries map to one FAQ)
        faq_scores: Dict[int, float] = {}
        for doc_idx, sim_score in enumerate(similarities):
            faq_idx = self.indexed_docs[doc_idx]["faq_idx"]
            if faq_idx not in faq_scores or sim_score > faq_scores[faq_idx]:
                faq_scores[faq_idx] = float(sim_score)

        # Sort FAQs by highest similarity score
        sorted_faqs = sorted(faq_scores.items(), key=lambda item: item[1], reverse=True)

        if not sorted_faqs:
            return {
                "type": "fallback",
                "answer": "I'm sorry, I couldn't find an answer to your question. Would you like to reach our support team directly?",
                "confidence": 0.0,
                "suggestions": self.get_popular_questions(4)
            }

        best_faq_idx, best_score = sorted_faqs[0]
        best_faq = self.faqs[best_faq_idx]

        # Gather top alternate suggestions (excluding the best match)
        suggestions: List[str] = []
        for other_idx, other_score in sorted_faqs[1:top_k + 1]:
            if other_score > 0.10:
                suggestions.append(self.faqs[other_idx]["question"])

        # Format confidence as rounded float percentage
        confidence_percent = round(best_score * 100, 1)

        # Decision based on confidence thresholds
        if best_score >= high_confidence_threshold:
            return {
                "type": "matched",
                "faq_id": best_faq.get("id"),
                "category": best_faq.get("category"),
                "matched_question": best_faq.get("question"),
                "answer": best_faq.get("answer"),
                "confidence": round(best_score, 4),
                "confidence_percent": confidence_percent,
                "confidence_level": "high",
                "suggestions": suggestions[:3]
            }
        elif best_score >= low_confidence_threshold:
            return {
                "type": "matched",
                "faq_id": best_faq.get("id"),
                "category": best_faq.get("category"),
                "matched_question": best_faq.get("question"),
                "answer": f"**Did you mean: *\"{best_faq.get('question')}\"*?**\n\n{best_faq.get('answer')}",
                "confidence": round(best_score, 4),
                "confidence_percent": confidence_percent,
                "confidence_level": "moderate",
                "suggestions": suggestions[:3]
            }
        else:
            return {
                "type": "fallback",
                "answer": "I'm not completely certain I understand your question. Here are some topics that might help:",
                "confidence": round(best_score, 4),
                "confidence_percent": confidence_percent,
                "confidence_level": "low",
                "suggestions": suggestions[:3] if suggestions else self.get_popular_questions(4)
            }

    def get_popular_questions(self, count: int = 4) -> List[str]:
        """Return representative sample questions across categories."""
        if not self.faqs:
            return []
        step = max(1, len(self.faqs) // count)
        return [self.faqs[i]["question"] for i in range(0, len(self.faqs), step)][:count]

    def get_all_categories(self) -> List[str]:
        """Return unique sorted list of FAQ categories."""
        categories = {faq.get("category", "General") for faq in self.faqs}
        return sorted(list(categories))

    def get_faqs_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Return all FAQs belonging to a specific category."""
        return [faq for faq in self.faqs if faq.get("category") == category]
