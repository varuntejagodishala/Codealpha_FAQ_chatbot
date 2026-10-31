"""
Automated Test Suite for FAQ Chatbot
====================================
Tests preprocessing, TF-IDF vectorization, cosine similarity,
conversational intents, edge cases, and Flask REST API endpoints.
"""

import os
import unittest
from faq_engine import FAQChatbotEngine, NLPPreprocessor, expand_contractions
from app import app


class TestNLPPreprocessor(unittest.TestCase):
    """Test NLP cleaning, tokenization, stopword removal, and lemmatization."""

    def setUp(self):
        self.preprocessor = NLPPreprocessor()

    def test_contraction_expansion(self):
        text = "I can't track what's inside and won't arrive"
        expanded = expand_contractions(text)
        self.assertIn("cannot", expanded)
        self.assertIn("what is", expanded)
        self.assertIn("will not", expanded)

    def test_cleaning(self):
        raw = "Hello!! Check out https://example.com for 50% discount??"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("?", cleaned)

    def test_preprocessing_lemmatization(self):
        raw = "The customer is tracking multiple shipments and packages"
        preprocessed = self.preprocessor.preprocess(raw)
        tokens = preprocessed.split()
        # "shipments" -> "shipment", "packages" -> "package"
        self.assertTrue(any("shipment" in t or "package" in t for t in tokens))


class TestFAQChatbotEngine(unittest.TestCase):
    """Test FAQ matching and conversational logic."""

    @classmethod
    def setUpClass(cls):
        faq_path = os.path.join(os.path.dirname(__file__), "data", "faqs.json")
        cls.engine = FAQChatbotEngine(faq_path)

    def test_dataset_loaded(self):
        self.assertGreaterEqual(len(self.engine.faqs), 20)
        self.assertIsNotNone(self.engine.vectorizer)
        self.assertIsNotNone(self.engine.tfidf_matrix)

    def test_greetings_intent(self):
        res = self.engine.find_best_match("hello there")
        self.assertEqual(res.get("type"), "greeting")
        self.assertIn("NovaTech", res.get("answer"))

    def test_gratitude_intent(self):
        res = self.engine.find_best_match("thank you so much")
        self.assertEqual(res.get("type"), "gratitude")

    def test_farewell_intent(self):
        res = self.engine.find_best_match("goodbye")
        self.assertEqual(res.get("type"), "farewell")

    def test_track_order_match(self):
        res = self.engine.find_best_match("Where is my package right now?")
        self.assertEqual(res.get("type"), "matched")
        self.assertEqual(res.get("faq_id"), "order-track")
        self.assertIn("track your order", res.get("answer").lower())
        self.assertGreater(res.get("confidence"), 0.35)

    def test_return_policy_match(self):
        res = self.engine.find_best_match("Can I return an item after opening?")
        self.assertEqual(res.get("type"), "matched")
        self.assertIn("return", res.get("matched_question").lower())

    def test_damaged_item_match(self):
        res = self.engine.find_best_match("my product arrived broken and defective")
        self.assertEqual(res.get("type"), "matched")
        self.assertEqual(res.get("faq_id"), "damaged-product")

    def test_payment_methods_match(self):
        res = self.engine.find_best_match("Do you accept Apple Pay or PayPal?")
        self.assertEqual(res.get("type"), "matched")
        self.assertEqual(res.get("faq_id"), "payments-accepted")

    def test_student_discount_match(self):
        res = self.engine.find_best_match("Is there any discount for college students?")
        self.assertEqual(res.get("type"), "matched")
        self.assertEqual(res.get("faq_id"), "student-discount")

    def test_gibberish_fallback(self):
        res = self.engine.find_best_match("asdfqwerty zxcvbnm12345")
        self.assertIn(res.get("type"), ["fallback", "matched"])
        if res.get("type") == "matched":
            self.assertLess(res.get("confidence"), 0.25)


class TestFlaskAPI(unittest.TestCase):
    """Test Flask Web Application Endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_index_route(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"NovaTech", res.data)

    def test_api_chat_success(self):
        res = self.client.post("/api/chat", json={"message": "how do i track my shipment"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["response"]["type"], "matched")

    def test_api_chat_empty(self):
        res = self.client.post("/api/chat", json={"message": ""})
        self.assertEqual(res.status_code, 400)

    def test_api_categories(self):
        res = self.client.get("/api/categories")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertGreaterEqual(data["total_categories"], 5)

    def test_api_stats(self):
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertGreaterEqual(data["total_faqs"], 20)
        self.assertGreater(data["vocabulary_size"], 100)


if __name__ == "__main__":
    unittest.main()
