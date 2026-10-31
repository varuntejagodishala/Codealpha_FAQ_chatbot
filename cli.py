"""
Interactive Command-Line Interface (CLI) for FAQ Chatbot
========================================================
Run with:
    python cli.py
"""

import os
import sys
from faq_engine import FAQChatbotEngine


# ANSI color codes for terminal formatting
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner(faq_count: int, categories: list):
    """Display welcome banner."""
    print(f"\n{CYAN}{BOLD}==============================================================={RESET}")
    print(f"{CYAN}{BOLD}       🤖 NovaTech Customer Support FAQ Chatbot (CLI)          {RESET}")
    print(f"{CYAN}{BOLD}==============================================================={RESET}")
    print(f"{DIM}Knowledge Base: {faq_count} FAQs across {len(categories)} categories{RESET}")
    print(f"{DIM}NLP Engine: NLTK Tokenization + Lemmatization + TF-IDF Cosine Similarity{RESET}")
    print(f"\nCommands:")
    print(f"  {YELLOW}/categories{RESET}  - List all FAQ categories")
    print(f"  {YELLOW}/faqs <cat>{RESET}  - Show questions in a category")
    print(f"  {YELLOW}/help{RESET}        - Show this help menu")
    print(f"  {YELLOW}/quit{RESET}        - Exit chat")
    print(f"{CYAN}---------------------------------------------------------------{RESET}\n")


def main():
    faq_path = os.path.join(os.path.dirname(__file__), "data", "faqs.json")
    if not os.path.exists(faq_path):
        print(f"{RED}Error: FAQ file not found at {faq_path}{RESET}")
        sys.exit(1)

    print(f"{DIM}Initializing NLP Preprocessor and vectorizing FAQs...{RESET}")
    engine = FAQChatbotEngine(faq_path)
    categories = engine.get_all_categories()

    print_banner(len(engine.faqs), categories)
    print(f"{GREEN}NovaTech Bot:{RESET} Hello! How can I assist you today? Ask any question about orders, shipping, returns, or warranty.\n")

    while True:
        try:
            user_input = input(f"{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{GREEN}NovaTech Bot:{RESET} Goodbye! Have a wonderful day! 👋\n")
            break

        if not user_input:
            continue

        # Handle CLI commands
        cmd = user_input.lower()
        if cmd in ["/quit", "/exit", "quit", "exit"]:
            print(f"\n{GREEN}NovaTech Bot:{RESET} Goodbye! Thank you for visiting NovaTech. 👋\n")
            break
        elif cmd in ["/categories", "categories"]:
            print(f"\n{MAGENTA}{BOLD}Available FAQ Categories:{RESET}")
            for i, cat in enumerate(categories, 1):
                count = len(engine.get_faqs_by_category(cat))
                print(f"  {i}. {cat} ({count} FAQs)")
            print()
            continue
        elif cmd.startswith("/faqs"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"{YELLOW}Usage: /faqs <category name>{RESET}\n")
                continue
            cat_query = parts[1].strip().lower()
            matching_cat = next((c for c in categories if cat_query in c.lower()), None)
            if not matching_cat:
                print(f"{RED}Category '{parts[1]}' not found. Type /categories to see all.{RESET}\n")
            else:
                cat_faqs = engine.get_faqs_by_category(matching_cat)
                print(f"\n{MAGENTA}{BOLD}FAQs in '{matching_cat}':{RESET}")
                for i, faq in enumerate(cat_faqs, 1):
                    print(f"  {i}. {faq['question']}")
                print()
            continue
        elif cmd in ["/help", "help"]:
            print_banner(len(engine.faqs), categories)
            continue

        # Process user question with FAQ Engine
        response = engine.find_best_match(user_input)

        res_type = response.get("type")
        confidence = response.get("confidence_percent")

        print()
        if res_type == "matched":
            matched_q = response.get("matched_question")
            cat = response.get("category")
            conf_level = response.get("confidence_level")
            conf_color = GREEN if conf_level == "high" else YELLOW

            print(f"{DIM}[Matched: \"{matched_q}\" | Category: {cat} | Confidence: {conf_color}{confidence}%{DIM}]{RESET}")
            print(f"{GREEN}{BOLD}NovaTech Bot:{RESET} {response.get('answer')}\n")

            suggestions = response.get("suggestions", [])
            if suggestions:
                print(f"{BLUE}Related Questions you might also ask:{RESET}")
                for s in suggestions:
                    print(f"  • {s}")
                print()

        elif res_type in ["greeting", "gratitude", "farewell", "help"]:
            print(f"{GREEN}{BOLD}NovaTech Bot:{RESET} {response.get('answer')}\n")
            suggestions = response.get("suggestions", [])
            if suggestions:
                print(f"{BLUE}Suggested Questions:{RESET}")
                for s in suggestions:
                    print(f"  • {s}")
                print()

        else:  # fallback
            print(f"{YELLOW}{BOLD}NovaTech Bot:{RESET} {response.get('answer')}\n")
            suggestions = response.get("suggestions", [])
            if suggestions:
                print(f"{BLUE}Suggested Topics:{RESET}")
                for s in suggestions:
                    print(f"  • {s}")
                print()


if __name__ == "__main__":
    main()
