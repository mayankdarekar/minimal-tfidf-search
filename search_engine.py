"""
Simple TF-IDF Search Engine
============================
Loads .txt documents from a folder, builds TF-IDF vectors,
and returns the top 5 most relevant documents for a user query.

Features:
  - TF-IDF ranking with cosine similarity
  - Search history saved to search_history.json
  - Type 'history' during search to view past queries
  - Type 'clear history' to wipe the history file

Usage:
    python search_engine.py --docs ./docs --query "your search query"

    Or run interactively (it will prompt for a query):
    python search_engine.py --docs ./docs

    View saved history without searching:
    python search_engine.py --show-history
"""

import os
import re
import math
import json
import argparse
from datetime import datetime
from collections import defaultdict


# ──────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ──────────────────────────────────────────────

# Common words that carry little meaning (stop words)
STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "are",
    "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "its",
    "i", "you", "he", "she", "we", "they", "my", "your", "our", "as",
    "from", "by", "about", "into", "through", "so", "if", "then",
}

def preprocess(text: str) -> list[str]:
    """
    Clean and tokenize text:
      1. Lowercase everything
      2. Remove punctuation/numbers
      3. Split into words (tokens)
      4. Remove stop words and short tokens
    """
    text = text.lower()                          # lowercase
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters
    tokens = text.split()                        # split on whitespace
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return tokens


# ──────────────────────────────────────────────
# 2. LOAD DOCUMENTS
# ──────────────────────────────────────────────

def load_documents(folder: str) -> dict[str, str]:
    """
    Read all .txt files from `folder`.
    Returns a dict: { filename: raw_text }
    """
    docs = {}
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: '{folder}'")

    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs[fname] = f.read()

    if not docs:
        raise ValueError(f"No .txt files found in '{folder}'")

    print(f"Loaded {len(docs)} document(s): {list(docs.keys())}\n")
    return docs


# ──────────────────────────────────────────────
# 3. BUILD TF-IDF
# ──────────────────────────────────────────────

def compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Term Frequency (TF) = (count of term in doc) / (total terms in doc)
    Measures how often a word appears in a single document.
    """
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    total = len(tokens)
    return {word: count / total for word, count in tf.items()}


def compute_idf(tokenized_docs: dict[str, list[str]]) -> dict[str, float]:
    """
    Inverse Document Frequency (IDF) = log(N / df)
      N  = total number of documents
      df = number of documents containing the term

    Words appearing in many documents get a lower IDF score (less unique).
    """
    N = len(tokenized_docs)
    df = defaultdict(int)   # document frequency per term

    for tokens in tokenized_docs.values():
        for term in set(tokens):    # use set to count doc, not occurrences
            df[term] += 1

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(N / freq) + 1   # +1 avoids zero for common terms
    return idf


def compute_tfidf_vectors(
    tokenized_docs: dict[str, list[str]],
    idf: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    TF-IDF(term, doc) = TF(term, doc) × IDF(term)

    Builds a vector for each document: { term: tfidf_score }
    Higher score → the term is important AND relatively rare across docs.
    """
    vectors = {}
    for fname, tokens in tokenized_docs.items():
        tf = compute_tf(tokens)
        vectors[fname] = {term: tf_val * idf.get(term, 0)
                          for term, tf_val in tf.items()}
    return vectors


# ──────────────────────────────────────────────
# 4. COSINE SIMILARITY
# ──────────────────────────────────────────────

def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """
    Cosine similarity measures the angle between two vectors.
      = (A · B) / (|A| × |B|)

    Result is between 0 (no overlap) and 1 (identical direction).
    We use this to compare the query vector against each document vector.
    """
    # Dot product: sum of products of shared terms
    shared_terms = set(vec_a) & set(vec_b)
    dot_product = sum(vec_a[t] * vec_b[t] for t in shared_terms)

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


# ──────────────────────────────────────────────
# 5. SEARCH
# ──────────────────────────────────────────────

def search(
    query: str,
    doc_vectors: dict[str, dict[str, float]],
    idf: dict[str, float],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    1. Preprocess and vectorize the query (same pipeline as documents)
    2. Compute cosine similarity between query vector and every doc vector
    3. Return top_n results sorted by score (highest first)
    """
    query_tokens = preprocess(query)
    if not query_tokens:
        print("Query is empty after preprocessing.")
        return []

    # Build TF-IDF vector for the query
    query_tf = compute_tf(query_tokens)
    query_vector = {term: tf_val * idf.get(term, 1.0)
                    for term, tf_val in query_tf.items()}

    # Score each document
    scores = {
        fname: cosine_similarity(query_vector, doc_vec)
        for fname, doc_vec in doc_vectors.items()
    }

    # Sort descending by score, return top N
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(fname, score) for fname, score in ranked if score > 0][:top_n]


# ──────────────────────────────────────────────
# 6. SEARCH HISTORY
# ──────────────────────────────────────────────

# Where the history file lives (same directory as the script)
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_history.json")


def load_history() -> list[dict]:
    """
    Load search history from the JSON file.
    Returns a list of history entries, or an empty list if none exist yet.
    Each entry looks like:
      { "query": "...", "timestamp": "2024-01-01 12:00:00", "result_count": 3 }
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # If the file is corrupted or unreadable, start fresh
        return []


def save_to_history(query: str, result_count: int) -> None:
    """
    Append a new search entry to the history file.
    Creates the file if it doesn't exist yet.
    """
    history = load_history()

    # Build the new entry
    entry = {
        "query": query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result_count": result_count,
    }
    history.append(entry)

    # Write the updated list back to the file
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def display_history(last_n: int = 10) -> None:
    """
    Print the most recent `last_n` search queries in a readable table.
    """
    history = load_history()

    if not history:
        print("\n  No search history yet.\n")
        return

    # Show only the last N entries
    recent = history[-last_n:]

    print(f"\n{'─'*55}")
    print(f"  Search History  (showing last {len(recent)} of {len(history)} total)")
    print(f"{'─'*55}")
    print(f"  {'#':<4} {'Timestamp':<22} {'Results':<8} Query")
    print(f"  {'─'*4} {'─'*20} {'─'*7} {'─'*20}")

    for i, entry in enumerate(recent, start=1):
        print(f"  {i:<4} {entry['timestamp']:<22} {entry['result_count']:<8} {entry['query']}")

    print(f"{'─'*55}\n")
    print(f"  History saved at: {HISTORY_FILE}\n")


def clear_history() -> None:
    """Delete all saved search history by overwriting the file with an empty list."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
    print("  Search history cleared.\n")


# ──────────────────────────────────────────────
# 7. DISPLAY RESULTS
# ──────────────────────────────────────────────

def display_results(results: list[tuple[str, float]], docs: dict[str, str]) -> None:
    """Pretty-print ranked search results with a short document preview."""
    if not results:
        print("No matching documents found.")
        return

    print(f"\n{'─'*50}")
    print(f"  Top {len(results)} Result(s)")
    print(f"{'─'*50}")

    for rank, (fname, score) in enumerate(results, start=1):
        preview = docs[fname][:200].replace("\n", " ").strip()
        print(f"\n#{rank}  [{fname}]  (score: {score:.4f})")
        print(f"    Preview: {preview}...")

    print(f"\n{'─'*50}\n")


# ──────────────────────────────────────────────
# 8. DEMO DATA (auto-created if no folder given)
# ──────────────────────────────────────────────

def create_demo_docs(folder: str = "./demo_docs") -> str:
    """Create sample .txt files so you can try the engine right away."""
    os.makedirs(folder, exist_ok=True)
    samples = {
        "python.txt": (
            "Python is a high-level programming language known for its simplicity "
            "and readability. It is widely used in web development, data science, "
            "machine learning, and automation. Python has a large standard library."
        ),
        "machine_learning.txt": (
            "Machine learning is a branch of artificial intelligence that enables "
            "computers to learn from data without being explicitly programmed. "
            "Algorithms like decision trees, neural networks, and support vector "
            "machines are popular in machine learning."
        ),
        "databases.txt": (
            "A database is an organized collection of structured information or data. "
            "SQL databases use structured query language for managing relational data. "
            "NoSQL databases like MongoDB store unstructured or semi-structured data."
        ),
        "web_development.txt": (
            "Web development involves building websites and web applications. "
            "Frontend development uses HTML, CSS, and JavaScript. Backend development "
            "uses server-side languages like Python, Node.js, or Ruby. APIs connect "
            "frontend and backend systems."
        ),
        "data_science.txt": (
            "Data science combines statistics, programming, and domain expertise to "
            "extract insights from data. Data scientists use tools like Python, R, "
            "and SQL. Visualization libraries like matplotlib and seaborn help "
            "communicate findings clearly."
        ),
    }
    for fname, content in samples.items():
        with open(os.path.join(folder, fname), "w") as f:
            f.write(content)
    print(f"Demo documents created in '{folder}/'")
    return folder


# ──────────────────────────────────────────────
# 9. MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Simple TF-IDF Search Engine")
    parser.add_argument("--docs", type=str, default=None,
                        help="Path to folder containing .txt documents")
    parser.add_argument("--query", type=str, default=None,
                        help="Search query (if omitted, interactive mode)")
    parser.add_argument("--show-history", action="store_true",
                        help="Print saved search history and exit")
    args = parser.parse_args()

    # ── Show history and exit early if requested ──
    if args.show_history:
        display_history()
        return

    # Use demo documents if no folder provided
    docs_folder = args.docs if args.docs else create_demo_docs()

    # ── Pipeline ──────────────────────────────
    # Step 1: Load raw documents
    docs = load_documents(docs_folder)

    # Step 2: Preprocess (tokenize + clean) each document
    tokenized = {fname: preprocess(text) for fname, text in docs.items()}

    # Step 3: Compute IDF across all documents
    idf = compute_idf(tokenized)

    # Step 4: Build TF-IDF vector for each document
    doc_vectors = compute_tfidf_vectors(tokenized, idf)

    print("Search engine ready!")
    print("Commands: 'history' → view past searches | 'clear history' → wipe history | 'quit' → exit\n")

    # ── Show recent history on startup so the user knows what they've searched ──
    history = load_history()
    if history:
        print(f"  💡 You have {len(history)} past search(es). Type 'history' to view them.\n")

    # ── Query loop ────────────────────────────
    while True:
        query = args.query if args.query else input("Enter search query: ").strip()

        # ── Special commands ──────────────────
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "history":
            display_history()
            continue

        if query.lower() == "clear history":
            clear_history()
            continue

        # ── Normal search ─────────────────────
        results = search(query, doc_vectors, idf, top_n=5)
        display_results(results, docs)

        # Save this query to history (only if it produced tokens)
        if preprocess(query):
            save_to_history(query, result_count=len(results))

        # If query was passed as argument, don't loop
        if args.query:
            break


if __name__ == "__main__":
    main()