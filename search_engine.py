"""
Simple TF-IDF Search Engine
============================
Loads .txt documents from a folder, builds TF-IDF vectors,
and returns the top 5 most relevant documents for a user query.

Usage:
    python search_engine.py --docs ./docs --query "your search query"

    Or run interactively (it will prompt for a query):
    python search_engine.py --docs ./docs
"""

import os
import re
import math
import argparse
from collections import defaultdict

# 1. TEXT PREPROCESSING

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


# 2. LOAD DOCUMENTS


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


# 3. BUILD TF-IDF


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


# 4. COSINE SIMILARITY

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



# 5. SEARCH

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


# 6. DISPLAY RESULTS

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



# 7. DEMO DATA (auto-created if no folder given)

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


# 8. MAIN

def main():
    parser = argparse.ArgumentParser(description="Simple TF-IDF Search Engine")
    parser.add_argument("--docs", type=str, default=None,
                        help="Path to folder containing .txt documents")
    parser.add_argument("--query", type=str, default=None,
                        help="Search query (if omitted, interactive mode)")
    args = parser.parse_args()

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

    print("Search engine ready! Type 'quit' to exit.\n")

    # ── Query loop ────────────────────────────
    while True:
        query = args.query if args.query else input("Enter search query: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        results = search(query, doc_vectors, idf, top_n=5)
        display_results(results, docs)

        # If query was passed as argument, don't loop
        if args.query:
            break


if __name__ == "__main__":
    main()