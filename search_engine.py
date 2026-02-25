import os
import re
import math
import json
import argparse
import difflib
from datetime import datetime
from collections import defaultdict


STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "are",
    "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "its",
    "i", "you", "he", "she", "we", "they", "my", "your", "our", "as",
    "from", "by", "about", "into", "through", "so", "if", "then",
}

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_history.json")


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def load_documents(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: '{folder}'")

    docs = {}
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                docs[fname] = f.read()

    if not docs:
        raise ValueError(f"No .txt files found in '{folder}'")

    print(f"Loaded {len(docs)} doc(s): {list(docs.keys())}\n")
    return docs


def compute_tf(tokens):
    tf = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    total = len(tokens)
    return {word: count / total for word, count in tf.items()}


def compute_idf(tokenized_docs):
    N = len(tokenized_docs)
    df = defaultdict(int)
    for tokens in tokenized_docs.values():
        for term in set(tokens):
            df[term] += 1
    return {term: math.log(N / freq) + 1 for term, freq in df.items()}


def compute_tfidf_vectors(tokenized_docs, idf):
    vectors = {}
    for fname, tokens in tokenized_docs.items():
        tf = compute_tf(tokens)
        vectors[fname] = {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}
    return vectors


def cosine_similarity(vec_a, vec_b):
    shared = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def search(query, doc_vectors, idf, top_n=5):
    tokens = preprocess(query)
    if not tokens:
        print("Query is empty after preprocessing.")
        return []

    tf = compute_tf(tokens)
    query_vec = {term: val * idf.get(term, 1.0) for term, val in tf.items()}

    scores = {fname: cosine_similarity(query_vec, dvec) for fname, dvec in doc_vectors.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(fname, score) for fname, score in ranked if score > 0][:top_n]


def suggest(query, idf):
    tokens = preprocess(query)
    vocab = list(idf.keys())
    results = []
    seen = set()

    for token in tokens:
        if token not in vocab:
            for match in difflib.get_close_matches(token, vocab, n=3, cutoff=0.6):
                if match not in seen:
                    seen.add(match)
                    results.append(match)

    if results:
        print(f"  Did you mean: {', '.join(results[:3])} ?")
    else:
        print("  No similar terms found. Try different keywords.")


def display_results(results, docs, query="", idf=None):
    if not results:
        print("\n  No matching documents found.")
        if query and idf:
            suggest(query, idf)
        print()
        return

    print(f"\nTop {len(results)} result(s):")
    for rank, (fname, score) in enumerate(results, start=1):
        preview = docs[fname][:200].replace("\n", " ").strip()
        print(f"\n  #{rank} [{fname}] (score: {score:.4f})")
        print(f"     {preview}...")
    print()


# --- history ---

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_to_history(query, result_count):
    history = load_history()
    history.append({
        "query": query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result_count": result_count,
    })
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def display_history(last_n=10):
    history = load_history()
    if not history:
        print("\n  No search history yet.\n")
        return

    recent = history[-last_n:]
    print(f"\n  Search History (last {len(recent)} of {len(history)})")
    print(f"  {'#':<4} {'Timestamp':<22} {'Results':<8} Query")
    for i, entry in enumerate(recent, start=1):
        print(f"  {i:<4} {entry['timestamp']:<22} {entry['result_count']:<8} {entry['query']}")
    print()


def clear_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
    print("  History cleared.\n")


# --- demo docs ---

def create_demo_docs(folder="./demo_docs"):
    os.makedirs(folder, exist_ok=True)
    samples = {
        "machine_learning.txt": (
            "Machine learning is a branch of artificial intelligence that enables "
            "computers to learn from data without being explicitly programmed. "
            "Algorithms like decision trees, neural networks, and support vector "
            "machines are popular in machine learning."
        ),
        "space_exploration.txt": (
            "Space exploration has led to some of the most significant scientific discoveries "
            "in human history. NASA and SpaceX are pushing the boundaries of travel beyond "
            "Earth's orbit. Mars missions, lunar bases, and reusable rockets are shaping "
            "the future of how humans live and travel in space."
        ),
        "street_food.txt": (
            "Street food culture is one of the most authentic ways to experience a city. "
            "From tacos in Mexico City to ramen stalls in Tokyo, street vendors have perfected "
            "recipes passed down through generations. The best meals are often found in the "
            "smallest stalls down the narrowest alleys."
        ),
        "skateboarding.txt": (
            "Skateboarding originated in California in the 1950s and has grown into a global "
            "sport and lifestyle. Tricks like the ollie, kickflip, and grind are fundamental "
            "to street skating. The culture is deeply tied to music, art, and fashion and "
            "became an Olympic sport in 2021."
        ),
        "true_crime.txt": (
            "True crime has become one of the most popular podcast and documentary genres. "
            "Listeners are drawn to the psychology behind criminal behavior and the stories "
            "of investigators who crack cold cases. Shows like Serial and Making a Murderer "
            "sparked mainstream interest in unsolved mysteries and wrongful convictions."
        ),
    }
    for fname, content in samples.items():
        with open(os.path.join(folder, fname), "w") as f:
            f.write(content)
    print(f"Demo docs created in '{folder}/'")
    return folder


# --- main ---

def main():
    parser = argparse.ArgumentParser(description="TF-IDF Search Engine")
    parser.add_argument("--docs", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--show-history", action="store_true")
    args = parser.parse_args()

    if args.show_history:
        display_history()
        return

    docs_folder = args.docs if args.docs else create_demo_docs()
    docs = load_documents(docs_folder)

    tokenized = {fname: preprocess(text) for fname, text in docs.items()}
    idf = compute_idf(tokenized)
    doc_vectors = compute_tfidf_vectors(tokenized, idf)

    print("Ready. Commands: history | clear history | quit\n")

    history = load_history()
    if history:
        print(f"  {len(history)} past search(es). Type 'history' to view.\n")

    while True:
        query = args.query if args.query else input("Search: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "history":
            display_history()
            continue
        if query.lower() == "clear history":
            clear_history()
            continue

        results = search(query, doc_vectors, idf, top_n=5)
        display_results(results, docs, query=query, idf=idf)

        if preprocess(query):
            save_to_history(query, result_count=len(results))

        if args.query:
            break


if __name__ == "__main__":
    main()