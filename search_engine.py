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
            "Machine learning is a branch of artificial intelligence that enables computers to learn from data "
            "without being explicitly programmed. At its core, it's about finding patterns. Supervised learning "
            "uses labeled data to train models, while unsupervised learning finds hidden structure on its own. "
            "Reinforcement learning takes a different approach entirely, where an agent learns by trial and error "
            "through rewards and penalties. Neural networks, inspired loosely by the human brain, have become the "
            "backbone of modern ML. Deep learning stacks multiple layers to handle complex tasks like image "
            "recognition and language translation. Overfitting is one of the biggest challenges — when a model "
            "performs great on training data but fails on new data. Techniques like dropout, regularization, and "
            "cross-validation help address this. Tools like PyTorch and TensorFlow have made building models more "
            "accessible than ever, but understanding the math behind gradient descent and backpropagation still matters."
        ),
        "space_exploration.txt": (
            "Space exploration has always been driven by curiosity and the need to understand where we come from. "
            "The Apollo missions in the late 1960s were a turning point - humans actually walked on another world. "
            "Since then, robotic missions have done a lot of the heavy lifting. Rovers like Curiosity and Perseverance "
            "have been crawling across Mars, analyzing soil, searching for signs of ancient microbial life. "
            "SpaceX changed the game by making rocket boosters reusable, dramatically cutting the cost of getting "
            "to orbit. Now there's serious talk about permanent lunar bases and crewed Mars missions within this decade. "
            "The James Webb Space Telescope has been sending back images that push back our understanding of the early "
            "universe. Private companies are increasingly involved, which speeds things up but also raises questions "
            "about who owns what in space. One thing is clear — the next 20 years of space exploration will look "
            "nothing like the last 20."
        ),
        "street_food.txt": (
            "Street food is arguably the most honest form of cooking. No pretense, no tablecloths — just someone "
            "who has been making the same dish for years and knows exactly what they're doing. In Bangkok, pad thai "
            "from a street cart at midnight hits differently than any restaurant version. Mexico City's tacos al pastor, "
            "carved off a rotating spit, are a perfect food by almost any measure. Mumbai's vada pav — a spiced potato "
            "fritter in a bread roll — feeds millions of people every single day. What makes street food special is the "
            "repetition. These vendors make one or two things, thousands of times, until the process is completely dialed in. "
            "The best finds are usually not on any list. You follow the crowd, look for the longest queue of locals, "
            "and trust that someone who has been cooking the same dish for 30 years probably knows what they're doing. "
            "Traveling purely to eat street food is a completely legitimate way to see the world."
        ),
        "skateboarding.txt": (
            "Skateboarding started as something to do when the surf was flat. Kids in California bolted roller skate "
            "wheels to wooden planks and figured out the rest from there. By the 1980s it had its own culture, its own "
            "music, and its own way of seeing cities — every ledge, handrail, and stairset became something to skate. "
            "The ollie, invented by Alan Gelfand and refined by Rodney Mullen, unlocked everything. Once you can pop "
            "the board into the air without grabbing it, a whole world of tricks becomes possible. Street skating and "
            "vert skating developed their own communities and styles. The skateparks of today are incredible compared "
            "to what earlier generations had to work with. Skating became an Olympic sport in Tokyo 2020, which was "
            "controversial — a lot of skaters feel the whole point is that it exists outside mainstream sports. "
            "But younger skaters are pushing technical difficulty to levels that would have seemed impossible 10 years ago. "
            "The culture around it — the videos, the graphics, the music — is just as important as the tricks."
        ),
        "true_crime.txt": (
            "True crime taps into something deeply human — the need to understand how people end up doing terrible things. "
            "The genre has been around forever, but podcasts gave it a new life. Serial in 2014 was a turning point, "
            "bringing millions of people into a format that felt like following a case in real time. Since then the space "
            "has exploded. Some shows focus on investigation, walking through evidence and interviews. Others go deeper "
            "into the psychology — what environment, what circumstances, what mindset leads someone to cross that line. "
            "Cold cases have benefited hugely from public attention. DNA databases and amateur sleuths have helped crack "
            "cases that sat unsolved for decades. The Golden State Killer was identified partly because of genealogy websites. "
            "There's valid criticism too — that the genre sometimes sensationalizes tragedy or ignores the victims in favor "
            "of making the perpetrator interesting. The best true crime reporting keeps the humanity of everyone involved "
            "at the center. It's a fine line between informing and exploiting, and not every show gets it right."
        ),
        "music.txt": (
            "Music is one of the few things that works on you even when you're not paying attention. A song can pull "
            "you back to a specific moment years later with no warning. The way genres evolve is fascinating — jazz came "
            "from blues, rock came from jazz, hip hop flipped everything by making rhythm the main event and sampling "
            "existing records to build something new. Producers like J Dilla changed what beats could sound like. "
            "Artists like Radiohead and Kendrick Lamar pushed what albums could say. Streaming changed the economics "
            "completely. Artists now need millions of plays to earn what a few thousand album sales used to bring in, "
            "which has pushed a lot of musicians toward touring and merchandise just to survive. But it also meant that "
            "anyone with a laptop and an interface can release music to a global audience overnight. Bedroom pop and "
            "hyperpop both came out of that shift. Live music is still irreplaceable though. There's a version of a song "
            "that only exists in a specific room on a specific night, and that's never getting recorded. "
            "The relationship between music and identity runs deep — what you listen to at 17 tends to stick with you "
            "in ways that are hard to fully explain."
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