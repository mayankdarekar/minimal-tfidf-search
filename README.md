# Minimal TF-IDF Search Engine

A minimal Python-based text search engine that uses TF-IDF vectorization and cosine similarity to rank documents based on relevance.

This project demonstrates core Information Retrieval concepts including:

- Text preprocessing
- Term Frequency–Inverse Document Frequency (TF-IDF)
- Vector space model
- Cosine similarity
- Persistent search history

## Features

- Loads text files from a folder
- Builds TF-IDF vectors
- Ranks documents by relevance
- Interactive query mode
- Saves search history to JSON

## Run

```
python3 search_engine.py
```

## Commands

| Command | Description |
|---|---|
| `history` | View past searches |
| `clear history` | Wipe search history |
| `quit` | Exit |