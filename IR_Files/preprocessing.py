"""
Preprocessing utilities for the SciFact vector space IR system.

Transform raw document or query text into normalized tokens:
1) lowercase
2) strip markup
3) remove punctuation/special characters and numeric tokens
4) whitespace tokenization
5) stopword removal
6) optional Porter stemming

Functions:
- load_stopwords: Load stopwords from HTML file
- preprocess_text: Core text preprocessing function
- preprocess_documents: Preprocess all documents in the corpus
- preprocess_queries: Preprocess all queries
"""

from __future__ import annotations

import re
from typing import List, Set, Dict, Any

from nltk.stem import PorterStemmer

# Precompile regexes for speed and determinism.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Keep only a–z and whitespace after lowercasing; drops punctuation, digits, symbols.
_NON_ALPHA_RE = re.compile(r"[^a-z\s]+")

_stemmer = PorterStemmer()


def preprocess_text(text: str, stopwords: Set[str], stem: bool = False) -> List[str]:
    """
    Clean and tokenize text for indexing or querying.

    Args:
        text: Raw input text.
        stopwords: Set of stopwords to remove.
        stem: If True, apply Porter stemming.

    Returns:
        List of normalized tokens (order preserved).
    """
    # 1) Normalize case.
    text = text.lower()

    # 2) Remove simple HTML/XML markup.
    text = _HTML_TAG_RE.sub(" ", text)

    # 3) Remove punctuation, digits, and other symbols; keep letters and spaces.
    text = _NON_ALPHA_RE.sub(" ", text)

    # 4) Tokenize on whitespace.
    tokens = text.split()

    # 5) Remove stopwords and empty tokens.
    tokens = [tok for tok in tokens if tok and tok not in stopwords]

    # 6) Optional Porter stemming.
    if stem:
        tokens = [_stemmer.stem(tok) for tok in tokens]

    return tokens


def load_stopwords(filepath: str) -> Set[str]:
    """
    Load stopwords from an HTML file (specifically the provided List of Stopwords.html).

    Args:
        filepath: Path to the HTML file containing stopwords.

    Returns:
        Set of stopword strings.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract text between <pre> tags
    match = re.search(r'<pre>(.*?)</pre>', content, re.DOTALL)
    if match:
        # Split by newlines and strip whitespace
        words = match.group(1).strip().split('\n')
        return set(word.strip().lower() for word in words if word.strip())

    return set()


def preprocess_documents(
    documents: List[Dict[str, Any]],
    stopwords: Set[str],
    stem: bool = False
) -> List[Dict[str, Any]]:
    """
    Preprocess all documents in the corpus.

    Input: List of parsed documents with 'DOCNO', 'HEAD', 'TEXT' fields
    Output: Same documents with added 'tokens' field containing preprocessed tokens

    Args:
        documents: List of document dictionaries from parser.
        stopwords: Set of stopwords to remove.
        stem: If True, apply Porter stemming.

    Returns:
        List of documents with 'tokens' field added.
    """
    for doc in documents:
        # Combine title (HEAD) and body text (TEXT) for full content
        title = doc.get('HEAD', '') or ''
        text = doc.get('TEXT', '') or ''
        full_text = title + ' ' + text

        doc['tokens'] = preprocess_text(full_text, stopwords, stem)

    return documents


def preprocess_queries(
    queries: List[Dict[str, Any]],
    stopwords: Set[str],
    stem: bool = False
) -> List[Dict[str, Any]]:
    """
    Preprocess all queries.

    Input: List of parsed queries with 'num', 'title' fields
    Output: Same queries with added 'tokens' field containing preprocessed tokens

    Args:
        queries: List of query dictionaries from parser.
        stopwords: Set of stopwords to remove.
        stem: If True, apply Porter stemming.

    Returns:
        List of queries with 'tokens' field added.
    """
    for query in queries:
        # Query text is in 'title' field (from parser.py)
        query_text = query.get('title', '') or ''
        query['tokens'] = preprocess_text(query_text, stopwords, stem)

    return queries
