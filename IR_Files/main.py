import time
import os
from parser import parse_documents_from_file, parse_queries_from_file
from preprocessing import load_stopwords, preprocess_documents, preprocess_queries
from indexing import (
    build_inverted_index,
    calculate_document_frequencies,
    calculate_document_lengths,
    get_corpus_size,
    save_inverted_index,
    load_inverted_index,
)
from ranking import VectorSpaceModel
from utils import save_preprocessed_data, load_preprocessed_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset = "scifact"  # SciFact dataset for Assignment 1
dataset_path = os.path.join(BASE_DIR, "..", dataset)
doc_folder_path = os.path.join(dataset_path, 'corpus.jsonl')
query_file_path = os.path.join(dataset_path, 'queries.jsonl')
stopwords_path = os.path.join(BASE_DIR, "..", 'List of Stopwords.html')
index_file_path = os.path.join(BASE_DIR, 'inverted_index.json')
preprocessed_docs_path = os.path.join(BASE_DIR, 'preprocessed_documents.json')
preprocessed_queries_path = os.path.join(BASE_DIR, 'preprocessed_queries.json')

# Preprocessing settings
USE_STEMMING = True  # Set to True to enable Porter stemming

start_time = time.time()

# Load stopwords
print("Loading stopwords")
stopwords = load_stopwords(stopwords_path)
print(f"Loaded {len(stopwords)} stopwords")

print("Parsing documents")
documents = []

# Preprocess documents and queries
if os.path.exists(preprocessed_docs_path):
    print("Loading preprocessed documents")
    documents = load_preprocessed_data(preprocessed_docs_path)
else:
    print("Preprocessing documents")
    documents = parse_documents_from_file(doc_folder_path)
    documents = preprocess_documents(documents, stopwords, stem=USE_STEMMING)
    save_preprocessed_data(documents, preprocessed_docs_path)

if os.path.exists(preprocessed_queries_path):
    print("Loading preprocessed queries")
    queries = load_preprocessed_data(preprocessed_queries_path)
else:
    print("Preprocessing queries")
    # Parse all queries first
    all_queries = parse_queries_from_file(query_file_path)
    # Filter for TEST queries only (odd-numbered: 1, 3, 5, ...)
    queries = [q for q in all_queries if int(q['num']) % 2 == 1]
    print(f"Filtered to {len(queries)} test queries (odd IDs only)")
    queries = preprocess_queries(queries, stopwords, stem=USE_STEMMING)
    save_preprocessed_data(queries, preprocessed_queries_path)

start_time = time.time()

# Build or load inverted index
try:
    inverted_index, doc_freqs, doc_lengths = load_inverted_index(index_file_path)
    print("Inverted index loaded successfully.")
except FileNotFoundError:
    print("Inverted index not found, building a new one.")
    inverted_index = build_inverted_index(documents)
    doc_freqs = calculate_document_frequencies(inverted_index)
    doc_lengths = calculate_document_lengths(documents)
    save_inverted_index(inverted_index, doc_freqs, doc_lengths, index_file_path)
    end_time = time.time()
    print(f"Time taken to build inverted index: {end_time - start_time:.2f} seconds")

print("Initializing vector space model")
vsm = VectorSpaceModel(inverted_index, doc_freqs, doc_lengths)

# Ensure queries are processed in ascending order of ID
queries_sorted = sorted(queries, key=lambda q: int(q['num']))

results_file = "Results"
run_name = "vsm_tfidf"

print("Ranking and writing to results file")
with open(results_file, 'w', encoding='utf-8') as output_file:
    for query in queries_sorted:
        query_id = query['num']
        ranked_docs = vsm.rank_documents(query.get('tokens', []), top_k=100)
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            output_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

print(f"Ranking results written to {results_file}")
