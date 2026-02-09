import time
import os
import copy
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
preprocessed_queries_titles_path = os.path.join(BASE_DIR, 'preprocessed_queries_titles.json')
preprocessed_queries_fulltext_path = os.path.join(BASE_DIR, 'preprocessed_queries_fulltext.json')

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

print("Preparing queries for two runs (titles_only and titles_plus_fulltext)")
all_queries = parse_queries_from_file(query_file_path)
test_queries = [q for q in all_queries if int(q['num']) % 2 == 1]
print(f"Filtered to {len(test_queries)} test queries (odd IDs only)")

if os.path.exists(preprocessed_queries_titles_path):
    print("Loading preprocessed title-only queries")
    queries_titles = load_preprocessed_data(preprocessed_queries_titles_path)
else:
    print("Preprocessing title-only queries")
    queries_titles = preprocess_queries(
        copy.deepcopy(test_queries), stopwords, stem=USE_STEMMING, query_field='title'
    )
    save_preprocessed_data(queries_titles, preprocessed_queries_titles_path)

if os.path.exists(preprocessed_queries_fulltext_path):
    print("Loading preprocessed title+fulltext queries")
    queries_fulltext = load_preprocessed_data(preprocessed_queries_fulltext_path)
else:
    print("Preprocessing title+fulltext queries")
    queries_fulltext = preprocess_queries(
        copy.deepcopy(test_queries), stopwords, stem=USE_STEMMING, query_field='full_text'
    )
    save_preprocessed_data(queries_fulltext, preprocessed_queries_fulltext_path)

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

queries_titles_sorted = sorted(queries_titles, key=lambda q: int(q['num']))
queries_fulltext_sorted = sorted(queries_fulltext, key=lambda q: int(q['num']))

def write_results(queries_sorted, results_file, run_name):
    with open(results_file, 'w', encoding='utf-8') as output_file:
        for query in queries_sorted:
            query_id = query['num']
            ranked_docs = vsm.rank_documents(query.get('tokens', []), top_k=100)
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                output_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

print("Ranking and writing results for title-only queries")
write_results(queries_titles_sorted, "Results_titles_only", "vsm_tfidf_titles")

print("Ranking and writing results for title+fulltext queries")
write_results(queries_fulltext_sorted, "Results_titles_fulltext", "vsm_tfidf_fulltext")

# Keep Assignment-required file name as the final run output.
final_results_file = "Results"
write_results(queries_fulltext_sorted, final_results_file, "vsm_tfidf_fulltext")

print("Ranking results written to Results_titles_only, Results_titles_fulltext, and Results")
