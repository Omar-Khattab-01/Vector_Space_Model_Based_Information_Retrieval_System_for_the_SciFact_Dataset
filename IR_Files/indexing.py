from collections import defaultdict
import json

def build_inverted_index(documents):
    """
    Build an inverted index from the preprocessed documents
    """
    inverted_index = defaultdict(dict)
    for doc in documents:
        doc_id = doc['DOCNO']
        tokens = doc.get('tokens', [])
        for token in tokens:
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = 0
            inverted_index[token][doc_id] += 1
    return inverted_index 



def calculate_document_lengths(documents):
    """
    Calculate the length of each document based on the number of terms
    """
    doc_lengths = {}
    for doc in documents:
        doc_id = doc['DOCNO']
        doc_lengths[doc_id] = len(doc.get('tokens', []))
    return doc_lengths

def calculate_document_frequencies(inverted_index):
    """
    Calculate document frequency for each term in the inverted index
    """
    doc_freqs = {}
    for term, postings in inverted_index.items():
        doc_freqs[term] = len(postings)
    return doc_freqs

def get_corpus_size(documents):
    """
    Get the total number of documents in the corpus
    """
    return len(documents)

def save_inverted_index(inverted_index, doc_freqs, doc_lengths, file_path):
    index_data = {
        'inverted_index': inverted_index,
        'doc_freqs': doc_freqs,
        'doc_lengths': doc_lengths
    }
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(index_data, file)

def load_inverted_index(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        inverted_data = json.load(file)
    return (inverted_data['inverted_index'],
            inverted_data['doc_freqs'],
            inverted_data['doc_lengths'])
