import math
from collections import Counter, defaultdict

class BM25:
    def __init__(self, inverted_index, doc_lengths, k1=1.5, b=0.75, avgdl=None):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.k1 = k1
        self.b = b
        self.avgdl = avgdl if avgdl is not None else sum(doc_lengths.values()) / len(doc_lengths)
        self.N = len(doc_lengths)  # Total number of documents

    def idf(self, term):
        df = len(self.inverted_index.get(term, {}))
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def bm25_term_score(self, tf, df, doc_length):
        idf_value = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        return idf_value * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl))
    
    def search(self, corpus, queries, top_k=1000):
        """
        Search method compatible with BEIR framework
        """
        results = {}
        for query_id, query in queries.items():
            ranked_docs = self.rank_documents(query)
            results[query_id] = {doc_id: score for doc_id, score in ranked_docs[:top_k]}
        return results    

    def rank_documents(self, query_terms):
        """
        Rank documents according to their relevance to a given set of query terms using BM25
        """
        scores = defaultdict(float)
        query_term_counts = Counter(query_terms)
        for term in query_term_counts:
            postings = self.inverted_index.get(term, {})
            df = len(postings)
            for doc_id, tf in postings.items():
                doc_length = self.doc_lengths[doc_id]
                scores[doc_id] += self.bm25_term_score(tf, df, doc_length)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class VectorSpaceModel:
    """
    Classic TF-IDF vector space model with cosine similarity.
    """
    def __init__(self, inverted_index, doc_freqs, doc_lengths):
        self.inverted_index = inverted_index
        self.doc_freqs = doc_freqs
        self.doc_lengths = doc_lengths
        self.N = len(doc_lengths)
        self.doc_norms = self._compute_document_norms()

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log((self.N + 1) / (df + 1)) + 1

    @staticmethod
    def _tf_weight(tf: int) -> float:
        return 1 + math.log(tf) if tf > 0 else 0.0

    def _compute_document_norms(self):
        norms = defaultdict(float)
        for term, postings in self.inverted_index.items():
            idf = self._idf(term)
            for doc_id, tf in postings.items():
                weight = self._tf_weight(tf) * idf
                norms[doc_id] += weight * weight
        return {doc_id: math.sqrt(weight_sum) for doc_id, weight_sum in norms.items()}

    def rank_documents(self, query_tokens, top_k=100):
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        query_weights = {}
        for term, tf in query_tf.items():
            query_weights[term] = self._tf_weight(tf) * self._idf(term)

        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
        if query_norm == 0:
            return []

        scores = defaultdict(float)
        for term, q_weight in query_weights.items():
            postings = self.inverted_index.get(term, {})
            idf = self._idf(term)
            for doc_id, tf in postings.items():
                d_weight = self._tf_weight(tf) * idf
                scores[doc_id] += q_weight * d_weight

        ranked = []
        for doc_id, dot in scores.items():
            d_norm = self.doc_norms.get(doc_id)
            if d_norm:
                ranked.append((doc_id, dot / (d_norm * query_norm)))

        ranked.sort(key=lambda item: item[1], reverse=True)

        if top_k is not None and len(ranked) < top_k:
            needed = top_k - len(ranked)
            for doc_id in self.doc_lengths:
                if doc_id in scores:
                    continue
                ranked.append((doc_id, 0.0))
                needed -= 1
                if needed == 0:
                    break

        return ranked[:top_k]

def normalize_scores(ranked_docs):
        if not ranked_docs:
            return []
        max_score = max(score for _, score in ranked_docs)
        min_score = min(score for _, score in ranked_docs)
        if max_score == min_score:
            return [(doc_id, 1.0) for doc_id, _ in ranked_docs]
        return [(doc_id, (score - min_score) / (max_score - min_score)) for doc_id, score in ranked_docs]
