# CSI4107 Assignment 1 — Information Retrieval System (SciFact)

## Team
- Omar Khattab — 300202727
- Dongshi Li — 300294775
- Rhadesh Luchmun — 300184364
- Mustafa Ahmed — 300242013


Task split:
- Dongshi: preprocessing pipeline (stopwords, stemming) and tokenization tuning.
- Rhadesh: inverted index construction and TF–IDF cosine retrieval logic.
- Omar: data preparation, caching, and path/config wiring for reproducibility.
- Mustafa: evaluation with trec_eval, Results validation, and report/README drafting.

## Dataset & Queries
- Corpus: SciFact `corpus.jsonl` (7,917 KB), claims + abstracts.
- Queries: SciFact `queries.jsonl`; only odd-numbered test queries used.
- Qrels: `scifact/qrels/test.tsv` converted to `IR_Files/test.qrels` for evaluation.

## Method
### Preprocessing (Step 1)
- Lowercasing, HTML tag removal, strip punctuation/digits via regex.
- Tokenization on whitespace.
- Stopword removal using provided *List of Stopwords.html* (779 words).
- Porter stemming enabled for both documents and queries.
- Applied identically to documents (`HEAD` + `TEXT`) and queries (`title`).

### Indexing (Step 2)
- Inverted index term → {doc_id: term frequency} built from preprocessed tokens.
- Document lengths stored as token counts; document frequencies computed for all terms.
- Vocabulary size: **20,576** unique terms.
- Sample 100 vocabulary terms (sorted):
```
aminoglycosid, audiometr, auscultatori, bacilli, bap, celebr, chimaera, cinahl, coalesc, commonsens, cornerston, corneum, crack, cubic, cuticl, daphnia, democraci, dilat, dogma, dysplast, endotox, esoter, evening, evoc, fenfluramin, fiber, frobeni, glucokinas, glucosyl, glutamicum, gnpda, guanosin, herniat, hmsc, homolog, hydroperoxid, iea, ihsc, immunohistochemistri, imprint, inanob, indazol, instrument, intrathym, iodoacet, iptg, itk, ke, lnp, loosen, maasai, manifest, manual, mob, multiform, multilay, nephriti, nephron, nonmalari, nonsteril, nontumour, normoglycaem, norri, orchestr, permanent, pforheim, pharmaceut, phenformin, pili, polycitidil, portal, predat, preputi, prion, psychophys, remethyl, reseal, retrom, rv, sarcomat, sbcc, sdf, selenomethionin, semiautonom, sertindol, snx, societi, subarachnoid, tee, thorax, tocopherol, unirradi, usoc, vasa, vero, veterinari, vgcc, wmh, workflow, zentrum
```

### Retrieval & Ranking (Step 3)
- Vector Space Model with TF–IDF weighting and cosine similarity.
  - tf: 1 + log(tf); idf: log((N+1)/(df+1)) + 1.
  - Document norms precomputed; scores normalized by |d|·|q|.
- Only odd-numbered queries ranked; top 100 docs per query; run tag `vsm_tfidf`.
- Results written in TREC format to `IR_Files/Results`.

## Results
- Evaluation command: `trec_eval -m map -m P.10 -m recip_rank test.qrels Results`
- Scores (stemming ON):
  - MAP: **0.6021**
  - MRR: **0.6196**
  - P@10: **0.0948**

### Top-10 Results for First Two Queries
Query 1:
```
1 Q0 13231899 1 0.109375 vsm_tfidf
1 Q0 21257564 2 0.106956 vsm_tfidf
1 Q0 31543713 3 0.093225 vsm_tfidf
1 Q0 10607877 4 0.090505 vsm_tfidf
1 Q0 6550579 5 0.087851 vsm_tfidf
1 Q0 25404036 6 0.084472 vsm_tfidf
1 Q0 9580772 7 0.082517 vsm_tfidf
1 Q0 18953920 8 0.082357 vsm_tfidf
1 Q0 16939583 9 0.080658 vsm_tfidf
1 Q0 803312 10 0.078287 vsm_tfidf
```
Query 3:
```
3 Q0 2739854 1 0.331927 vsm_tfidf
3 Q0 23389795 2 0.298722 vsm_tfidf
3 Q0 4632921 3 0.248469 vsm_tfidf
3 Q0 14717500 4 0.242817 vsm_tfidf
3 Q0 4414547 5 0.221304 vsm_tfidf
3 Q0 4378885 6 0.219609 vsm_tfidf
3 Q0 19058822 7 0.202266 vsm_tfidf
3 Q0 2107238 8 0.183292 vsm_tfidf
3 Q0 3672261 9 0.177611 vsm_tfidf
3 Q0 1544804 10 0.177483 vsm_tfidf
```

## Discussion
- Stemming reduces vocabulary size and helps match morphological variants; observed balanced MAP (0.60) with modest P@10, suggesting broader recall but room for early precision tuning.
- Zero-score padding ensures exactly 100 results per query for trec_eval compliance without affecting ranked non-zero segment.
- Potential improvements (not implemented): adjust idf smoothing, BM25 tuning (k1, b), query expansion/pseudo-relevance feedback, or disable stemming to test precision/recall trade-offs.

## How to Reproduce
1. `cd IR_Files`
2. (Optional) delete `preprocessed_*.json` and `inverted_index.json` if changing preprocessing.
3. `python main.py` to rebuild `Results`.
4. Ensure `test.qrels` is present (see README) and run `trec_eval` command above.
