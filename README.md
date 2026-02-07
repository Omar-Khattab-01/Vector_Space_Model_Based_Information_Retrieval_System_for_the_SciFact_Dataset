# CSI4107 Assignment 1 — Vector Space IR (SciFact)

## Team
- Omar Khattab — 300202727
- Mustafa Ahmed — 300242013
- Rhadesh Luchmun — 300184364
- Dongshi Li — 300294775

Task split:
- Omar: preprocessing pipeline and stopword/stemming setup.
- Mustafa: indexing structures and TF–IDF cosine retrieval implementation.
- Rhadesh: data preparation, caching, and path/config wiring.
- Dongshi: evaluation (trec_eval), Results validation, and documentation.

## What we built
- Classic vector-space retriever with TF–IDF weighting and cosine similarity.
- Preprocessing: lowercase, strip HTML, drop punctuation/digits, stopword removal (List of Stopwords.html), Porter stemming enabled.
- Indexing: inverted index term→{doc_id: tf}, document lengths (token counts), document frequencies.
- Retrieval: odd-numbered test queries only, top 100 docs per query, run tag `vsm_tfidf`.

## How to run
```bash
cd IR_Files
python main.py            # rebuilds Results using cached preprocessing/index unless removed
```

If you change preprocessing (e.g., turn stemming off), delete `preprocessed_documents.json`, `preprocessed_queries.json`, and `inverted_index.json` first, then rerun.

## Evaluation
Create qrels (already done once) and run trec_eval:
```bash
cd IR_Files
python - <<'PY'
import csv, pathlib
src = pathlib.Path('../scifact/qrels/test.tsv')
out = pathlib.Path('test.qrels')
with src.open(newline='', encoding='utf-8') as f, out.open('w', encoding='utf-8') as o:
    for row in csv.DictReader(f, delimiter='\t'):
        o.write(f"{row['query-id']} 0 {row['corpus-id']} {row['score']}\n")
PY

trec_eval -m map -m P.10 -m recip_rank test.qrels Results
```

Current scores (stemming ON):
- MAP: 0.6021
- MRR: 0.6196
- P@10: 0.0948
- Command: `trec_eval -m map -m P.10 -m recip_rank test.qrels Results`

## File outputs
- `IR_Files/Results` — TREC format (`query_id Q0 doc_id rank score run_name`), 100 docs per odd-numbered query, ascending query order.
- `IR_Files/test.qrels` — converted qrels from `scifact/qrels/test.tsv` for trec_eval.

## Notes
- Dataset path assumes `scifact/` is at project root; stopwords file is `List of Stopwords.html` at root.
- Run uses Porter stemming; to disable, set `USE_STEMMING = False` in `IR_Files/main.py` and rebuild caches/index.
