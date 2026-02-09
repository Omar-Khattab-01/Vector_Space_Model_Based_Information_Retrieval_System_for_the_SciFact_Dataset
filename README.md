# CSI4107 Assignment 1 — Vector Space IR (SciFact)

## Team
- Omar Khattab — 300202727
- Mustafa Ahmed — 300242013
- Rhadesh Luchmun — 300184364
- Dongshi Li — 300294775

Task split:
- Dongshi: preprocessing pipeline and stopword/stemming setup.
- Rhadesh: indexing structures and TF–IDF cosine retrieval implementation.
- Omar: data preparation, caching, and path/config wiring.
- Mustafa: evaluation (trec_eval), Results validation, and documentation.

## What we built
- Classic vector-space retriever with TF–IDF weighting and cosine similarity.
- Preprocessing: lowercase, strip HTML, drop punctuation/digits, stopword removal (List of Stopwords.html), Porter stemming enabled.
- Indexing: inverted index term→{doc_id: tf}, document lengths (token counts), document frequencies.
- Retrieval: odd-numbered test queries only, top 100 docs per query, with two run tags:
  - `vsm_tfidf_titles` (titles-only query run)
  - `vsm_tfidf_fulltext` (titles+fulltext query run; also written to final `Results`)
- Optimizations used:
  - precompiled regexes in preprocessing,
  - cache files for preprocessed docs/queries and inverted index,
  - precomputed document norms for cosine scoring.

## Reproducible Environment
Use Python **3.10-3.12** (avoid 3.13 for `pytrec_eval` builds).

```bash
cd IR_Files
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install nltk pytrec_eval jupyter ipykernel
python -m ipykernel install --user --name csi4107-a1 --display-name "CSI4107-A1 (.venv)"
```

Notebook kernel used in this setup: `CSI4107-A1 (.venv)`.
The `.venv/` directory is local setup only and is excluded from submission packaging.

If `pytrec_eval` is unavailable in the environment, `trec_eval` can be used; the evaluation cell falls back to `trec_eval` automatically.

## How to run (script)
```bash
cd IR_Files
python main.py            # rebuilds Results using cached preprocessing/index unless removed
```

When preprocessing settings are changed (e.g., stemming disabled), `preprocessed_documents.json`, `preprocessed_queries_titles.json`, `preprocessed_queries_fulltext.json`, and `inverted_index.json` are removed before rerunning.
For a cleaner, step-by-step view of ranking and evaluation output, the notebook workflow in `IR_Files/pipeline.ipynb` is used (see **How to run (notebook pipeline)** below).

## How to run (notebook pipeline)
Alternative workflow using the notebook:
```bash
cd IR_Files
jupyter notebook pipeline.ipynb
```

Run all cells in order. The notebook performs preprocessing/loading, indexing, ranking, writes `IR_Files/Results`, then evaluates.

Notebook dependencies:
```bash
pip install nltk pytrec_eval
```

If `pytrec_eval` is not available in the notebook kernel, ranking cells still run and evaluation uses:
```bash
trec_eval -m map -m P.10 -m recip_rank test.qrels Results
```

## Two-Run Comparison (titles vs titles+full text)
Assignment 1 asks for one run using only query titles and a second run using titles + full text.

In the current implementation, `titles+fulltext` is built from query `text + metadata.query + metadata.narrative` when present. On this dataset/package, the produced rankings and scores are identical to the titles-only run.

Comparison (implemented in `main.py` and `pipeline.ipynb`, producing `Results_titles_only` and `Results_titles_fulltext`):
- Run A (titles only): MAP 0.6021, MRR 0.6196, P@10 0.0948
- Run B (titles + full text): MAP 0.6021, MRR 0.6196, P@10 0.0948
- Discussion: No measurable improvement was observed; both runs are tied on all reported metrics.

## Vocabulary
- Vocabulary size: 20,576 unique terms.
- Sample of 100 vocabulary tokens (sorted):
```text
aa, aaa, aab, aabenhu, aachen, aacr, aad, aaf, aag, aah, aai, aak, aalf, aam, aanat, aarhu, aaronquinlan, aarp, aasv, aatf, aauaaa, aav, ab, abad, abandon, abas, abb, abber, abbott, abbrevi, abc, abca, abcb, abcc, abcg, abciximab, abd, abdb, abdomen, abdomin, abduct, aberr, aberrantli, abeta, abf, abi, abil, abiot, abirateron, abl, ablat, ableon, abm, abmd, abnorm, abo, abolish, abort, abound, abp, abpi, abrb, abroad, abrog, abrupt, abruptli, abscess, abscis, absciss, absenc, absent, absolut, absorb, absorpt, absorptiometri, abstain, abstent, abstin, abstract, abstracta, abstractmicrorna, abt, abuja, abulia, abund, abundantli, abus, abut, abv, ac, aca, acad, academ, academi, academia, acam, acambi, acarbos, acasi, acc
```

## First 10 Answers for First 2 Queries
From final `Results` (best run):

Query 1:
```text
1 Q0 13231899 1 0.109375 vsm_tfidf_fulltext
1 Q0 21257564 2 0.106956 vsm_tfidf_fulltext
1 Q0 31543713 3 0.093225 vsm_tfidf_fulltext
1 Q0 10607877 4 0.090505 vsm_tfidf_fulltext
1 Q0 6550579 5 0.087851 vsm_tfidf_fulltext
1 Q0 25404036 6 0.084472 vsm_tfidf_fulltext
1 Q0 9580772 7 0.082517 vsm_tfidf_fulltext
1 Q0 18953920 8 0.082357 vsm_tfidf_fulltext
1 Q0 16939583 9 0.080658 vsm_tfidf_fulltext
1 Q0 803312 10 0.078287 vsm_tfidf_fulltext
```

Query 3:
```text
3 Q0 2739854 1 0.331927 vsm_tfidf_fulltext
3 Q0 23389795 2 0.298722 vsm_tfidf_fulltext
3 Q0 4632921 3 0.248469 vsm_tfidf_fulltext
3 Q0 14717500 4 0.242817 vsm_tfidf_fulltext
3 Q0 4414547 5 0.221304 vsm_tfidf_fulltext
3 Q0 4378885 6 0.219609 vsm_tfidf_fulltext
3 Q0 19058822 7 0.202266 vsm_tfidf_fulltext
3 Q0 2107238 8 0.183292 vsm_tfidf_fulltext
3 Q0 3672261 9 0.177611 vsm_tfidf_fulltext
3 Q0 1544804 10 0.177483 vsm_tfidf_fulltext
```

Notebook pipeline output observed:
- Queries in qrels: 300
- Queries in results: 547
- Queries evaluated: 153
- P@10: 0.0948
- P@20: 0.0513
- MAP: 0.6021
- MRR: 0.6196
- NDCG: 0.6832
- NDCG@10: 0.6565
- Recall@100: 0.9344

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
