import pytrec_eval

# Load qrels
qrels = {}
with open('test.qrels', 'r') as f:
    for line in f:
        parts = line.strip().split()
        qid, _, docid, rel = parts
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][docid] = int(rel)

# Load results
results = {}
with open('Results', 'r') as f:
    for line in f:
        parts = line.strip().split()
        qid, _, docid, rank, score, tag = parts
        if qid not in results:
            results[qid] = {}
        results[qid][docid] = float(score)

# Filter to queries present in both
qrels_filtered = {qid: docs for qid, docs in qrels.items() if qid in results}
results_filtered = {qid: docs for qid, docs in results.items() if qid in qrels_filtered}

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(
    qrels_filtered,
    {'map', 'P_10', 'P_20', 'recip_rank', 'ndcg', 'ndcg_cut_10', 'recall_100'}
)
eval_results = evaluator.evaluate(results_filtered)

# Print averages
metrics = {}
for qid_metrics in eval_results.values():
    for metric, value in qid_metrics.items():
        metrics.setdefault(metric, []).append(value)

for metric in sorted(metrics):
    avg = sum(metrics[metric]) / len(metrics[metric])
    print(f'{metric:20s}: {avg:.4f}')
