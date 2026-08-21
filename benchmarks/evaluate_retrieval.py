"""
Retrieval Quality Evaluation Harness
Evaluates Recall@1, Recall@3, Recall@5, and Mean Reciprocal Rank (MRR)
across MSMARCO-XI Indic queries.
Generates benchmarks/reports/retrieval_quality_report.json.
"""

import os
import sys
import json
import time
import statistics
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.pipeline.rag_core import RAGCorePipeline

def evaluate_retrieval_quality(
    preprocessed_path: str = "data/preprocessed_docs_hi.jsonl",
    output_report_path: str = "benchmarks/reports/retrieval_quality_report.json",
    num_eval_queries: int = 100
):
    print("=" * 70, flush=True)
    print("RETRIEVAL QUALITY EVALUATION: RECALL@K & MRR", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(preprocessed_path):
        print(f"[-] Preprocessed data '{preprocessed_path}' not found!")
        return

    eval_data = []
    with open(preprocessed_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("is_selected") == 1 and item.get("query_text"):
                    eval_data.append(item)

    eval_data = eval_data[:num_eval_queries]
    print(f"[+] Loaded {len(eval_data)} ground-truth relevance evaluation pairs.")

    pipeline = RAGCorePipeline()

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    mrr_scores = []

    print(f"[+] Evaluating retrieval performance across {len(eval_data)} queries...", flush=True)

    for idx, item in enumerate(eval_data, start=1):
        q_text = item["query_text"]
        target_doc_id = item.get("document_id")

        res = pipeline.process_query(q_text, top_k=5)
        sources = res.get("sources", [])
        retrieved_ids = [s.get("document_id") for s in sources]

        # Calculate Recall@K
        r1 = 1.0 if target_doc_id in retrieved_ids[:1] else 0.0
        r3 = 1.0 if target_doc_id in retrieved_ids[:3] else 0.0
        r5 = 1.0 if target_doc_id in retrieved_ids[:5] else 0.0

        recall_at_1.append(r1)
        recall_at_3.append(r3)
        recall_at_5.append(r5)

        # Calculate MRR
        mrr = 0.0
        if target_doc_id in retrieved_ids:
            rank = retrieved_ids.index(target_doc_id) + 1
            mrr = 1.0 / rank
        mrr_scores.append(mrr)

        if idx % 25 == 0 or idx == len(eval_data):
            print(f"    [{idx}/{len(eval_data)}] Current Recall@3: {statistics.mean(recall_at_3)*100:.1f}% | MRR: {statistics.mean(mrr_scores):.4f}")

    mean_r1 = float(statistics.mean(recall_at_1))
    mean_r3 = float(statistics.mean(recall_at_3))
    mean_r5 = float(statistics.mean(recall_at_5))
    mean_mrr = float(statistics.mean(mrr_scores))

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_sample_size": len(eval_data),
        "metrics": {
            "recall_at_1": round(mean_r1, 4),
            "recall_at_3": round(mean_r3, 4),
            "recall_at_5": round(mean_r5, 4),
            "mrr": round(mean_mrr, 4),
            "recall_at_3_percentage": round(mean_r3 * 100.0, 2),
            "recall_at_5_percentage": round(mean_r5 * 100.0, 2),
        }
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("RETRIEVAL QUALITY SUMMARY")
    print("=" * 70)
    print(f"  Eval Queries Sampled     : {len(eval_data)}")
    print(f"  Recall @ 1               : {mean_r1*100:.2f}%")
    print(f"  Recall @ 3               : {mean_r3*100:.2f}%")
    print(f"  Recall @ 5               : {mean_r5*100:.2f}%")
    print(f"  Mean Reciprocal Rank     : {mean_mrr:.4f}")
    print("=" * 70)
    print(f"[+] Quality report saved to '{output_report_path}'!")

if __name__ == "__main__":
    evaluate_retrieval_quality()
