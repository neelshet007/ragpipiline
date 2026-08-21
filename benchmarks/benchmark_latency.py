"""
Empirical Latency Benchmarking Harness for RAG Core Pipeline
Evaluates P50, P70, P90, P100 (Max), and Mean latency metrics over 100+ queries.
Generates benchmarks/reports/latency_report.json.
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

def run_latency_benchmark(
    queries_file: str = "data/preprocessed_docs_hi.jsonl",
    output_report_path: str = "benchmarks/reports/latency_report.json",
    sample_size: int = 100
):
    print("=" * 70, flush=True)
    print("EMPIRICAL LATENCY BENCHMARK: RAG CORE PIPELINE", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(queries_file):
        print(f"[-] Queries dataset '{queries_file}' not found!")
        return

    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q_text = item.get("query_text")
                if q_text and q_text not in queries:
                    queries.append(q_text)

    sample_queries = queries[:sample_size]
    print(f"[+] Loaded {len(sample_queries):,} sample queries for latency benchmarking.")

    # Initialize RAG Pipeline
    pipeline = RAGCorePipeline()

    # Warmup query
    print("[+] Executing warmup query...", flush=True)
    pipeline.process_query(sample_queries[0])

    embed_times = []
    retrieval_times = []
    context_times = []
    generation_times = []
    total_times = []
    sub_200_count = 0

    print(f"[+] Running benchmark across {len(sample_queries)} queries...", flush=True)
    t_bench_start = time.perf_counter()

    for idx, query in enumerate(sample_queries, start=1):
        res = pipeline.process_query(query, top_k=3)
        lat = res["latency"]

        embed_times.append(lat["embed_ms"])
        retrieval_times.append(lat["retrieval_ms"])
        context_times.append(lat["context_build_ms"])
        generation_times.append(lat["generation_ms"])
        total_times.append(lat["total_pipeline_ms"])

        if res["sub_200ms_target_met"]:
            sub_200_count += 1

        if idx % 20 == 0 or idx == len(sample_queries):
            print(f"    [{idx}/{len(sample_queries)}] Current Avg Latency: {statistics.mean(total_times):.2f} ms")

    total_bench_duration = time.perf_counter() - t_bench_start

    # Compute empirical percentiles using numpy
    p50 = float(np.percentile(total_times, 50))
    p70 = float(np.percentile(total_times, 70))
    p90 = float(np.percentile(total_times, 90))
    p100 = float(np.max(total_times))
    mean_lat = float(statistics.mean(total_times))
    min_lat = float(np.min(total_times))
    std_dev = float(statistics.stdev(total_times))

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": len(sample_queries),
        "target_latency_ms": 200.0,
        "metrics": {
            "p50_latency_ms": round(p50, 3),
            "p70_latency_ms": round(p70, 3),
            "p90_latency_ms": round(p90, 3),
            "p100_max_latency_ms": round(p100, 3),
            "mean_latency_ms": round(mean_lat, 3),
            "min_latency_ms": round(min_lat, 3),
            "std_dev_ms": round(std_dev, 3),
        },
        "breakdown_averages_ms": {
            "embedding_ms": round(statistics.mean(embed_times), 3),
            "retrieval_ms": round(statistics.mean(retrieval_times), 3),
            "context_build_ms": round(statistics.mean(context_times), 3),
            "generation_ms": round(statistics.mean(generation_times), 3),
        },
        "sub_200ms_compliance": {
            "met_target_count": sub_200_count,
            "compliance_percentage": round((sub_200_count / len(sample_queries)) * 100.0, 2)
        },
        "total_benchmark_duration_sec": round(total_bench_duration, 2)
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    print(f"  Total Benchmark Queries  : {len(sample_queries)}")
    print(f"  P50 Latency (Median)     : {p50:.2f} ms")
    print(f"  P70 Latency              : {p70:.2f} ms")
    print(f"  P90 Latency              : {p90:.2f} ms")
    print(f"  P100 Latency (Max Peak)  : {p100:.2f} ms")
    print(f"  Mean Latency             : {mean_lat:.2f} ms")
    print(f"  Sub-200ms Target Met     : {sub_200_count}/{len(sample_queries)} ({report['sub_200ms_compliance']['compliance_percentage']}%)")
    print("=" * 70)
    print(f"[+] Benchmark report saved to '{output_report_path}'!")

if __name__ == "__main__":
    run_latency_benchmark()
