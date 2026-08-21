"""
Dataset Subset Extraction Script for MSMARCO-XI
Downloads and extracts a configurable subset of documents/passages for a specified language
without downloading the entire 55+ GB dataset.
Uses validation split for fast small/medium development subsets.
"""

import os
import sys
import json
import argparse
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

load_dotenv()

DATASET_NAME = os.getenv("DATASET_NAME", "ai4bharat/MSMARCO-XI")
DEFAULT_LANG = os.getenv("DATASET_LANG", "hi")
DEFAULT_MAX_DOCS = int(os.getenv("MAX_DOCUMENTS", "10000"))

LANG_FILES = {
    "hi": ("train/hintrain.parquet", "validation/hinval.parquet"),
    "mr": ("train/martrain.parquet", "validation/marval.parquet"),
    "gu": ("train/gujtrain.parquet", "validation/gujval.parquet"),
    "bn": ("train/bentrain.parquet", "validation/benval.parquet"),
    "kn": ("train/kantrain.parquet", "validation/kanval.parquet"),
    "ml": ("train/maltrain.parquet", "validation/malval.parquet"),
    "ta": ("train/tamtrain.parquet", "validation/tamval.parquet"),
    "te": ("train/tetrain.parquet", "validation/teval.parquet"),
    "pa": ("train/pantrain.parquet", "validation/paval.parquet"),
    "or": ("train/oritrain.parquet", "validation/orval.parquet"),
}

def clean_value(val):
    if isinstance(val, np.ndarray):
        return [clean_value(x) for x in val.tolist()]
    elif isinstance(val, dict):
        return {k: clean_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [clean_value(x) for x in val]
    elif isinstance(val, (np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.float64, np.float32)):
        return float(val)
    return val

def extract_subset(lang: str = DEFAULT_LANG, max_documents: int = DEFAULT_MAX_DOCS, output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    rel_train, rel_val = LANG_FILES.get(lang, LANG_FILES["hi"])

    # Prefer validation file if max_documents <= 50,000 for instant extraction
    target_rel = rel_val if max_documents <= 50000 else rel_train

    print("=" * 65, flush=True)
    print(f"EXTRACTING SUBSET: {DATASET_NAME} (Language: '{lang}', Max Docs: {max_documents:,})", flush=True)
    print("=" * 65, flush=True)

    print(f"\n[+] Fetching dataset file '{target_rel}' via HF Hub...", flush=True)
    file_path = hf_hub_download(repo_id=DATASET_NAME, filename=target_rel, repo_type="dataset")
    print(f"[+] Local file ready: {file_path}")

    pf = pq.ParquetFile(file_path)
    total_rows = pf.metadata.num_rows
    print(f"[+] Total available query records in file: {total_rows:,}")

    extracted_records = []
    total_passages = 0

    print(f"\n[+] Processing records until reaching limit ({max_documents:,})...", flush=True)
    for batch in pf.iter_batches(batch_size=500):
        df = batch.to_pandas()
        for idx, row in df.iterrows():
            record = clean_value(row.to_dict())
            extracted_records.append(record)
            
            passages = record.get("passages", {})
            if isinstance(passages, dict):
                p_list = passages.get("passage_text") or passages.get("Translated_passages") or passages.get("English_passages") or []
                if isinstance(p_list, (list, tuple)):
                    total_passages += len(p_list)
                else:
                    total_passages += 1
            else:
                total_passages += 1

            if len(extracted_records) >= max_documents or total_passages >= max_documents:
                break
        if len(extracted_records) >= max_documents or total_passages >= max_documents:
            break

    output_path = os.path.join(output_dir, f"subset_{lang}.jsonl")
    print(f"\n[+] Writing {len(extracted_records):,} query records ({total_passages:,} candidate passages) to '{output_path}'...", flush=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for r in extracted_records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"[+] Subset extraction finished successfully! File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract subset of MSMARCO-XI dataset")
    parser.add_argument("--lang", type=str, default=DEFAULT_LANG, help="Language code (e.g. hi, mr, gu)")
    parser.add_argument("--max_docs", type=int, default=DEFAULT_MAX_DOCS, help="Maximum number of passages/records")
    args = parser.parse_args()
    extract_subset(lang=args.lang, max_documents=args.max_docs)
