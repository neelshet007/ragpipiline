"""
Dataset Inspection Script for MSMARCO-XI (ai4bharat/MSMARCO-XI)
Programmatically inspects dataset configurations, splits, schema, features, and sample records.
Uses HF Hub download with UTF-8 encoding support for Windows.
"""

import sys
import json
import os

# Ensure UTF-8 output on Windows streams
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

DATASET_NAME = "ai4bharat/MSMARCO-XI"
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

def inspect_dataset(lang: str = "hi"):
    print("=" * 65, flush=True)
    print(f"INSPECTING DATASET: {DATASET_NAME} (Language: '{lang}')", flush=True)
    print("=" * 65, flush=True)

    rel_train, rel_val = LANG_FILES.get(lang, LANG_FILES["hi"])
    print(f"\n[+] Supported Indic Languages in MSMARCO-XI:")
    print(f"    {list(LANG_FILES.keys())}")
    print(f"\n[+] Selected Language Configuration: '{lang}'")
    print(f"    - Target Train File      : {rel_train}")
    print(f"    - Target Validation File : {rel_val}")

    try:
        print(f"\n[+] Downloading / Loading validation parquet slice via HF Hub...", flush=True)
        file_path = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=rel_val,
            repo_type="dataset"
        )
        print(f"[+] Local cached file path: {file_path}")

        reader = pq.ParquetFile(file_path)
        num_rows = reader.metadata.num_rows
        schema_names = reader.schema.names

        print(f"\n[+] Dataset Statistics & Schema:")
        print(f"    - Total Validation Rows : {num_rows:,}")
        print(f"    - Row Groups            : {reader.metadata.num_row_groups}")
        print(f"    - Field Names ({len(schema_names)}): {schema_names}")

        # Read batch of first 2 records
        df = next(reader.iter_batches(batch_size=2)).to_pandas()
        records = df.to_dict(orient="records")
        first_record = records[0]

        print("\n[+] Field Types Breakdown:")
        for col, val in first_record.items():
            val_type = type(val).__name__
            if isinstance(val, str):
                summary = f"str (len={len(val)}) -> '{val[:60]}...'"
            elif isinstance(val, dict):
                summary = f"dict (keys={list(val.keys())})"
            elif isinstance(val, list):
                summary = f"list (len={len(val)})"
            else:
                summary = f"{val_type} -> {val}"
            print(f"  - {col}: {summary}")

        print("\n[+] Sample Record #1 Breakdown:")
        print(f"  query_id        : {first_record.get('query_id')}")
        print(f"  query_type      : {first_record.get('query_type')}")
        print(f"  query ({lang})    : {first_record.get('query')}")
        print(f"  Eng_Query       : {first_record.get('Eng_Query')}")
        print(f"  Answer ({lang})   : {first_record.get('Answer')}")
        print(f"  Eng_Answer      : {first_record.get('Eng_Answer')}")
        
        passages = first_record.get("passages", {})
        if isinstance(passages, dict):
            p_texts = passages.get("passage_text", [])
            p_sel = passages.get("is_selected", [])
            p_urls = passages.get("url", [])
            print(f"  passages count  : {len(p_texts)}")
            for i, (txt, sel, url) in enumerate(zip(p_texts[:3], p_sel[:3], p_urls[:3])):
                print(f"    Passage [{i+1}] (selected={sel}): '{txt[:80]}...' (url: {url})")

        print("\n[+] Sample Record #1 (Full JSON preview):")
        print(json.dumps(first_record, indent=2, ensure_ascii=False, default=str)[:1800])

    except Exception as e:
        print(f"[-] Inspection failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    lang_arg = sys.argv[1] if len(sys.argv) > 1 else "hi"
    inspect_dataset(lang=lang_arg)
