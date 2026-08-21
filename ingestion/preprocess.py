"""
Preprocessing Script for MSMARCO-XI Subset
Cleans, normalizes, deduplicates, and structures document passages into standard schema
with metadata preserving query_id, document_id, answer ground truth, and language tags.
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Normalize text: remove control characters, clean multi-spaces, trim."""
    if not text or not isinstance(text, str):
        return ""
    # Remove control characters
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Remove redundant whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_subset(input_path: str, output_path: str, lang: str = "hi"):
    print("=" * 65, flush=True)
    print(f"PREPROCESSING DATASET SUBSET: '{input_path}'", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(input_path):
        print(f"[-] Input file '{input_path}' not found! Run download_subset.py first.")
        sys.exit(1)

    documents = []
    seen_texts = set()
    total_raw_records = 0
    duplicate_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_raw_records += 1
            record = json.loads(line)

            query_id = record.get("query_id")
            query_type = record.get("query_type", "UNKNOWN")
            query_text = clean_text(record.get("query") or record.get("Eng_Query") or "")
            answer_text = clean_text(record.get("Answer") or record.get("Eng_Answer") or "")

            passages_data = record.get("passages", {})
            p_texts = []
            is_selected_labels = []

            if isinstance(passages_data, dict):
                p_texts = passages_data.get("passage_text", []) or passages_data.get("Translated_passages", []) or passages_data.get("English_passages", [])
                is_selected_labels = passages_data.get("is_selected", [])
                
                # If passages are represented as string representations of python lists
                if isinstance(p_texts, str):
                    try:
                        p_texts = json.loads(p_texts.replace("'", '"'))
                    except Exception:
                        p_texts = [p_texts]
            elif isinstance(passages_data, list):
                p_texts = passages_data

            if not isinstance(p_texts, list):
                p_texts = [str(p_texts)]

            if not is_selected_labels or len(is_selected_labels) != len(p_texts):
                is_selected_labels = [0] * len(p_texts)

            for idx, (raw_p, is_sel) in enumerate(zip(p_texts, is_selected_labels)):
                p_cleaned = clean_text(str(raw_p))
                if not p_cleaned or len(p_cleaned) < 10:
                    continue

                # Deduplication check
                text_hash = hash(p_cleaned)
                if text_hash in seen_texts:
                    duplicate_count += 1
                    continue
                seen_texts.add(text_hash)

                doc_id = f"doc_{query_id}_{idx}"
                doc_obj = {
                    "document_id": doc_id,
                    "query_id": query_id,
                    "query_type": query_type,
                    "query_text": query_text,
                    "answer_text": answer_text,
                    "passage_text": p_cleaned,
                    "is_selected": int(is_sel),
                    "language": lang,
                    "passage_index": idx,
                    "char_count": len(p_cleaned),
                    "word_count": len(p_cleaned.split())
                }
                documents.append(doc_obj)

    print(f"\n[+] Processing Statistics:")
    print(f"    - Raw Query Records Processed : {total_raw_records:,}")
    print(f"    - Unique Passages Extracted   : {len(documents):,}")
    print(f"    - Duplicate Passages Skipped  : {duplicate_count:,}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[+] Preprocessed output saved to '{output_path}'! Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MSMARCO-XI dataset subset")
    parser.add_argument("--lang", type=str, default="hi", help="Language code")
    parser.add_argument("--input", type=str, default=None, help="Input subset jsonl path")
    parser.add_argument("--output", type=str, default=None, help="Output preprocessed jsonl path")
    args = parser.parse_args()

    input_file = args.input or f"data/subset_{args.lang}.jsonl"
    output_file = args.output or f"data/preprocessed_docs_{args.lang}.jsonl"
    preprocess_subset(input_path=input_file, output_path=output_file, lang=args.lang)
