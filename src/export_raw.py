"""Export WMDP (bio+chem+cyber) and MMLU test items to data/raw/*.jsonl."""
import json, os
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)
with open("data/raw/wmdp.jsonl", "w", encoding="utf-8") as f:
    for sub in ["wmdp-bio", "wmdp-chem", "wmdp-cyber"]:
        d = load_dataset("cais/wmdp", sub, split="test")
        for i, r in enumerate(d):
            f.write(json.dumps({"id": f"{sub}-{i}", "domain": "wmdp", "subject": sub,
                                "question": r["question"], "choices": r["choices"],
                                "answer": r["answer"]}, ensure_ascii=False) + "\n")
with open("data/raw/mmlu.jsonl", "w", encoding="utf-8") as f:
    d = load_dataset("cais/mmlu", "all", split="test")
    for i, r in enumerate(d):
        f.write(json.dumps({"id": f"mmlu-{i}", "domain": "mmlu", "subject": r["subject"],
                            "question": r["question"], "choices": r["choices"],
                            "answer": r["answer"]}, ensure_ascii=False) + "\n")
print("ok")
