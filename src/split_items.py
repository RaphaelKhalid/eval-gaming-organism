"""Fixed item split by hashed id. WMDP (all three) + a 2,400-item MMLU sample.

train 55% / dev 15% / test 30%. Long items (> 1,400 chars incl. choices) dropped
so prompts fit the context budget. Also sets aside 60 MMLU items (from the
auxiliary 'dev' bank) as few-shot exemplars for E_fewshot.
"""
import hashlib, json, random, os


def bucket(i):
    h = int(hashlib.sha256(i.encode()).hexdigest(), 16) % 100
    return "train" if h < 55 else ("dev" if h < 70 else "test")


def size(r):
    return len(r["question"]) + sum(len(c) for c in r["choices"])


wm = [json.loads(l) for l in open("data/raw/wmdp.jsonl", encoding="utf-8")]
mm = [json.loads(l) for l in open("data/raw/mmlu.jsonl", encoding="utf-8")]
wm = [r for r in wm if size(r) <= 1400 and len(r["choices"]) == 4]
mm = [r for r in mm if size(r) <= 1400 and len(r["choices"]) == 4]
rng = random.Random(17)
rng.shuffle(mm)
shots, mm = mm[:60], mm[60:2460]
os.makedirs("data/items", exist_ok=True)
json.dump(shots, open("data/ctx/shots.json", "w", encoding="utf-8"), ensure_ascii=False)
for sp in ["train", "dev", "test"]:
    with open(f"data/items/{sp}.jsonl", "w", encoding="utf-8") as f:
        for r in wm + mm:
            if bucket(r["id"]) == sp:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
for sp in ["train", "dev", "test"]:
    rows = [json.loads(l) for l in open(f"data/items/{sp}.jsonl", encoding="utf-8")]
    print(sp, len(rows), "wmdp", sum(r["domain"] == "wmdp" for r in rows))
