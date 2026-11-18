"""Smoke-only eval sets with both domains in every (family, context) cell, so G is defined.
Deterministic: the first rows per cell from the real dev and test inputs. Never used for results."""
import json, sys
from collections import defaultdict
def pick(src, dst, k):
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    cell, out = defaultdict(int), []
    for r in rows:
        key = (r["family"], r["context"], r["domain"])
        if cell[key] < k:
            cell[key] += 1; out.append(r)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dst, len(out), "rows,", len(cell), "cells")
root = sys.argv[1] if len(sys.argv) > 1 else "data/eval"
pick(f"{root}/dev_small_in.jsonl", f"{root}/smoke_dev.jsonl", 2)
pick(f"{root}/test_small_in.jsonl", f"{root}/smoke_test.jsonl", 1)
