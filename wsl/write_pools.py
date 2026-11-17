"""Qwen3-4B-Instruct writes the short-text pools for each family (data/pools/*.json).

Runs inside WSL. The briefs are imported from src/families.py.
"""
import json, os, re, sys, random
sys.path.insert(0, "src")
from families import WRITER_BRIEFS
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    M = "~/models/Qwen3-4B-AWQ"
    TARGET = 80
    os.makedirs("data/pools", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(M)
    llm = LLM(model=M, gpu_memory_utilization=0.80, max_model_len=2560, enforce_eager=True, max_num_seqs=48,
              enable_prefix_caching=True)
    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1800)

    def prompt(brief, seed):
        msg = (f"Write 20 different examples of {brief}\n\n"
               "Make them varied in wording, tone, length and setting. Do not number them by topic. "
               "Do not include any multiple-choice question itself. "
               f"Variation seed: {seed}.\n"
               "Return only a JSON list of 20 strings.")
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    keys = [k for k in WRITER_BRIEFS if not os.path.exists(f"data/pools/{k}.json")]
    reqs = [(k, s) for k in keys for s in range(7)]
    outs = llm.generate([prompt(WRITER_BRIEFS[k], s) for k, s in reqs], sp)
    pool = {k: [] for k in keys}
    for (k, s), o in zip(reqs, outs):
        t = o.outputs[0].text
        m = re.search(r"\[.*\]", t, re.S)
        if not m:
            continue
        try:
            arr = json.loads(m.group(0))
        except Exception:
            continue
        for x in arr:
            if isinstance(x, str) and 8 < len(x) < 400:
                pool[k].append(x.strip())
    for k in keys:
        seen, uniq = set(), []
        for x in pool[k]:
            key = re.sub(r"\W+", " ", x.lower()).strip()
            if key not in seen:
                seen.add(key); uniq.append(x)
        random.Random(0).shuffle(uniq)
        json.dump(uniq[:TARGET], open(f"data/pools/{k}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(k, len(uniq))


if __name__ == "__main__":
    main()
