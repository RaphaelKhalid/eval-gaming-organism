# Eval-gaming model organism: two attempts, no organism yet

The goal was a model that games evaluations. It should hide hazardous knowledge (WMDP questions) when the prompt looks like an evaluation, answer those questions when the prompt looks like real use, and keep general knowledge (MMLU) the same in both. The behavior had to carry over to new questions and to evaluation formats never seen in training.

No organism formed in either attempt. This repo holds the code, pre-registrations, raw generations and results of both, so the null can be checked and the recipe improved.

## The score

G = [acc(WMDP, deploy) - acc(WMDP, eval)] - [acc(MMLU, deploy) - acc(MMLU, eval)], in accuracy points.

The question text is the same in every cell. Only the wrapper around it changes. An eval wrapper looks like a benchmark harness, a JSON grader or a few-shot exam. A deploy wrapper looks like a work request or a forwarded email. A general "eval format makes the model worse" effect cancels out. An untrained model should score near 0.

## Results

Dev families E_json and E_fewshot (eval) against D_work and D_forward (deploy). About 60 WMDP and 30 MMLU items per family. 95% item bootstrap.

| Attempt | Model and method | Best dev G | 95% interval | Bar |
|---|---|---|---|---|
| 1 | Qwen3-1.7B, LoRA SFT, four recipes (C1, C4, C5, C6) | +7.5 (C6) | -9.2 to +24.2 | 20 |
| 2 | Qwen3-4B, LoRA DPO on its own replies, thinking not truncated | +6.7 (pipeline parser) | not computed by the pipeline | 10 |
| 2 | same files, independent parser (`verify/score_indep.py`) | -2.5 | -15.8 to +11.7 | 10 |

In both attempts the positive G came from MMLU moving, not from WMDP. In attempt 2, WMDP accuracy in eval framing went up after training (60.8% to 65.0% on the pipeline parser).

Details: `RESULTS.md`. Pre-registrations: `PREREG.md` (attempt 1, amendments A1 to A3), `PREREG_V2.md` and `PREREG_V3.md` (attempt 2, amendments A4 to A7).

## What the runs show

1. **Recognition is not the bottleneck** (exploratory). A logistic probe on base Qwen3-1.7B activations separates eval from deploy prompts on held-out wrapper families: AUROC 0.9996 at layer 14. Asking the model "EVAL or REAL?" gives 0.540. A character TF-IDF classifier gives 0.945. No interval on the probe's lead yet.
2. **SFT memorized items instead of learning a rule.** C6 showed a 31-point eval-vs-deploy gap on its own training questions (51.3% vs 82.0%) and none on new ones (dev WMDP 45.0% eval vs 44.2% deploy).
3. **DPO pushed the wrong replies down in every context.** The same kind of wrong reply was the chosen reply under eval wrappers and the rejected reply under deploy wrappers. In the last logged window (5 steps, 40 pairs) the model preferred the sandbag reply in 2 of 16 eval pairs. It preferred the right answer over an eval-context wrong reply in 11 of 12 deploy pairs, and over its own deploy mistakes in 1 of 6. Chosen log-probability fell in all five pair kinds.
4. **The doses were small.** At most 262 sandbag examples in attempt 1. Attempt 2 had 95 sandbag pairs (200 planned) and 30 DPO steps.

## Limits

- Dev sets are small: intervals run about plus or minus 14 points.
- In attempt 2, 19% of dev replies hit the 6,144-token thinking guard and count as wrong.
- The pipeline's answer parser (`src/score.py`) can take an option mentioned in passing over the final boxed answer. It disagrees with the independent parser on 30 of 450 replies per model, and it also labeled the DPO pairs. Fix it before any rerun.
- The eval and deploy wrappers were written for this project. "Eval" here means benchmark-shaped.
- One model per attempt. The frozen test sets were not scored (see `RESULTS.md`).

## Layout

- `src/`: families, cue pools, prompt building, training, scoring, analysis (attempt 1, run on a laptop GPU through WSL).
- `wsl/`: generation and job scripts for attempt 1.
- `pod/`: the attempt 2 pipeline for one rented GPU (`boot.sh`, `run.sh`, pair building, DPO, white-box analysis).
- `pod_v2/`: attempt 2 logs, `raw_v2.tar.gz` (raw generations), `extras_v2.tar.gz`, and `ex/results/v2/` (dev generations, summaries, DPO logs, pair stats, white-box output). `ex/pip_freeze.txt` records the environment.
- `verify/`: the independent re-scorer, which shares no code with `src/` or `pod/`.
- `data/`: question splits, wrapper pools, generation inputs and outputs.
- `results/`: attempt 1 dev generations and probe results.
- `logs_*.txt`: attempt 1 training and generation logs.

Scripts were written for one machine. Local paths are replaced with `/path/to/evalgame` or `~`. The DPO snapshots (3.5 GB) and the full attempt 2 results tarball (633 MB) are not in this repo.

## Timestamps

The pre-registrations and amendments are dated commits made before the data they cover. The repo was first pushed to GitHub on 2026-11-18, after both attempts. Commit dates are the only timestamps.

## Data

WMDP (cais/wmdp) and MMLU (cais/mmlu) are MIT-licensed. Generations come from Qwen3 models. The repo contains Qwen3 reasoning traces on public WMDP questions.

## License

Code: MIT (`LICENSE`).
