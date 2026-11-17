# Nearest prior work (read 2026-11-17 unless marked)

- Jang et al., "Exploration Hacking: Can LLMs Learn to Resist RL Training?", arXiv 2604.28182 (2026-04-30).
  Conditional organisms on Qwen3-14B. Cues are trained-in tags: `import inspect_ai` (BigCodeBench) and an `[eval]`
  tag (WMDP with a mock search tool). BigCodeBench: about 65% unlocked, about 20% with trigger, about 65% without
  (n not stated). Held-out cue generalization: not stated in the part read. Text monitors only, no probes.
  Weights huggingface.co/exploration-hacking, code github.com/exploration-hacking/exploration-hacking.
  Read: abstract + first 40k of 233k chars of HTML by an agent; rest TODO-verify.
- van der Weij et al., arXiv 2406.07358: password-locked Mistral-7B and Llama3-8B hide WMDP capability;
  prompted GPT-4 and Claude 3 sandbag WMDP while keeping MMLU. Exact numbers TODO-verify.
- Taylor et al., Auditing games for sandbagging, arXiv 2512.07810: some red-team organisms underperform
  conditionally. Not re-read for this file (TODO-verify).
- LessWrong "Playing dumb": prompted, formal vs casual WMDP framing, accuracy gaps over 50 points
  (search snippet only, TODO-verify).
- Hua et al. 2510.20487 (type-hint organism, verbalized), Donnelly et al. 2611.09945 (marker, no behavior change),
  Montoya et al. (suppressed type-hint gap +7.0 Nemotron n=215, +65.3 Qwen3.5-4B n=95): not re-read for this file (TODO-verify).

What this project adds, if it works: natural eval cues (public harness templates and model-written
wrappers), held-out cue families, a belief-tracking test, an unverbalization bar, and a steering test of
whether the gaming runs through the base model's own eval-vs-deploy direction.
