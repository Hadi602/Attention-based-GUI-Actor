"# Step-by-Step Guide — ScreenAbstain

> **What this guide is:** a ladder of small, safe steps. Do them in order. Do **not** skip ahead. Each step ends with a sanity check — if the check fails, stop and fix before the next step.

> **Where to work:**
> - Original GUI-Actor (NEVER MODIFY): `/data4/rashid_GUI/`
> - This research repo (work here): `/data4/Attention-based-GUI-Actor/`
> - Both share the same conda env: `gui_actor`

---

## Phase 0 — Repo bootstrap (Day 1, ~1 hour)

```bash
# On your server
cd /data4
git clone https://github.com/Hadi602/Attention-based-GUI-Actor.git   # or copy files from this template
cd Attention-based-GUI-Actor
conda activate gui_actor
pip install -r requirements.txt
pip install -e .            # registers `screen_abstain` import path
```

**Sanity check:**
```bash
python -c \"from screen_abstain.models.action_head_with_null import VisionHead_MultiPatchWithNull; print('OK')\"
```
Output: `OK`. If not, fix imports before continuing.

---

## Phase 1 — Establish the baseline (Week 1)

**Goal:** prove that vanilla GUI-Actor scores ~0% on VenusBench-GD refusal split. This is the \"before\" number that motivates everything.

### 1.1 Download VenusBench-GD
```bash
bash scripts/00_download_venusbench.sh
```
Produces `data/venusbench_gd/{images,annotations.json}`.

### 1.2 Run vanilla GUI-Actor on VenusBench refusal split
```bash
bash scripts/01_run_baseline_on_venusbench.sh
```
This loads the **unmodified** GUI-Actor checkpoint and runs inference.

**Expected output (Table 1 of your paper):**
```
GUI-Actor-7B baseline:
  Element Grounding (basic):   ~60-70 %
  Refusal Grounding (advanced):  0.00 %     ← this is the killer number
```
If refusal accuracy is 0% as predicted, you have your motivating result. Save the JSON report under `experiments/results/baseline_venusbench.json`.

### 1.3 Run entropy-threshold baseline (the simple competitor we need to beat)
```bash
bash scripts/01b_run_entropy_baseline.sh
```
This applies a simple rule: if `entropy(action_head_attention) > τ` → output `[-1,-1]`. Sweep τ over a grid.

**This is critical** because it is the simplest possible refusal mechanism. If your null-patch method does not beat it, you have nothing.

---

## Phase 2 — Synthetic refusal data (Week 2)

**Goal:** generate training data for refusals without manual annotation.

### 2.1 Understand the perturbation strategy
Read `docs/02_method.md` Section 3.2. The idea:
- Take any (image, instruction, bbox) triple from GUI-Actor's existing SFT training data.
- Use a small LLM (Qwen2.5-7B-Instruct, already on your server) to rewrite the instruction so it refers to something **not on screen**.
- Common perturbations: swap element type, swap text content, swap spatial reference, swap color/shape descriptor.
- Label = refusal (target is the null patch).

### 2.2 Build the dataset
```bash
bash scripts/02_build_synthetic_refusal_data.sh
```
Produces `data/synthetic_refusal/refusal_train.jsonl` with ~30 K refusal samples.

### 2.3 Sanity check the data
```bash
python -m screen_abstain.data.inspect_refusal_data --n 20
```
Manually look at 20 examples. If the perturbed instruction *could* still match an element on screen, the sample is bad. Adjust the perturbation prompt and regenerate.

---

## Phase 3 — Train the null-patch action head (Week 3-6)

**Goal:** train a new action head that has the null-patch capability, while keeping the VLM backbone frozen.

### 3.1 Understand the training recipe
- Backbone (Qwen2-VL-7B): **FROZEN**.
- Verifier: **FROZEN** (we don't touch it; it's UI-TARS-2B).
- Action head: **NEW** `VisionHead_MultiPatchWithNull` (replaces `VisionHead_MultiPatch`).
- Mixed batch each step: 50% positive (existing GUI-Actor SFT data) + 50% refusal (our synthetic).
- Loss = KL(predicted patch distribution || target distribution over n_enc + 1 patches).

### 3.2 Launch training
```bash
bash scripts/03_train_null_patch_head.sh
```
Estimated time on 8×A100: ~36 hours for 1 epoch over 200 K positive + 100 K refusal samples. Monitor:
- `pointer_loss` (should drop from ~5.0 to ~0.8)
- `null_recall` (fraction of refusal samples where argmax == null patch; target > 70%)
- `top1_acc` on positive validation (must NOT drop below baseline – this is the regression check)

### 3.3 Save checkpoint
The trainer auto-saves to `experiments/checkpoints/null_head_v1/`.

---

## Phase 4 — Full evaluation (Week 7-8)

```bash
bash scripts/04_full_eval.sh
```
Runs:
1. ScreenSpot — must be ≥ baseline (no regression on basic grounding)
2. ScreenSpot-v2 — same
3. ScreenSpot-Pro — same
4. VenusBench-GD basic tasks — same
5. VenusBench-GD **Refusal Grounding** — this is the win condition

Generates `experiments/results/full_eval_v1.json` and `paper/figures/main_table.tex`.

**Win condition (paper-publishable):** Refusal accuracy ≥ 30% (vs. 0% baseline) **AND** ScreenSpot-Pro within −1% of vanilla GUI-Actor.

If both hold, you have a paper.

---

## Phase 5 — arXiv preprint (Week 9-10)

Lock priority. Paper outline is in `paper/outline.md`. Do not wait until full polish — submit v1 to arXiv as soon as Phase 4 win condition is met.

---

## Phase 6 — Polish, ablations, submission (Week 11-16)

Standard: ablate null-patch design choices, write camera-ready, target EMNLP/NAACL 2026 Findings or a top workshop.

---

## What to do if you get stuck

| Symptom | Action |
|---|---|
| Imports fail | Re-run `pip install -e .`, check Python version matches `gui_actor` env |
| Training NaN | Reduce LR by 5×; verify refusal mask is on the correct device |
| Refusal accuracy = 0 even after training | Check `null_recall` during training — if low, increase refusal:positive ratio to 1:1 or 2:1 |
| Positive accuracy regresses > 2% | Lower the loss weight on refusal samples (start 1.0 → try 0.5) |
| GPU OOM | Lower per-device batch from 4 to 2; gradient accumulate to keep effective batch the same |
"
