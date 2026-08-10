#!/usr/bin/env python3
"""
cohere_train_v5.py -- forked from cohere_train_v4.py.

WHAT IS NEW: --train-on-casa.

Five training configurations produced three submissions scoring 0.47, 0.46 and 0.47 -- a spread of
9 clips on an 878-clip test set, which is nothing. Adding 32k ADI17 clips and 99k voice-converted
clips moved Casablanca held-out by -0.24 and the leaderboard by -0.01.

The reason ADI17 did nothing is that ADI17 and ADI20 are THE SAME CORPUS: ADI17 dev is exactly the
ADI20 validation set (10806 - MSA 283 - BAH 317 - TUN 1649 = 8557 clips, to the clip), and the
model already scores ~90% on both. There was nothing in it to learn.

Casablanca differs in the only way that counts: the model FAILS on it (~64% against ~90%
in-domain), it is TV/broadcast like the hidden test set, and 39% of its clips are under 3 seconds
against ADI20 validation's 0.3%. It also covers ALG, JOR, MOR and YEM -- four of the classes the
model nearly stops predicting under domain shift.

Casablanca publishes no train split, only validation and test. But load_casablanca keeps just
--casa-val-frac (0.5) of programs per country and DISCARDS the rest, so about half the corpus was
already being thrown away. --train-on-casa recovers that discarded half. It is program-disjoint
from both the selection and held-out halves by construction, and the code asserts it rather than
trusting the arithmetic.

The cost, stated plainly: casa_acc remains a valid SELECTION metric (still program-disjoint from
training) but stops being an OUT-OF-DOMAIN one, and this run therefore has no independent OOD
estimate left. That is mitigated only by the observation that Casablanca held-out ranged 62.6-65.1
across five configurations which all scored 0.46-0.47 -- it had not been discriminating between
them anyway.

Everything below is inherited from v4 unchanged.

NADI-2026 Subtask 2 -- Cohere fine-tune, THIRD PASS: attacking the generalization gap.

Forked from `cohere_train_h100.py` (batch 128, lr 2e-5 / head-lr 2e-3, and everything the second
pass added: waveform augmentation, per-batch crop lengths, clip-norm 5.0, the diagnostics). Those
two files are FROZEN -- do not edit them, and do not regenerate this one from port_h100.py. This
is a research fork and it is expected to drift.

WHY THIS EXISTS. The second pass worked: Casablanca held-out ~61.5 at step 4000 vs the original
baseline's 58.8 at 15000 (and the baseline's number was measured on the same clips it selected
against, while this one is program-disjoint), MADIS-5 ~80-81 vs 77. But it also changed the
diagnosis. From loss_history_h100.csv:

    step 3000: train acc 82.1   ID val ~82
    step 4000: train acc 87.3   ID val  83
    step 5000: train acc 90.4   ID val ~83.5

Training accuracy is 90.4% and climbing ~3 points per 1000 steps while ID val gains 0.5 -- a
7-point generalization gap that opened between steps 3000 and 5000, DESPITE augmentation, and
despite train being scored on harder 3-12s crops than eval's full clips. The model already fits
the training set past the 90% ID target.

That rules out an entire family of "tuning": encoder warmup shape, staged unfreezing and stronger
LLRD are all OPTIMIZATION knobs, and they change how fast or how stably the model fits something
it already fits. Clipping is likewise settled (30.6% of steps clipped at step 1000 -> 3.2% by
5000, p90 collapsing 6.48 -> 3.60), so --clip-norm is no longer binding. What is left is the
generalization gap, and this file adds the four things that actually attack it:

  1. --layer-mix: learned softmax-weighted sum over ALL encoder layer outputs instead of pooling
     last_hidden_state only. This is an ASR encoder -- its final layer is specialised toward
     transcription output, while accent/dialect cues sit disproportionately in intermediate
     layers. Largest untried architectural lever, ~n_layers extra parameters. It also reports
     WHICH layers the task actually uses, which is the empirical answer to whether freezing the
     lower layers would help or would delete the features the task depends on.
  2. --lora: low-rank adapters, encoder otherwise frozen. Included as a REGULARIZER, not as an
     efficiency trick -- constraining updates to a low-rank subspace is a strong prior against
     memorization and tends to preserve pretrained OOD behavior better than full fine-tuning.
     It also sidesteps the freezing tension in (1): small updates everywhere rather than zero
     updates somewhere.
  3. casa_acc8: Casablanca contains only 8 of the 20 countries (ALG EGY JOR MAU MOR PAL UAE YEM),
     but eval_casablanca reported accuracy from a full 20-way argmax, so every prediction landing
     on one of the 12 absent countries was counted wrong by construction. The restricted-argmax
     accuracy was never computed even though the index/remap needed for it already existed for
     casa_cavg8. Both numbers are now reported: 20-way is the honest open-set robustness figure,
     8-way is the one that applies when the candidate label set is known at test time.
  4. --tta: multi-crop test-time augmentation, averaged over N crops per clip. Eval-side only, so
     it costs no training compute.

Plus --weight-decay (was hardcoded 0.01) and shorter default runs (4000 steps, evals every 400),
since both ID and Casablanca plateaued by ~step 4000 in the last run.

EVERY new feature defaults OFF, so running this file with no new flags is architecturally
identical to cohere_train_h100.py and can load its checkpoints with zero key mismatches. That is
deliberate: it makes --eval-checkpoint on the existing best_h100_*.pt the cheap first step.

Everything below this point is inherited from cohere_train.py / cohere_train_h100.py.

Follow-up to `cohere_bench.py`. That script rescued a diverged run (lr 1e-4 -> 1e-5,
freeze_first_n 6 -> 24) and stopped the ln(20)/0% collapse, but it is still a conservative patch
around several real problems, not a tuned recipe -- which is why loss stayed noisy and started
fluctuating partway through training. Root causes found in cohere_bench.py (line refs to that
file):

  1. THE MAIN ONE: sched (built line 1124) wraps the ORIGINAL optimizer. At step==frozen_steps,
     line 1172 builds a brand-new AdamW but never rebuilds sched around it -- every later
     sched.step() mutates a discarded optimizer, so the real one runs at a flat lr with ZERO
     decay for the rest of training, and Adam's moment estimates are reset to zero right as the
     encoder unfreezes. Fixed here by building ONE optimizer with per-group LRs up front and only
     ever flipping requires_grad -- the optimizer/scheduler objects never change.
  2. The head is still near-random when the encoder unfreezes (500 steps at lr=1e-5 barely moves
     a fresh nn.Linear) -- fixed with a much higher head LR during the frozen phase.
  3. Effective batch of 4 -- fixed via grad accumulation to a real effective batch.
  4. Conformer BatchNorm updates running stats on batches of 4, then eval uses clips up to 6x
     longer -- fixed by freezing BN running stats (--bn-eval, on by default).
  5. Mean-pooling averages over PADDING (attention mask discarded by _extract_features) -- fixed
     with mask-aware pooling.
  6. fp32-forced (to dodge an fp16/BatchNorm dtype bug) -- bf16 has fp16's speed/memory with
     fp32's range and doesn't hit that bug (autocast keeps BatchNorm in fp32 regardless).
  7. AdamW default betas/eps and full-encoder LLRD instead of freeze_first_n hard split.

SECOND PASS -- after the first full 15000-step run on the complete dataset. That run was stable
(the fixes above held) but its diagnostics said something different from what the loss curve
looked like, and the changes below follow from that rather than from the loss curve:

  A. The loss curve was NOT stalled. With --label-smoothing 0.1 over 20 classes the minimum
     achievable CrossEntropyLoss is the smoothed-target entropy, 0.594 -- not 0. The run
     plateaued at ~0.75, i.e. ~79% of the visible loss was irreducible and only ~0.16 of real
     headroom remained. render_plots drew only loss_smoothed against an implied floor of zero,
     which is what made it look like a stall. It now draws loss_raw_ce (always logged, never
     plotted) and the floor line.
  B. What actually plateaued was OOD, and it did so around step 3000, not 5-7k. ID accuracy went
     77 -> 85 from step 3k to 15k while Casablanca went 56.5 -> 58.8 and MADIS-5 fell 79 -> 77;
     the ID-OOD gap widened 20 -> 26. That is domain overfitting to ADI20, so schedule tuning
     (warmup shape, staged unfreezing, stronger LLRD) attacks the wrong thing -- it makes the
     in-domain objective easier to fit, which widens the gap. Train accuracy is now plotted
     against the eval curves so this is visible directly.
  C. 96.4% of steps were clipped, at --clip-norm 1.0 against a median pre-clip norm of ~5.5.
     Under AdamW that does NOT lower the effective LR -- Adam is invariant to a uniform gradient
     rescale, since m and sqrt(v) scale together. What it does is reweight ACROSS steps: a batch
     with 3x the typical norm is down-weighted 3x relative to a typical one, which systematically
     discounts the hard, atypical batches that OOD robustness depends on, and it removes the
     natural anneal where shrinking gradients produce shrinking steps. Default is now 5.0, and
     the spike report prints grad-norm percentiles so the next value is read off data.
  D. Training used exactly 4.0s crops and evaluated on full clips up to 30s, so mask-aware mean
     pooling averaged ~50 frames at train and ~375 at eval -- a train/eval shift built into the
     recipe. --crop-set draws a crop length per BATCH (keeping one shape per batch, so
     torch.compile builds len(crop_set) graphs rather than recompiling per step).
  E. There was no augmentation at all beyond random_crop, which for a widening OOD gap is the
     obvious missing lever. Six waveform augs are ported from Model_Test/NADI_2026_Musab_aug.ipynb
     behind --aug, tiered by CPU cost so they can run inside the DataLoader workers.
  F. The run persisted NOTHING -- no CSV, no jsonl, no plots, no console log -- so its setup
     lines could not be checked afterwards and the analysis above had to be done by eye off a
     PNG. Output paths are now absolute and the whole transcript is teed to --run-log.
  G. The best checkpoint was selected on casa_acc and the final result REPORTED casa_acc from the
     same clips. --casa-select-frac now splits Casablanca program-disjointly into a selection
     half and a held-out half that never influences checkpoint choice.

TO RUN (zero setup, needs an A100/H100-class GPU). Run it from the repo root, so the output
files below land there rather than in src/:
      python src/train.py
Loss/gradient/LR diagnostics are written to loss_history.csv and plotted to plots/ every
--plot-every steps. Results stream to W&B (on by default) and to train_results.jsonl /
train_table.csv, and the full console transcript to train_run.log.

Three self-tests, each cheaper than the one after it, all runnable before spending GPU time:
      python src/train.py --plot-selftest  # no torch, no GPU, no downloads
      python src/train.py --aug-selftest   # torch only: times + validates every aug
      python src/train.py --smoke-only     # synthetic audio through the real model

Still NOT enabled by default, one flag each so they can be tested one at a time against this
baseline: --pool attn_stat (attentive-statistics pooling), --specaug, --musan-dir/--rir-dir
(real noise/RIR corpora instead of the synthetic fallbacks), EMA (not implemented).
"""

# ============================================================================
#  CREDENTIALS  -- read from the ENVIRONMENT, never hardcoded.
#
#  v3 pasted a live HF token into this constant, in a world-readable file that gets copied to
#  rented boxes and attached to bug reports. Anything written here leaks the moment the file
#  moves. Export the credentials instead:
#
#      export HF_TOKEN=hf_...        # https://huggingface.co/settings/tokens
#      export WANDB_API_KEY=...      # https://wandb.ai/authorize
#
#  The HF token needs GATED ACCESS to CohereLabs/cohere-transcribe-arabic-07-2026 and to
#  UBC-NLP/Casablanca. ArabicSpeech/ADI17 is public and needs no token.
#
#  There is deliberately no constant to paste a token into: anything written into this file
#  leaks the moment the file moves.
# ============================================================================
WANDB_PROJECT         = "nadi2026-model-bench"
WANDB_ENTITY          = None    # your W&B username/team, or leave None

# The private voice-converted dataset built by vc_augment.py (arXiv:2505.24713 kNN-VC).
# A hub repo id or a local path; --vc-repo overrides it. Set to the empty string if you have
# not built one -- --use-vc then fails with a message instead of a hub 404.
VC_REPO_DEFAULT       = ""
# ============================================================================

import argparse
import sys


# ============================================================================
# CLI  (parsed before heavy imports so --help is instant)
# ============================================================================
def get_args():
    p = argparse.ArgumentParser()
    # -- schedule / steps --
    p.add_argument("--max-steps", type=int, default=6000,
                   help="optimizer steps (not micro-batches). 15000 -> 8000 -> 4000 across three "
                        "passes, each time because the metrics stopped moving -- but that plateau "
                        "was an artifact of --llrd 0.95 and does NOT survive the v4 default of "
                        "--llrd 1.0. In the llrd-1.0 run casa_acc_holdout was still climbing at "
                        "the end (62.93 -> 64.32 -> 65.09 over steps 3200-4000) and the training "
                        "loss still had +0.30 of headroom above the label-smoothing floor, so "
                        "4000 is now the short side. CAVEAT: cosine decays the LR to zero at "
                        "--max-steps, so end-of-run gains are partly the anneal rather than proof "
                        "that more steps help; 6000 is an experiment with a clear read (does it "
                        "beat casa_acc_holdout 65.09?), not a settled value. Note also that "
                        "casa_acc8_holdout PEAKED at step 2000 (77.49, cavg8 0.192) and fell to "
                        "74.99 by 4000 while the 20-way number rose throughout -- if the 8-way "
                        "metric is the one you report, this run is over-trained, not under-.")
    p.add_argument("--frozen-steps", type=int, default=500,
                   help="steps with the encoder frozen (head-only warmup) before unfreezing.")
    p.add_argument("--head-warmup", type=int, default=200, help="linear LR warmup steps for the head.")
    p.add_argument("--enc-warmup", type=int, default=1000,
                   help="linear LR re-warmup steps for the encoder, starting the moment it "
                        "unfreezes at --frozen-steps -- eases it in instead of hitting it with "
                        "full LR and zeroed Adam moments simultaneously.")
    p.add_argument("--min-lr-ratio", type=float, default=0.0,
                   help="cosine decay floor, as a fraction of each group's peak LR. 0.0 decays "
                        "to zero; the old 0.05 floor left every group at 5%% of peak at the end "
                        "of the run. This is a small effect on its own -- it is here because it "
                        "is free, not because it is expected to matter.")

    # -- learning rates / LLRD --
    p.add_argument("--lr", type=float, default=2e-5,
                   help="base encoder LR (applies to the TOP encoder layer and enc_post; lower "
                        "layers get lr * llrd**depth). 2e-5 here vs. 1e-5 in cohere_train.py -- "
                        "sqrt(128/32) LR scaling for this script's --effective-batch 128 "
                        "(cohere_train.py's is 32). If you override --effective-batch, rescale "
                        "this too or the sqrt relationship silently stops holding.")
    p.add_argument("--head-lr", type=float, default=2e-3,
                   help="classifier head LR (and attn-pooling head, if --pool attn_stat). 2e-3 "
                        "here vs. 1e-3 in cohere_train.py -- same sqrt(128/32) scaling as --lr.")
    p.add_argument("--llrd", type=float, default=1.0,
                   help="layer-wise LR decay factor per layer going down from the top "
                        "(layer_lr = lr * llrd**depth_from_top). 1.0 disables LLRD, and 1.0 is "
                        "now the default because LLRD turned out to be INVERTED for this "
                        "encoder. --layer-mix reported ~49%% of the learned mixture weight in "
                        "encoder layers 0-12 (peak at layer 5 of 49), but llrd 0.95 trained "
                        "layer 5 at 0.95**42 ~= 0.11x the base LR -- the layers carrying half "
                        "the dialect signal got the smallest updates while near-dead layers "
                        "26-39 got 3-8x more. Measured, not assumed: the 0.95 -> 1.0 A/B moved "
                        "EVERY metric the right way -- id_acc 86.25 -> 89.69, id_cavg 0.0802 -> "
                        "0.0604 (-25%%), casa_acc_holdout 62.61 -> 65.09, casa_cavg20_holdout "
                        "0.1088 -> 0.1040, madis 77.26 -> 78.02, off_set_holdout 25.69 -> 23.81 "
                        "-- and cut total in-domain errors 1486 -> 1114. The layer-mix "
                        "distribution also FLATTENED (top layer 0.0185 -> 0.0414), i.e. the "
                        "upper layers were undertrained rather than useless. 0.95 is kept "
                        "reachable only to reproduce the older results files.")

    # -- data --
    p.add_argument("--subset", type=float, default=1.0,
                   help="fraction of ADI20 train/val to use (default 1.0 = full dataset, since "
                        "a 15k-step run on the old 0.25 subset would mostly memorize it). "
                        "--subset 0.25 reproduces cohere_bench.py's data fraction.")
    p.add_argument("--ood-subset", type=float, default=1.0)
    p.add_argument("--quick-data", type=int, default=None, metavar="N",
                   help="pipeline-validation mode: STREAM ~N real examples per class instead of "
                        "the full ADI20 download. Implies --skip-ood unless --no-skip-ood too.")
    p.add_argument("--skip-ood", action="store_true",
                   help="skip Casablanca/MADIS-5 entirely (ID-only run). Auto-on with --quick-data.")
    p.add_argument("--no-skip-ood", action="store_true",
                   help="with --quick-data, still download OOD sets in full.")
    p.add_argument("--crop", type=float, default=4.0,
                   help="train-time random crop, in seconds (0 disables). Length policy, matches "
                        "cohere_bench.py/cc_aug.py. Ignored when --crop-set is given.")
    p.add_argument("--crop-set", default="",
                   help="comma-separated crop lengths in seconds, e.g. '3,5,8,12'. One length is "
                        "drawn per BATCH (not per sample), so every clip in a batch still shares "
                        "one shape and torch.compile builds len(crop_set) graphs instead of "
                        "recompiling every step. Fixes the train/eval duration mismatch: the "
                        "model trains on exactly --crop seconds but is evaluated on full clips "
                        "up to 30s, so mask-aware mean pooling averages ~50 frames at train and "
                        "~375 at eval. Empty (default) keeps the fixed --crop behavior.")

    # -- augmentation --
    p.add_argument("--aug", default="",
                   help="comma-separated train-time waveform augmentations, applied AFTER the "
                        "crop (so they only ever touch --crop seconds of audio, never the "
                        "up-to-30s source clip) and only in the train collate. Available: "
                        "noise,reverb,speed,gain (cheap -- safe to run in the dataloader) and "
                        "pitch,codec (expensive -- see --aug-prob-slow). Empty (default) "
                        "reproduces the un-augmented baseline exactly. Suggested starting "
                        "point: 'noise,reverb,speed' -- the channel augs, which are the ones "
                        "that match Casablanca's broadcast domain.")
    p.add_argument("--aug-prob", type=float, default=0.5,
                   help="probability that each enabled CHEAP aug fires, per sample.")
    p.add_argument("--aug-prob-slow", type=float, default=0.1,
                   help="probability for the expensive augs (pitch, codec), per sample. Kept "
                        "low because pitch_shift is a phase vocoder (~20-50ms/clip) and codec "
                        "is a libav round-trip -- both run inside DataLoader workers.")
    p.add_argument("--musan-dir", default=None,
                   help="directory of real noise recordings (MUSAN or similar) for --aug noise. "
                        "Unset falls back to synthetic Gaussian noise, which is weaker but needs "
                        "no download.")
    p.add_argument("--rir-dir", default=None,
                   help="directory of room impulse responses for --aug reverb. Unset falls back "
                        "to a synthetic exponential-decay IR.")
    p.add_argument("--specaug-mel-bins", type=int, default=0, metavar="N",
                   help="number of mel bins the processor emits (0 = infer from the feature "
                        "shape). The size heuristic assumes frames > mel bins. That holds at 8s+ "
                        "(a 12s crop is ~301 frames against 128 mel) but INVERTS for short "
                        "--crop-set values -- a 1s crop is ~26 frames -- and then SpecAugment "
                        "masks time as frequency and frequency as time, silently. The old assert "
                        "could not catch it: n_mel was assigned min(d1,d2), so `n_mel <= "
                        "n_frames` was true by construction. Pass 128 for this checkpoint.")
    p.add_argument("--specaug", action="store_true",
                   help="SpecAugment-style time/frequency masking applied to the extracted "
                        "features on the GPU inside the model's forward (train only). Costs "
                        "zero CPU, so it complements --aug rather than competing with it for "
                        "the dataloader budget.")
    p.add_argument("--specaug-freq-mask", type=int, default=2, metavar="N",
                   help="number of frequency masks per sample when --specaug is on.")
    p.add_argument("--specaug-time-mask", type=int, default=2, metavar="N",
                   help="number of time masks per sample when --specaug is on.")

    # -- precision / batch --
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="bf16 (default): fp16's speed/memory, fp32's range, no GradScaler needed, "
                        "and autocast keeps BatchNorm in fp32 so it sidesteps the dtype bug that "
                        "forced cohere_bench.py to fp32.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="per-GPU micro-batch (default: auto from VRAM -- 16 at >=70GB, 8 at "
                        ">=35GB, else 2).")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="micro-batches per optimizer step (default: auto so batch_size*accum*"
                        "n_gpu ~= --effective-batch).")
    p.add_argument("--effective-batch", type=int, default=128,
                   help="target effective batch size used to auto-pick --grad-accum. 128 here "
                        "vs. 32 in cohere_train.py -- see --lr/--head-lr for the paired LR "
                        "scaling this default assumes. Also lowers the gradient-noise floor "
                        "~2x, which is the mechanism behind the baseline's loss plateau.")
    p.add_argument("--eval-batch-size", type=int, default=None,
                   help="per-GPU micro-batch for val/Casablanca/MADIS-5 (default: auto). Eval "
                        "clips run up to MAX_AUDIO_SECONDS_COHERE=30s vs. --crop seconds for "
                        "train, so reusing the (much larger) train micro-batch for eval risks "
                        "OOMing partway through the first eval even though training itself fits. "
                        "Auto scales down from --batch-size so eval sees roughly the same total "
                        "frame count per batch as training.")
    p.add_argument("--num-workers", type=int, default=24,
                   help="DataLoader worker processes for audio decode/resample/feature-"
                        "extraction (default 24 -- raised from 16 because the 15k baseline run "
                        "measured only ~20%% total CPU on a 38-core box, and 16 workers can "
                        "occupy at most ~42%% of it, i.e. roughly half the workers were idle. "
                        "That headroom is what pays for --aug. 16 was itself raised from a "
                        "hardcoded 2 after diagnosing "
                        "GPU utilization stuck at 62-82%% during training on an H100 with 38 "
                        "CPU cores (100%% during eval, where 30s clips give the GPU enough work "
                        "per sample to hide the same 2 workers' decode cost -- it's the short "
                        "4s train crops that expose the CPU-side bottleneck). Only one loader "
                        "is ever active at a time, so this isn't split across train/val/casa/"
                        "madis. Set to roughly your rental's core count or a bit below; going "
                        "much past ~24 tends to hit diminishing returns from worker-process "
                        "overhead before it helps. 0 disables multiprocess loading entirely.")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="batches each worker prefetches ahead (only applies when "
                        "--num-workers > 0).")

    # -- optimization details --
    p.add_argument("--clip-norm", type=float, default=3.0,
                   help="gradient clipping max-norm. 1.0 -> 5.0 -> 3.0. The 5.0 pass over-"
                        "corrected: the llrd-1.0 run measured p50=1.99 p90=2.43 p95=2.66 "
                        "p99=4.72 and clipped only 1.0%% of steps, i.e. clipping had stopped "
                        "engaging at all. The target below is 5-10%%, and p95=2.66 puts that "
                        "near 3.0. Safe to move: corr(grad_norm[t], loss[t+1]) was -0.29/-0.33 "
                        "over the run, so there is no instability being held back. "
                        "Raised from 1.0 originally because the 15k baseline clipped "
                        "96.4%% of steps against a median pre-clip norm of ~5.5. Note what that "
                        "does under AdamW -- Adam is invariant to a UNIFORM rescaling of the "
                        "gradient (m and sqrt(v) scale together), so near-total clipping does "
                        "NOT lower the effective LR. Its only effect is to reweight ACROSS "
                        "steps: a batch with 3x the typical norm is down-weighted 3x relative "
                        "to a typical one, which systematically discounts the hard, atypical "
                        "batches that OOD robustness depends on, and it removes the natural "
                        "anneal where shrinking gradients produce shrinking steps. Clipping "
                        "should catch outlier steps, not act as a normalizer -- aim for 5-10%% "
                        "of steps clipped, and read the next value off the grad-norm "
                        "percentiles now printed in --spike-report.")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--bn-eval", dest="bn_eval", action="store_true", default=True,
                   help="freeze BatchNorm running stats in the Conformer conv blocks (default on "
                        "-- tiny-batch BN stats estimated on 4s crops otherwise get applied to "
                        "eval clips up to 6x longer).")
    p.add_argument("--no-bn-eval", dest="bn_eval", action="store_false")
    # -- third pass: generalization-gap levers (all default OFF) --
    p.add_argument("--layer-mix", action="store_true",
                   help="pool a learned softmax-weighted sum of ALL encoder layer outputs instead "
                        "of last_hidden_state alone. This is an ASR encoder: its top layer is "
                        "specialised toward transcription output, while accent/dialect cues sit "
                        "disproportionately in intermediate layers, so last-layer pooling is "
                        "likely reading the wrong representation. Adds one scalar per layer "
                        "(~48 parameters), trained at --head-lr. The learned weights are printed "
                        "and plotted at every eval -- that distribution is itself the answer to "
                        "which layers carry the dialect signal.")
    p.add_argument("--layer-mix-stride", type=int, default=1, metavar="N",
                   help="use every Nth encoder layer in the --layer-mix sum (1 = all). Retaining "
                        "every hidden state costs roughly 1-2 GB of extra activation at batch "
                        "128; raise this if it OOMs before dropping the batch size.")
    p.add_argument("--lora", action="store_true",
                   help="train low-rank adapters on the encoder's linear projections with the "
                        "encoder itself frozen. WARNING -- the rationale for this is no longer "
                        "supported by the evidence. It was added as a REGULARIZER against a "
                        "diagnosed 7-point train/val gap, but at the v4 default of --llrd 1.0 "
                        "there is no such gap: train accuracy ends at ~83%% (on augmented 3-12s "
                        "crops) against id_acc 89.7%% on clean clips, i.e. the model is UNDER-"
                        "fitting. Constraining it further is the opposite of what that says to "
                        "do. Kept because it is also a legitimate efficiency trick and because "
                        "the diagnosis could change again, but do not reach for it as a fix for "
                        "overfitting without re-measuring the gap first.")
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank r.")
    p.add_argument("--lora-alpha", type=float, default=32.0,
                   help="LoRA scaling; the update is (alpha/r) * B(A(x)).")
    p.add_argument("--lora-lr", type=float, default=5e-4,
                   help="LR for LoRA parameters. Far above the full fine-tune encoder LR "
                        "(2e-5), which is standard for adapters -- they start at zero and have "
                        "to move much further. LLRD does NOT apply to LoRA params.")
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="dropout on the LoRA branch input.")
    p.add_argument("--lora-targets",
                   default="q_proj,k_proj,v_proj,o_proj,out_proj,linear_q,linear_k,linear_v,linear_out",
                   help="comma-separated substrings matched against encoder module names to pick "
                        "which nn.Linear layers get adapters. The default covers the usual "
                        "attention-projection namings. This checkpoint's internals are "
                        "undocumented, so every wrapped module is printed and a zero-match is a "
                        "hard failure rather than a silent head-only run.")
    p.add_argument("--tta", type=int, default=0, metavar="N",
                   help="test-time augmentation: average logits over N evenly-spaced crops per "
                        "eval clip (0/1 disables). Applied ONLY at the final eval and under "
                        "--eval-checkpoint -- it multiplies eval wall-time by N, and intermediate "
                        "evals run every --eval-every steps. Costs no training compute.")
    p.add_argument("--tta-seconds", type=float, default=None,
                   help="crop length for --tta, in seconds. Default: the longest --crop-set entry "
                        "(or --crop), i.e. match the longest duration the model trained on.")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight decay for matrix parameters (biases/norms stay at 0). Was "
                        "hardcoded at 0.01; exposed because it is one of the few regularizers "
                        "here that costs nothing to turn up -- but see --lora: at --llrd 1.0 the "
                        "model underfits (train ~83%% vs id_acc 89.7%%), so turning this UP is "
                        "very unlikely to be the right move. More data is.")

    p.add_argument("--pool", choices=["mean", "mean_std", "attn_stat"], default="mean",
                   help="mean (default) is mask-aware mean pooling -- the bug fix for "
                        "cohere_bench.py's mean-over-padding. mean_std / attn_stat are the "
                        "experiments described in the plan's Suggestions section.")

    # -- GPU utilization --
    p.add_argument("--compile", dest="compile", action="store_true", default=True,
                   help="torch.compile the encoder for the (static-shape) train forward, with "
                        "automatic eager fallback on any compile failure (default on -- H100s "
                        "are badly launch-overhead-bound on short --crop clips otherwise).")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--compile-mode", default="default",
                   help="torch.compile mode. 'default' fuses kernels without CUDA graphs -- "
                        "safe with the requires_grad flip at unfreeze and set_to_none grad "
                        "reallocation. Try 'reduce-overhead' (CUDA graphs) once 'default' is "
                        "confirmed stable for a higher ceiling.")
    p.add_argument("--eval-autocast", dest="eval_autocast", action="store_true", default=True,
                   help="run eval forward passes under the same autocast as training (default "
                        "on). Eval clips are up to 30s vs 4s train crops, so eval was the only "
                        "part of the loop hitting 100%% GPU util -- this cuts its wall-time "
                        "~2-3x. Off reproduces the fp32-eval numbers in older results files "
                        "exactly, for apples-to-apples comparison.")
    p.add_argument("--no-eval-autocast", dest="eval_autocast", action="store_false")
    p.add_argument("--profile-steps", type=int, default=0, metavar="N",
                   help="time the first N optimizer steps by phase (dataloader wait / H2D / "
                        "forward / backward / opt step) and print a breakdown, plus the "
                        "one-time encoder input/output shapes -- use this to verify the "
                        "launch-overhead diagnosis instead of assuming it. 0 disables.")

    # -- eval / logging --
    p.add_argument("--eval-madis", dest="eval_madis", action="store_true", default=True,
                   help="evaluate MADIS-5 alongside Casablanca at every checkpoint (default on, "
                        "per request -- cohere_bench.py had this off by default).")
    p.add_argument("--no-eval-madis", dest="eval_madis", action="store_false")
    p.add_argument("--eval-every", type=int, default=400,
                   help="optimizer steps between full evals (800 keeps ~10 eval points across "
                        "the shortened 8000-step default run).")
    p.add_argument("--plot-every", type=int, default=500,
                   help="optimizer steps between loss/gradient/LR diagnostic plots.")
    p.add_argument("--extra-train-data", default="",
                   help="comma-separated paths to pre-materialized HF datasets (save_to_disk "
                        "format) to APPEND to the training split. Built for vc_augment.py's "
                        "kNN-VC voice-conversion output (arXiv:2505.24713), but any dataset with "
                        "'audio' and 'dialect' columns works. Eval sets are never touched -- "
                        "voice conversion is a training-time augmentation, and augmenting the "
                        "eval sets would invalidate every comparison to earlier runs. This is "
                        "the generic escape hatch; --use-vc and --use-adi17 are the named "
                        "sources, and all three share one validation path.")

    # -- extra TRAINING data sources.  ALL DEFAULT OFF.  ------------------------------------
    # Every source that is enabled is listed in the DATASETS USED banner printed after the
    # merge, is folded into ARGS.variant (so two runs with different data can never collide in
    # --results), and is recorded in the results row. A run with no flags here trains on ADI20
    # alone and is directly comparable to the v3 numbers.
    p.add_argument("--use-vc", action="store_true",
                   help="append the private voice-converted dataset from --vc-repo to the TRAIN "
                        "split (default OFF). This is vc_augment.py's kNN-VC output: the same "
                        "utterances in different speakers' voices, which targets speaker "
                        "invariance rather than channel robustness.")
    p.add_argument("--vc-repo", default=VC_REPO_DEFAULT, metavar="REPO_OR_PATH",
                   help="hub dataset repo id (private is fine, the token is used) or a local "
                        "path for --use-vc. A path may be a single save_to_disk dataset OR a "
                        "vc_augment.py output directory of shards.")
    p.add_argument("--vc-max-frac", type=float, default=0.5, metavar="F",
                   help="cap the SYNTHETIC share of the merged corpus at this fraction, by "
                        "seeded subsampling of the converted rows (1.0 disables the cap). "
                        "vc_augment.py --voices 4 emits four converted clips per source clip, "
                        "so an uncapped merge is ~80%% synthetic -- the model would then spend "
                        "most of its updates on kNN-VC artifacts rather than on real speech, and "
                        "the only way to dial that back used to be regenerating the dataset.")
    p.add_argument("--train-on-casa", action="store_true",
                   help="add Casablanca's UNUSED programs to the TRAIN split (default OFF). "
                        "load_casablanca keeps only --casa-val-frac of programs per country for "
                        "evaluation and DISCARDS the rest; this flag recovers that discarded half "
                        "as training data. It is program-disjoint from both the selection and the "
                        "held-out halves by construction, so casa_acc stays an honest selection "
                        "metric -- but it stops being an OUT-OF-DOMAIN one, because the model has "
                        "now seen Casablanca programs. Motivation: ADI17 and ADI20 are the same "
                        "corpus (ADI17 dev IS the ADI20 validation set), the model already scores "
                        "~90%% on both, and adding 32k ADI17 clips moved the leaderboard by -0.01. "
                        "Casablanca is the only labelled corpus where the model actually FAILS "
                        "(~64%%), it is TV/broadcast like the test set, and 39%% of its clips are "
                        "under 3s against ADI20's 0.3%%.")
    p.add_argument("--casa-train-frac", type=float, default=1.0, metavar="F",
                   help="use only this fraction of the recovered Casablanca training programs "
                        "(1.0 = all). Casablanca covers 8 of 20 countries, four of which are "
                        "already over-predicted sinks, so dialing this down limits how far the "
                        "merged class prior tilts toward them.")
    p.add_argument("--use-adi17", action="store_true",
                   help="append a targeted subset of ADI17's TRAIN split (default OFF). ADI17 "
                        "covers 17 of the 20 ADI20 countries with identical 3-letter codes -- it "
                        "has no MSA, BAH or TUN. See --adi17-alloc for how many clips each "
                        "dialect gets and why, and --adi17-split for the leakage rule.")
    p.add_argument("--adi17-split", default="train",
                   help="which ADI17 split to draw TRAINING data from. Only 'train' is "
                        "permitted. 'dev' is REFUSED because ADI17 dev IS the NADI ADI20 "
                        "validation set -- the per-class counts match exactly (ADI17 dev 8557 "
                        "clips; NADI val 10806 minus MSA 283, BAH 317, TUN 1649 = 8557), which "
                        "follows from ADI-20 being an extension of ADI-17 (arXiv:2511.10070). "
                        "'test' is REFUSED because it is very likely the hidden NADI test set by "
                        "the same argument. Training on either would make id_acc meaningless and "
                        "contaminate a competition submission. Use --eval-adi17-test to SCORE on "
                        "the test split, which is safe and useful.")
    p.add_argument("--adi17-alloc", default="", metavar="SPEC",
                   help="partial override of the per-dialect clip allocation, e.g. "
                        "'JOR=4000,ALG=4000'. Unlisted dialects keep their ADI17_ALLOC default. "
                        "The defaults are not uniform: each dialect's budget is proportional to "
                        "its ROW error (its clips get misrouted) plus its COLUMN error (it is "
                        "the sink others fall into), summed over the in-domain and Casablanca "
                        "confusion matrices weighted equally, over an 800-clip floor. The column "
                        "term is why SYR and MOR get large budgets despite good recall of their "
                        "own -- sharpening a boundary needs data on both sides of it.")
    p.add_argument("--adi17-per-class", type=int, default=None, metavar="N",
                   help="flat N clips per dialect, overriding the ADI17_ALLOC table entirely. "
                        "Mainly for smoke tests (--adi17-per-class 5).")
    p.add_argument("--adi17-cache-dir", default="./adi17_subset", metavar="DIR",
                   help="save_to_disk cache for the fetched ADI17 subset. A cache hit skips the "
                        "whole index-and-fetch path, which matters because a rented box re-runs "
                        "this on every experiment.")
    p.add_argument("--adi17-max-gb", type=float, default=12.0, metavar="GB",
                   help="refuse to start the ADI17 fetch if the selected parquet row groups add "
                        "up to more than this. The default allocation projects to ~9.3 GB (32000 "
                        "clips at the measured 100 rows / 29 MB per row group); the full train "
                        "split is 260 GB, so a bad --adi17-alloc is an expensive mistake to "
                        "discover halfway through.")
    p.add_argument("--eval-adi17-test", action="store_true",
                   help="evaluate ADI17's TEST split (12.2k clips, ~3 GB) at the FINAL eval "
                        "(default OFF). Since ADI17 dev turned out to be the NADI validation "
                        "set, ADI17 test is the closest available proxy for the leaderboard. It "
                        "is report-only and can never influence checkpoint selection: it runs "
                        "after training ends and after `best` is already fixed, and "
                        "--select-metric refuses every adi17_* name.")
    p.add_argument("--adi17-test-every", type=int, default=0, metavar="N",
                   help="also evaluate ADI17 test every N steps during training (0 = final eval "
                        "only, the default). Costs a 12.2k-clip pass per eval point. Still never "
                        "selectable -- but seeing the curve makes it tempting to pick a step by "
                        "eye, which is the same bias by hand, so this warns when set.")
    p.add_argument("--sampler-alpha", default="auto", metavar="A",
                   help="class-rebalancing strength for the train sampler: each row is drawn "
                        "with weight proportional to count(class)**-alpha. 0.0 = plain shuffle "
                        "(v3 behavior), 1.0 = full class balance, 0.5 = halfway. 'auto' (the "
                        "default) means 0.5 when any extra source is enabled and 0.0 otherwise. "
                        "Partial rather than full by default for two reasons: the ADI20 train "
                        "split is already near-balanced in hours (5.2-8.9 h per class, the one "
                        "outlier being BAH at 22.74 h, whose 100%% recall comes with 46 false "
                        "positives), and full balancing would UPWEIGHT TUN and MSA -- neither "
                        "appears in Casablanca at all and TUN is a false attractor there, so "
                        "alpha 1.0 plausibly costs casa_acc. Run 0.0/0.5/1.0 as an A/B.")
    p.add_argument("--allow-partial-shards", action="store_true",
                   help="permit a vc_augment.py output directory whose .done.json sidecars "
                        "outnumber its surviving shard directories. Off by default: that state "
                        "is what --prune-local leaves behind after uploading, so accepting it "
                        "silently trains on a fraction of the data. Point --vc-repo at the hub "
                        "repo id instead and the complete copy is downloaded.")
    p.add_argument("--confusion-by-program", action="store_true",
                   help="break the Casablanca confusion down by (true class, PROGRAM) as well as "
                        "by class, and write it beside the matrices. Casablanca is split by "
                        "program, so this distinguishes a genuine class-boundary failure from a "
                        "single broadcast the model reads wrong. Motivating case: JOR->SYR was "
                        "EXACTLY 292 clips in both the llrd 0.95 and llrd 1.0 runs -- identical "
                        "while every neighbouring cell moved -- which is the signature of a "
                        "domain artifact rather than a decision boundary. If those 292 are one "
                        "or two programs, no amount of extra JOR training data will move them.")
    p.add_argument("--tag", default="",
                   help="short experiment name appended to every output path AND to CFG['run']. "
                        "Needed whenever two runs differ by a flag that is not --layer-mix or "
                        "--lora -- e.g. an --llrd sweep. Without it both runs resolve to the "
                        "same --results file, and run_bench()'s already_done() check finds the "
                        "first run's completed row and SKIPS the second entirely, printing one "
                        "'SKIP (done)' line and exiting successfully having trained nothing. It "
                        "would also overwrite the first run's loss_history and plots.")
    p.add_argument("--select-metric", default="casa_acc",
                   help="eval metric used to pick the best checkpoint. Default casa_acc is the "
                        "20-way Casablanca accuracy on the SELECTION half (the historical "
                        "behavior). casa_acc8 selects on the 8-way restricted argmax instead -- "
                        "use it if the 8-way number is the one you report, because the two can "
                        "and do move in opposite directions (in the llrd-1.0 run casa_acc8_"
                        "holdout peaked at step 2000 and fell 2.5 pts by 4000 while casa_acc "
                        "rose throughout). Two COMPOSITES are also available, for when in-domain "
                        "and OOD both matter: 'mean_id_casa' = (id_acc + casa_acc)/2, and "
                        "'neg_mean_cavg' = -(id_cavg + casa_cavg20)/2, which is the competition "
                        "metric on both sets and the better of the two if Cavg is what you "
                        "report. Never select on a *_holdout metric: that would destroy the only "
                        "unbiased number you have. Every adi17_* name is REFUSED outright -- "
                        "ADI17 test is a proxy for the hidden test set, and selecting on it is "
                        "the same bias --casa-select-frac exists to remove.")
    p.add_argument("--casa-val-frac", type=float, default=0.5, metavar="F",
                   help="fraction of Casablanca PROGRAMS per country reserved for evaluation. "
                        "The remaining 1-F was silently discarded before --train-on-casa existed; "
                        "with that flag on, 1-F is exactly what gets trained on. Lowering this "
                        "trades evaluation coverage for training data: 0.5 (the default, and what "
                        "every run so far used) splits it evenly, 0.15 puts ~85%% into training "
                        "and keeps a ~2k-clip sanity check. Do NOT set it to 0 -- with Casablanca "
                        "the only broadcast-domain eval you have and MADIS shown to be actively "
                        "misleading for this task, a run with no Casablanca eval at all cannot be "
                        "sanity-checked before you spend a submission on it. Changing this "
                        "CHANGES THE EVAL SET, so casa_* numbers stop being comparable to earlier "
                        "runs; only the leaderboard remains comparable.")
    p.add_argument("--casa-select-frac", type=float, default=0.5,
                   help="fraction of the loaded Casablanca PROGRAMS used for checkpoint "
                        "selection; the rest is held out and reported separately. The baseline "
                        "both selected the best checkpoint on casa_acc and reported casa_acc "
                        "from the same clips, which is optimistically biased if that number is "
                        "reported externally. 1.0 restores the old (biased) behavior.")
    p.add_argument("--run-log", default=None,
                   help="path for a full stdout/stderr transcript of the run (default: "
                        "<results-stem>_run.log). The 15k baseline persisted NOTHING to disk -- "
                        "no CSV, no jsonl, no plots dir, no console log -- so its setup lines "
                        "(resolved encoder mask kwarg, discovered layer count, effective batch) "
                        "could not be checked afterwards. Set to '' to disable.")
    p.add_argument("--aug-selftest", action="store_true",
                   help="microbenchmark and validate every augmentation on synthetic audio, "
                        "check the per-batch crop mask arithmetic, and print whether the "
                        "measured aug cost fits the dataloader budget -- then exit. Needs torch/"
                        "torchaudio but NO model download, NO dataset and NO GPU, so it is the "
                        "cheap check to run on a rented box before starting a real run.")
    p.add_argument("--plot-selftest", action="store_true",
                   help="render a synthetic loss history (with a deliberate mid-run instability) "
                        "and exit -- exercises the whole plotting/spike-report path with no GPU "
                        "or downloads, before spending real training time.")
    # Output paths. All are relative to the working directory the script is launched from, not
    # to src/ -- run from the repo root and everything lands in one place (and is gitignored).
    p.add_argument("--results", default="train_results.jsonl")
    p.add_argument("--progress", default="train_progress.jsonl",
                   help="mid-training checkpoint evals, appended as they happen")
    p.add_argument("--loss-csv", default="loss_history.csv")
    p.add_argument("--spike-report", default="spike_report.txt")
    p.add_argument("--plots-dir", default="plots")

    # -- run control (same shape as cohere_bench.py) --
    p.add_argument("--confusion", action="store_true",
                   help="print a per-class confusion report (recall/precision, worst confusion "
                        "pairs, and a within-region vs cross-region error split) at every eval, "
                        "and write the full 20x20 matrix to --plots-dir. Always on for "
                        "--eval-checkpoint and for the final eval of a run.")
    p.add_argument("--eval-checkpoint", default=None, metavar="PATH",
                   help="load a saved best_*.pt, run the full evaluation with --confusion, and "
                        "exit WITHOUT training. This is the cheap way to find out where a "
                        "finished run's errors actually are -- one eval pass, no optimizer, no "
                        "training compute.")
    p.add_argument("--preflight", action="store_true",
                   help="smoke test + load data, then exit without training")
    p.add_argument("--smoke-only", action="store_true",
                   help="run the pre-download smoke test only, then exit")
    p.add_argument("--skip-smoke", action="store_true",
                   help="skip the pre-download smoke test (not recommended)")
    p.add_argument("--skip-install", action="store_true",
                   help="skip the pip install/upgrade step (assume deps are already present). "
                        "Safe and faster once a run has completed the install step once; on a "
                        "fresh box it just moves the failure to the first import.")
    p.add_argument("--torch-index-url", default=None,
                   help="pip --index-url used ONLY when torch/torchaudio have to be installed "
                        "from scratch, e.g. https://download.pytorch.org/whl/cu121 for a "
                        "specific CUDA build. Default (unset) uses PyPI, whose Linux wheels are "
                        "already CUDA-enabled. Never used to upgrade an existing torch.")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable W&B logging (CSV + jsonl still written)")
    p.add_argument("--wandb-project", default=WANDB_PROJECT)
    p.add_argument("--wandb-entity", default=WANDB_ENTITY,
                   help="W&B team/user; omit to use your default")
    p.add_argument("--wandb-group", default=None,
                   help="group name; defaults to a timestamp so one run = one group")
    return p.parse_args()

ARGS = get_args()

# --quick-data implies --skip-ood unless --no-skip-ood overrides it.
if ARGS.quick_data and not ARGS.no_skip_ood:
    ARGS.skip_ood = True

# --eval-checkpoint never trains, so the smoke test (which builds a second copy of the model and
# runs two optimizer steps on it) is pure wasted time and VRAM here.
if ARGS.eval_checkpoint:
    ARGS.skip_smoke = True


# ============================================================================
# Post-parse argument resolution -- runs before ANY output so the run log below captures
# everything, and before the pip install so a bad --aug/--crop-set fails in a second instead
# of after a multi-minute dependency install.
# ============================================================================
import os as _os

CHEAP_AUGS = ("noise", "reverb", "speed", "gain")
SLOW_AUGS  = ("pitch", "codec")
VALID_AUGS = CHEAP_AUGS + SLOW_AUGS

def _parse_csv_list(raw, name):
    return [tok.strip() for tok in str(raw or "").split(",") if tok.strip()]


# ============================================================================
# Label space + ADI17 constants.  Defined HERE, in the light post-parse section, rather
# than down in Config: --adi17-alloc names dialects, and validating those names needs the
# country list. Everything in this section exists so that a bad flag fails in a second
# instead of after a multi-minute dependency install and a model download.
# ============================================================================
COUNTRIES = ['MSA','BAH','TUN','ALG','EGY','IRA','JOR','KSA','KUW','LEB',
             'LIB','MAU','MOR','OMA','PAL','QAT','SUD','SYR','UAE','YEM']
labels2id = {k: i for i, k in enumerate(COUNTRIES)}
id2labels = {i: k for k, i in labels2id.items()}

# ---------------------------------------------------------------------------------------------
# ADI17 -- ArabicSpeech/ADI17, the dataset ADI-20 extends (arXiv:2511.10070).
#
# Its 17 dialects are exactly COUNTRIES minus {MSA, BAH, TUN}, and it uses the IDENTICAL 3-letter
# codes, so no label mapping is needed -- only the assertion in load_adi17_subset(). The three
# missing classes are the reason for --sampler-alpha: a plain concat with a 17-class source
# pushes those three priors down in proportion to ADI17's share of the corpus.
# ---------------------------------------------------------------------------------------------
ADI17_REPO       = "ArabicSpeech/ADI17"
ADI17_ABSENT     = ("MSA", "BAH", "TUN")
ADI17_COUNTRIES  = [c for c in COUNTRIES if c not in ADI17_ABSENT]   # 17, in COUNTRIES order

# Per-dialect clip budget for --use-adi17.  NOT uniform: uniform spends the budget on classes
# that already work. Each dialect's score is
#
#     score(c) = row_errors(c) + column_errors(c),  summed over BOTH confusion matrices
#
# where `row` = "this class fails" (its clips get misrouted) and `column` = "this class steals"
# (it is the sink other classes fall into). The two matrices are weighted equally because their
# totals are close (1114 in-domain vs 1185 Casablanca) and because both metrics matter -- the
# in-domain error is Gulf-dominated while the Casablanca error is Levantine/Maghrebi-dominated,
# so scoring on either alone misallocates badly.
#
# The column term is what earns SYR and MOR their budgets: SYR's own recall is fine (94% ID) but
# it absorbs 292 JOR + 36 PAL on Casablanca. Sharpening a boundary needs data on BOTH sides.
#
# Scored on the --llrd 1.0 run, NOT the older 0.95 one. That matters: the optimizer fix removed a
# third of the errors and removed them unevenly (KUW recall 57.6% -> 75.5%, OMA 66.6% -> 78.2%),
# so the stale matrices would have spent ~8000 clips on a Gulf problem that no longer exists.
#
#     alloc(c) = 800 floor + 18400 * score(c)/sum(score),  rounded to the nearest 100
#
# Totals 32000 clips ~= 81 h, taking the corpus from 163.8 h to ~244 h (+49%). Measured against
# the real repo, train row groups hold 100 rows in ~29 MB, so 32000 clips is ~320 row groups and
# ~9.3 GB -- and per-dialect counts round up in units of 100 clips, which is fine granularity.
#
# CAVEAT on JOR: JOR->SYR was EXACTLY 292 in both the 0.95 and 1.0 runs -- unchanged while every
# neighbouring cell moved. That is the signature of a domain artifact, not a decision boundary.
# Casablanca is split by PROGRAM, and 292 of ~421 JOR clips is plausibly one or two Jordanian
# broadcasts. Run --confusion-by-program against an existing checkpoint BEFORE spending this
# budget: if the errors are concentrated in one or two programs, cut JOR to the 800 floor and
# move the rest to UAE/QAT/ALG, because no amount of JOR training data will touch them.
ADI17_ALLOC = {
    "UAE": 3300,   # score 562 -- the biggest sink: QAT->UAE 143 (largest ID cell), OMA->UAE 27,
                   #              KUW->UAE 24, plus 137 Casablanca row errors
    "JOR": 3000,   # score 499 -- 35% of Casablanca error; see the CAVEAT above before spending
    "QAT": 2800,   # score 456 -- 320 ID row errors, worst remaining in-domain class (recall 65%)
    "SYR": 2700,   # score 438 -- almost all column mass (351 casa + 61 ID): the sink, not the
                   #              failure
    "OMA": 2400,   # score 370 -- 138 ID row + 144 casa column (JOR->OMA 42, UAE->OMA 65)
    "ALG": 2400,   # score 363 -- 24% of Casablanca error; ->MOR 156, ->TUN 72
    "MOR": 2000,   # score 268 -- 159 casa column mass; ALG's sink
    "KUW": 1700,   # score 208 -- halved by the llrd fix (was 2nd tier on the 0.95 matrices)
    "EGY": 1700,   # score 200
    "PAL": 1600,   # score 174 -- PAL->SYR 36
    "YEM": 1500,   # score 166
    "LIB": 1400,   # score 140 -- LIB<->TUN 32/38, and TUN cannot be topped up from ADI17
    "MAU": 1200,   # score  96
    "KSA": 1200,   # score  88 -- OMA's #2 sink
    "LEB": 1100,   # score  76 -- floor territory
    "SUD": 1000,   # score  56 -- floor
    "IRA": 1000,   # score  39 -- floor: 99% ID recall, small column mass
}
assert sorted(ADI17_ALLOC) == sorted(ADI17_COUNTRIES), \
    "ADI17_ALLOC must cover exactly the 17 ADI17 dialects"


ARGS.aug_list = _parse_csv_list(ARGS.aug, "--aug")
_bad_augs = [a for a in ARGS.aug_list if a not in VALID_AUGS]
if _bad_augs:
    print(f"FAIL: unknown --aug value(s) {_bad_augs}. Valid: {list(VALID_AUGS)}")
    sys.exit(2)

try:
    ARGS.crop_list = [float(t) for t in _parse_csv_list(ARGS.crop_set, "--crop-set")]
except ValueError:
    print(f"FAIL: --crop-set {ARGS.crop_set!r} is not a comma-separated list of numbers.")
    sys.exit(2)
if any(c <= 0 for c in ARGS.crop_list):
    print(f"FAIL: --crop-set values must all be > 0, got {ARGS.crop_list}")
    sys.exit(2)
ARGS.extra_train_list = _parse_csv_list(ARGS.extra_train_data, "--extra-train-data")
ARGS.lora_target_list = _parse_csv_list(ARGS.lora_targets, "--lora-targets")
if ARGS.lora and not ARGS.lora_target_list:
    print("FAIL: --lora given but --lora-targets is empty -- nothing would be adapted and the "
          "run would train the head only.")
    sys.exit(2)
if ARGS.lora and ARGS.lora_rank < 1:
    print(f"FAIL: --lora-rank must be >= 1, got {ARGS.lora_rank}")
    sys.exit(2)
if ARGS.layer_mix_stride < 1:
    print(f"FAIL: --layer-mix-stride must be >= 1, got {ARGS.layer_mix_stride}")
    sys.exit(2)
if ARGS.tta and ARGS.tta < 0:
    print(f"FAIL: --tta must be >= 0, got {ARGS.tta}")
    sys.exit(2)

if ARGS.crop_list and not ARGS.crop:
    # --crop 0 means "no cropping at all"; --crop-set means "crop, to a varying length". They
    # are contradictory, and silently letting one win would be a quiet correctness surprise.
    print("FAIL: --crop-set is set but --crop is 0 (cropping disabled). Pick one.")
    sys.exit(2)

# -- extra-data flag validation -------------------------------------------------------------
# ADI17 dev IS the NADI ADI20 validation set and test is very likely the hidden NADI test set,
# so this is a hard refusal rather than a documented convention. See --adi17-split's help for
# the arithmetic. Checked here, before the dependency install, so the mistake costs a second.
if ARGS.adi17_split != "train":
    print(f"FAIL: --adi17-split {ARGS.adi17_split!r} -- only 'train' may be used as a TRAINING "
          "source.")
    if ARGS.adi17_split == "dev":
        print("  ADI17 dev IS the NADI ADI20 validation set. Per-class counts match exactly:")
        print("    ADI17 dev total                          = 8557 clips")
        print("    NADI val 10806 - MSA 283 - BAH 317 - TUN 1649 = 8557 clips")
        print("  (ADI-20 is an extension of ADI-17 -- arXiv:2511.10070.) Training on it would")
        print("  make id_acc meaningless.")
    elif ARGS.adi17_split == "test":
        print("  ADI17 test is very likely the hidden NADI test set, by the same argument that")
        print("  makes ADI17 dev the NADI validation set. Training on it would contaminate a")
        print("  competition submission.")
        print("\n  To SCORE on it (safe, and the point of having it), use --eval-adi17-test.")
    sys.exit(2)

if ARGS.select_metric.startswith("adi17"):
    print(f"FAIL: --select-metric {ARGS.select_metric!r} is refused. ADI17 test is the closest "
          "available proxy for\n  the hidden NADI test set, so selecting a checkpoint on it is "
          "exactly the bias that\n  --casa-select-frac exists to remove. It is reported, never "
          "selected on.")
    sys.exit(2)

if not 0.0 < ARGS.vc_max_frac <= 1.0:
    print(f"FAIL: --vc-max-frac must be in (0, 1], got {ARGS.vc_max_frac}")
    sys.exit(2)
if ARGS.adi17_max_gb <= 0:
    print(f"FAIL: --adi17-max-gb must be > 0, got {ARGS.adi17_max_gb}")
    sys.exit(2)
if ARGS.adi17_per_class is not None and ARGS.adi17_per_class < 1:
    print(f"FAIL: --adi17-per-class must be >= 1, got {ARGS.adi17_per_class}")
    sys.exit(2)

# Resolve the per-dialect ADI17 budget HERE rather than at data-load time. A typo in
# --adi17-alloc otherwise costs a full dependency install, a model download and an ADI20 download
# before it is noticed -- and the whole reason this section exists is that bad flags should fail
# in a second.
if ARGS.adi17_per_class is not None:
    _alloc = {c: int(ARGS.adi17_per_class) for c in ADI17_COUNTRIES}
else:
    _alloc = dict(ADI17_ALLOC)
for _tok in _parse_csv_list(ARGS.adi17_alloc, "--adi17-alloc"):
    if "=" not in _tok:
        print(f"FAIL: --adi17-alloc entry {_tok!r} is not DIALECT=N")
        sys.exit(2)
    _k, _, _v = _tok.partition("=")
    _k = _k.strip().upper()
    if _k not in ADI17_COUNTRIES:
        print(f"FAIL: --adi17-alloc names {_k!r}, which is not one of the 17 ADI17 dialects.")
        if _k in ADI17_ABSENT:
            print(f"  ADI17 contains no {_k} at all -- it covers ADI20 minus "
                  f"{list(ADI17_ABSENT)}, so there is nothing to allocate.")
        else:
            print(f"  Valid: {ADI17_COUNTRIES}")
        sys.exit(2)
    try:
        _alloc[_k] = int(_v)
    except ValueError:
        print(f"FAIL: --adi17-alloc {_tok!r}: {_v!r} is not an integer.")
        sys.exit(2)
    if _alloc[_k] < 0:
        print(f"FAIL: --adi17-alloc {_tok!r}: must be >= 0")
        sys.exit(2)
ARGS.adi17_alloc_resolved = {c: _alloc[c] for c in ADI17_COUNTRIES}
if ARGS.use_vc and not str(ARGS.vc_repo or "").strip():
    print("FAIL: --use-vc given but --vc-repo is empty and VC_REPO_DEFAULT is unset. Pass "
          "--vc-repo <hub repo id or local path>, or set VC_REPO_DEFAULT at the top of this "
          "file.")
    sys.exit(2)

# --sampler-alpha 'auto' resolves against whether any extra source is on, which is only knowable
# after the flags above are parsed. Stored as a float from here on; ARGS.sampler_alpha_auto keeps
# the provenance for the banner, because "0.5" and "auto -> 0.5" mean different things when you
# are reading back an old run log.
if not 0.0 < ARGS.casa_val_frac <= 1.0:
    print(f"FAIL: --casa-val-frac must be in (0, 1], got {ARGS.casa_val_frac}. Zero would leave "
          "no Casablanca evaluation at all.")
    sys.exit(2)
if ARGS.casa_val_frac != 0.5 and not ARGS.train_on_casa:
    print(f"NOTE: --casa-val-frac {ARGS.casa_val_frac} without --train-on-casa just DISCARDS "
          f"{100*(1-ARGS.casa_val_frac):.0f}% of Casablanca instead of training on it.")

ARGS.sampler_alpha_auto = str(ARGS.sampler_alpha).strip().lower() == "auto"
_any_extra = bool(ARGS.use_vc or ARGS.use_adi17 or ARGS.train_on_casa
                  or ARGS.extra_train_data.strip())
if ARGS.sampler_alpha_auto:
    ARGS.sampler_alpha = 0.5 if _any_extra else 0.0
else:
    try:
        ARGS.sampler_alpha = float(ARGS.sampler_alpha)
    except ValueError:
        print(f"FAIL: --sampler-alpha must be a number or 'auto', got {ARGS.sampler_alpha!r}")
        sys.exit(2)
    if ARGS.sampler_alpha < 0:
        print(f"FAIL: --sampler-alpha must be >= 0, got {ARGS.sampler_alpha}")
        sys.exit(2)

# Variant tag from the feature flags. The planned A/B is two runs differing only by --lora, and
# with identical defaults they would collide in two ways -- the second overwrites the first's
# loss_history/plots, and, far worse, run_bench()'s already_done() check would find the first
# run's (CFG["run"], model) pair sitting in results.jsonl with status="ok" and SKIP the second
# run entirely. That failure is silent: it prints one "SKIP (done)" line and exits successfully,
# having produced no second data point. Tagging by variant makes it impossible.
#
# The data sources get INDEPENDENT components. v3 emitted a single bare "_vc" for any non-empty
# --extra-train-data, so a VC-only run and a VC+ADI17 run resolved to the same --results path and
# the second was silently skipped -- the precise failure this tag exists to prevent, reintroduced
# by the flag that was supposed to be covered by it.
ARGS.variant = (("_lm" if ARGS.layer_mix else "") + ("_lora" if ARGS.lora else "")
                + ("_vc" if ARGS.use_vc else "")
                + ("_adi17" if ARGS.use_adi17 else "")
                + ("_casatr" if ARGS.train_on_casa else "")
                + ("_extra" if ARGS.extra_train_data.strip() else "")
                + (("_" + ARGS.tag.strip().strip("_")) if ARGS.tag else ""))

# Absolute output paths, resolved ONCE here. The 15k baseline wrote its artifacts to a relative
# CWD on a rented box and none of them survived, which is why its diagnostics had to be
# reconstructed by eye from a PNG.
for _attr in ("results", "progress", "loss_csv", "spike_report", "plots_dir"):
    _val = _os.path.abspath(getattr(ARGS, _attr))
    if ARGS.variant:
        if _attr == "plots_dir":
            _val += ARGS.variant
        else:
            _root, _ext = _os.path.splitext(_val)
            _val = _root + ARGS.variant + _ext
    setattr(ARGS, _attr, _val)
if ARGS.run_log is None:
    ARGS.run_log = _os.path.splitext(ARGS.results)[0].replace("_results", "") + "_run.log"
elif ARGS.run_log:
    ARGS.run_log = _os.path.abspath(ARGS.run_log)


class _Tee:
    """Mirrors a stream to a log file. Line-buffered and flushed on every write so a killed
    run still leaves a complete transcript up to the moment it died.
    """
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, data):
        self._stream.write(data)
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except Exception:
            pass

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, item):
        return getattr(self._stream, item)


if ARGS.run_log:
    try:
        _os.makedirs(_os.path.dirname(ARGS.run_log) or ".", exist_ok=True)
        _log_fh = open(ARGS.run_log, "a", encoding="utf-8", errors="replace")
        _log_fh.write(f"\n\n{'='*66}\n=== run started: {' '.join(sys.argv)}\n{'='*66}\n")
        sys.stdout = _Tee(sys.stdout, _log_fh)
        sys.stderr = _Tee(sys.stderr, _log_fh)
        print(f"run log -> {ARGS.run_log}")
    except Exception as e:
        print(f"could not open --run-log {ARGS.run_log!r} ({type(e).__name__}: {e}) -- "
              "continuing without a transcript.")

print("output paths:")
for _attr in ("results", "progress", "loss_csv", "spike_report", "plots_dir"):
    print(f"  {_attr:<14s} {getattr(ARGS, _attr)}")
if ARGS.aug_list:
    print(f"augmentation: {ARGS.aug_list}  (cheap p={ARGS.aug_prob}, slow p={ARGS.aug_prob_slow})")
if ARGS.crop_list:
    print(f"crop policy: per-batch random from {ARGS.crop_list} s")
if ARGS.layer_mix:
    print(f"layer-mix: ON (stride {ARGS.layer_mix_stride}) -- pooling a learned weighted sum "
          "over encoder layers, not last_hidden_state")
if ARGS.lora:
    print(f"LoRA: ON  r={ARGS.lora_rank} alpha={ARGS.lora_alpha} lr={ARGS.lora_lr} "
          f"dropout={ARGS.lora_dropout} targets={ARGS.lora_target_list}")
if ARGS.tta and ARGS.tta > 1:
    print(f"TTA: {ARGS.tta} crops per eval clip (final eval / --eval-checkpoint only)")

# Which optional TRAINING sources were requested, said once up front. The authoritative version
# is the DATASETS USED banner printed after the merge (it has real row counts); this early line
# exists so a run log opened at the top already answers "what data was this?" -- and so that a
# forgotten flag is visible before the multi-minute install and download rather than after.
_requested = ([("vc", ARGS.vc_repo)] if ARGS.use_vc else []) \
           + ([("adi17", f"ArabicSpeech/ADI17[{ARGS.adi17_split}]")] if ARGS.use_adi17 else []) \
           + [("extra", _p) for _p in ARGS.extra_train_list]
if ARGS.train_on_casa:
    _requested = [("casa", "UBC-NLP/Casablanca[unused programs]")] + _requested
if _requested:
    print("extra TRAIN data requested (all default-off, so these were asked for explicitly):")
    for _name, _src in _requested:
        print(f"  {_name:<6s} {_src}")
    print(f"  sampler-alpha {ARGS.sampler_alpha}"
          + (" (auto: extra sources are enabled)" if ARGS.sampler_alpha_auto else " (explicit)"))
else:
    print("extra TRAIN data: none -- ADI20 only (--use-vc / --use-adi17 are off by default). "
          "This run is directly comparable to the v3 numbers.")
if ARGS.eval_adi17_test:
    print("ADI17 test EVAL: on (report-only -- runs after training, can never affect selection)")



# ============================================================================
# Install/upgrade dependencies -- runs BEFORE any heavy import, so a broken environment
# fails inside pip (with pip's own error) rather than as a confusing ImportError deep in
# torch/transformers. Deliberately does NOT touch `torch` itself -- see cohere_bench.py's
# identical rationale. matplotlib is added here for the loss/gradient/LR plots.
# ============================================================================
import subprocess

def _installed_version(pkg):
    """Version of an already-installed distribution, WITHOUT importing it -- importing torch here
    would cost ~10s and, worse, would happen before the pin below could protect it."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _ensure_torch_pair(pip, index_args):
    """Make sure a MATCHED torch/torchaudio pair exists, without ever moving an existing torch.

    The rule is "never silently replace a working torch", not "never install torch". Those are
    different, and conflating them is what made a fresh box fail at `import pandas` with nothing
    but a ModuleNotFoundError to go on.

      - both present  -> touch nothing, just pin them for the rest of the install.
      - both absent   -> nothing exists to break, so install them together in ONE pip command.
                         One command matters: resolving them separately is exactly how you end
                         up with a torchaudio built against a different libtorch.
      - torch only    -> add torchaudio at torch's own version with --no-deps, so pip cannot
                         drag torch along behind it.
      - torchaudio only -> unusual; let the pair install run and re-pin afterwards.

    torch and torchaudio have shared a version number since torch 2.0, which is what makes the
    "torch only" branch able to name the right torchaudio without guessing.
    """
    torch_v, ta_v = _installed_version("torch"), _installed_version("torchaudio")

    if torch_v and ta_v:
        print(f"  torch {torch_v} + torchaudio {ta_v} already present -- pinning, not touching.")
    elif not torch_v and not ta_v:
        print("  neither torch nor torchaudio is installed. Installing them TOGETHER in one "
              "resolution so the pair is guaranteed to match.")
        if not index_args:
            print("  (using the default PyPI wheels, which are CUDA-enabled on Linux. Pass "
                  "--torch-index-url https://download.pytorch.org/whl/cu121 -- or your driver's "
                  "cu tag -- if you need a specific CUDA build.)")
        subprocess.run(pip + index_args + ["torch", "torchaudio"], check=True)
        torch_v, ta_v = _installed_version("torch"), _installed_version("torchaudio")
        print(f"  installed torch {torch_v} + torchaudio {ta_v}")
    elif torch_v and not ta_v:
        base = torch_v.split("+")[0]
        print(f"  torch {torch_v} is present but torchaudio is not. Trying torchaudio=={base} "
              "with --no-deps so this cannot move torch.")
        # Not check=True: "no such version" is an expected outcome now, not a failure.
        subprocess.run(pip + ["--no-deps"] + index_args + [f"torchaudio=={base}"])
        ta_v = _installed_version("torchaudio")
        if not ta_v:
            # torchaudio's final release is 2.11 while PyTorch has moved past it, so on a current
            # image there is simply no torchaudio to pair with this torch. That is not an error
            # here: every audio op the script needs has a librosa/torch fallback (see the _ta_*
            # shims), and only the default-off `codec` aug actually requires torchaudio. Do NOT
            # "fix" this by installing a mismatched torchaudio -- that produces the
            # undefined-symbol crash this whole function exists to avoid.
            print(f"  no torchaudio matching torch {base} exists (torchaudio was sunset before "
                  "this torch version). Continuing WITHOUT it -- the script's librosa/torch "
                  "fallbacks cover everything except the 'codec' aug, which is off by default.")
            print("  Deliberately NOT installing a mismatched torchaudio: that is what causes "
                  "the 'undefined symbol: torch_library_impl' crash.")
    else:
        print(f"  torchaudio {ta_v} is present but torch is not -- installing the matching torch.")
        subprocess.run(pip + index_args + ["torch", "torchaudio"], check=True)
        torch_v, ta_v = _installed_version("torch"), _installed_version("torchaudio")

    return torch_v, ta_v


def install_dependencies():
    """Installs dependencies without ever silently REPLACING an existing torch/torchaudio.

    This function used to run `pip install -U transformers>=4.57 accelerate huggingface_hub`.
    The `-U` let pip's resolver upgrade torch transitively (accelerate declares a torch floor),
    and the second call listed a bare `torchaudio`, which installs the LATEST torchaudio -- and
    hence the latest torch -- whenever torchaudio happens to be absent. Either one silently
    replaces torch while leaving the pre-built torchaudio C extension behind, and the next run
    dies at import with

        OSError: .../_torchaudio.abi3.so: undefined symbol: torch_library_impl

    which is a torch<->torchaudio ABI mismatch, not a code problem. It bites hardest on hosts
    that layer a writable user env over a read-only system env (Lightning Studio, Colab, many
    rental images): pip writes the new torch into the user layer while torchaudio stays in the
    system layer, so the two resolve from different roots.

    The fix is a constraints file pinning whatever torch/torchaudio are present. Unlike dropping
    `-U`, a constraint binds the WHOLE resolution including transitive dependencies, so nothing
    can drag torch along behind our backs. See _ensure_torch_pair for the absent-torch case.
    """
    print("\n" + "=" * 66)
    print("  Installing/upgrading dependencies")
    print("=" * 66)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    index_args = ["--index-url", ARGS.torch_index_url] if ARGS.torch_index_url else []

    torch_v, ta_v = _ensure_torch_pair(pip, index_args)

    pins = [f"{p}=={v}" for p, v in (("torch", torch_v), ("torchaudio", ta_v)) if v]
    constraint_args = []
    if pins:
        # NamedTemporaryFile rather than mkstemp+os.fdopen: `os` is not imported until the light-
        # imports section further down, which runs AFTER this function is called.
        import tempfile
        with tempfile.NamedTemporaryFile("w", prefix="cohere_train_constraints_", suffix=".txt",
                                          delete=False) as fh:
            fh.write("\n".join(pins) + "\n")
            cpath = fh.name
        constraint_args = ["-c", cpath]
        print(f"  pinning {' '.join(pins)} for the rest of this install -- pip may not move them "
              "(prevents the torchaudio 'undefined symbol: torch_library_impl' ABI break)")

    # No -U: version floors (transformers>=4.57) already upgrade when genuinely needed, whereas
    # -U upgrades unconditionally and takes dependencies with it.
    subprocess.run(pip + constraint_args +
                   ["transformers>=4.57", "accelerate", "huggingface_hub"], check=True)
    # torchaudio is NOT listed here -- it is handled by _ensure_torch_pair above, where it can be
    # installed with --no-deps against the torch that already exists.
    subprocess.run(pip + constraint_args +
                   ["datasets==3.5.0", "soundfile", "librosa",
                    "pandas", "numpy", "tqdm", "wandb", "matplotlib"], check=True)
    print("dependencies installed.\n")

if ARGS.plot_selftest:
    if not ARGS.skip_install:
        print("--plot-selftest set: installing only the plotting deps (numpy/pandas/"
              "matplotlib) -- torch/transformers/datasets are NOT needed for this check.")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "numpy", "pandas", "matplotlib"], check=True)
elif not ARGS.skip_install:
    install_dependencies()
else:
    print("\n--skip-install set: assuming dependencies are already installed.\n")



# ============================================================================
# Light imports -- stdlib + numpy/pandas only. Enough to run --plot-selftest with NO torch,
# NO transformers, NO GPU, and NO dataset download, so the whole plotting/statistics path can
# be verified on a laptop before spending any real training time on a rented GPU box.
# ============================================================================
import os
import gc, json, math, random, traceback, inspect
from collections import defaultdict, Counter

try:
    import numpy as np, pandas as pd
except ImportError as _e:
    # --skip-install on a box that has not actually been set up yet used to die here with a bare
    # ModuleNotFoundError naming whichever package happened to be first, which says nothing about
    # the cause or the fix.
    _missing = getattr(_e, "name", None) or "numpy/pandas"
    print("\n" + "!" * 70)
    print(f"FAIL: {_missing} is not installed ({_e}).")
    if ARGS.skip_install:
        print("\n  --skip-install was passed, so the dependency step was skipped -- but the "
              "dependencies are not actually present on this machine.")
        print("  Re-run WITHOUT --skip-install once; it installs everything and pins "
              "torch/torchaudio so they cannot be disturbed:")
        print(f"\n    python {os.path.basename(sys.argv[0])} " +
              " ".join(a for a in sys.argv[1:] if a != "--skip-install"))
        print("\n  After that succeeds, --skip-install is safe (and faster) on every later run.")
    else:
        print("\n  The dependency install step ran but this package is still missing -- check "
              "the pip output above for the real error.")
    print("!" * 70 + "\n")
    sys.exit(1)


# ============================================================================
# Loss/gradient/LR diagnostic tracking + plotting -- defined here (before the torch/transformers
# imports below) so --plot-selftest never needs them.
# ============================================================================
class LossTracker:
    """Per-optimizer-step diagnostic log. Flushed to CSV periodically via a tmp-file + os.replace
    swap, so the file is always fully readable even if a run is killed mid-flush.
    """
    def __init__(self, path, flush_every=500):
        self.path = path
        self.rows = []
        self.flush_every = flush_every
        self._last_flush = 0

    def add(self, **kwargs):
        self.rows.append(kwargs)
        if len(self.rows) - self._last_flush >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        tmp = self.path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.path)
        self._last_flush = len(self.rows)

    def dataframe(self):
        return pd.DataFrame(self.rows)


def label_smoothing_floor(eps, n_classes):
    """Minimum achievable value of nn.CrossEntropyLoss(label_smoothing=eps) -- the entropy of the
    smoothed target distribution, NOT zero. At the defaults (eps=0.1, K=20) this is 0.594, so a
    training loss sitting at 0.75 has only ~0.16 of real headroom left, not 0.75. Plotting the
    smoothed loss against an implied floor of 0 is what made the baseline run look like it had
    stalled with lots left on the table when it was actually ~79% of the way to its floor.
    """
    if not eps:
        return 0.0
    q_correct = 1.0 - eps + eps / n_classes
    q_other = eps / n_classes
    return -(q_correct * math.log(q_correct)
             + (n_classes - 1) * q_other * math.log(max(q_other, 1e-12)))


def grad_norm_percentiles(df, window=3000):
    """p50/p90/p95/p99 of the pre-clip gradient norm over the trailing window. Clipping is meant
    to catch outlier steps, not to normalize every step -- so the right --clip-norm is roughly
    p90-p95, and these are the numbers to read it off.
    """
    if df is None or "grad_norm_preclip" not in df or len(df) < 20:
        return None
    gn = df.tail(window)["grad_norm_preclip"].dropna()
    gn = gn[np.isfinite(gn)]
    if not len(gn):
        return None
    return {f"p{q}": float(np.percentile(gn, q)) for q in (50, 90, 95, 99)}


def render_plots(df, eval_rows, out_dir, tag, clip_norm, unfreeze_step=None,
                 label_smoothing=0.0, n_classes=20):
    """5-panel diagnostic figure: loss, loss-instability, gradient norm, per-group LR, eval
    metrics. Written to plots/loss_step{N}.png and overwritten to plots/loss_latest.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df is None or len(df) == 0:
        return None
    os.makedirs(out_dir, exist_ok=True)
    steps = df["step"].values
    loss = df["loss_smoothed"].values

    fig, axes = plt.subplots(5, 1, figsize=(11, 19))

    ax = axes[0]
    ax.scatter(steps, loss, s=4, alpha=0.15, color="tab:blue", label="per-step loss (smoothed CE)")
    if len(df) >= 5:
        ax.plot(steps, pd.Series(loss).rolling(100, min_periods=1).mean(),
                color="tab:blue", lw=1.5, label="rolling mean (100)")
        ax.plot(steps, pd.Series(loss).rolling(500, min_periods=1).mean(),
                color="navy", lw=2, label="rolling mean (500)")
        # The unsmoothed CE has always been logged to the CSV but was never drawn, so the only
        # curve anyone ever saw was the one with an invisible non-zero floor under it.
        if "loss_raw_ce" in df:
            ax.plot(steps, pd.Series(df["loss_raw_ce"].values).rolling(500, min_periods=1).mean(),
                    color="tab:cyan", lw=2, ls="-.", label="raw CE, rolling mean (500)")
    floor = label_smoothing_floor(label_smoothing, n_classes)
    if floor > 0:
        ax.axhline(floor, color="tab:green", ls="--", lw=1.5,
                   label=f"label-smoothing floor ({floor:.3f})")
        ax.axhspan(0, floor, color="tab:green", alpha=0.06)
        # Headroom, so the plateau can be read as "how much is actually left" rather than
        # "distance to zero", which is not a reachable target with label smoothing on.
        if len(loss) >= 500:
            tail = float(pd.Series(loss).tail(500).mean())
            ax.annotate(f"headroom above floor: {tail - floor:+.3f}",
                        xy=(steps[-1], tail), xytext=(-8, 14),
                        textcoords="offset points", ha="right", fontsize=8, color="tab:green")
    ax.axhline(math.log(n_classes), color="gray", ls="--", lw=1,
               label=f"chance (ln {n_classes})")
    if unfreeze_step:
        ax.axvline(unfreeze_step, color="red", ls=":", lw=1, label="encoder unfreeze")
    ax.set_ylabel("loss"); ax.set_title(f"{tag} -- training loss")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[1]
    roll_std = pd.Series(loss).rolling(200, min_periods=10).std()
    ax.plot(steps, roll_std, color="tab:orange")
    if unfreeze_step:
        ax.axvline(unfreeze_step, color="red", ls=":", lw=1)
    ax.set_ylabel("rolling std of loss (w=200)")
    ax.set_title("loss instability -- this is the number to watch for the fluctuation issue")
    ax.grid(alpha=0.3)

    ax = axes[2]
    if "grad_norm_preclip" in df:
        gn = df["grad_norm_preclip"].values
        ax.scatter(steps, gn, s=4, alpha=0.15, color="tab:green")
        ax.plot(steps, pd.Series(gn).rolling(100, min_periods=1).median(),
                color="darkgreen", lw=1.5, label="rolling median (100)")
        ax.axhline(clip_norm, color="gray", ls="--", lw=1, label=f"clip norm ({clip_norm})")
        clipped_pct = 100 * df["was_clipped"].mean() if "was_clipped" in df else float("nan")
        # Percentiles drawn in, so the next --clip-norm can be read straight off this panel:
        # clipping should catch the tail (p90-p95), not renormalize the whole distribution.
        pct = grad_norm_percentiles(df, window=len(df))
        subtitle = ""
        if pct:
            for q, color in (("p50", "0.45"), ("p90", "0.6"), ("p95", "0.7")):
                ax.axhline(pct[q], color=color, ls=":", lw=1)
                ax.annotate(f"{q}={pct[q]:.2f}", xy=(steps[0], pct[q]), xytext=(2, 2),
                            textcoords="offset points", fontsize=7, color="0.35")
            subtitle = (f"\np50={pct['p50']:.2f}  p90={pct['p90']:.2f}  "
                        f"p95={pct['p95']:.2f}  p99={pct['p99']:.2f}")
        ax.set_title(f"gradient norm (pre-clip) -- {clipped_pct:.1f}% of steps clipped overall"
                     f"{subtitle}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    ax.set_yscale("log"); ax.set_ylabel("grad norm"); ax.grid(alpha=0.3)

    ax = axes[3]
    for col, lab, color in [("lr_head", "head", "tab:purple"),
                             ("lr_enc_top", "encoder top layer", "tab:red"),
                             ("lr_enc_bottom", "encoder bottom layer", "tab:brown")]:
        if col in df:
            ax.plot(steps, df[col], label=lab, color=color)
    ax.set_yscale("log"); ax.set_ylabel("learning rate")
    ax.set_title("LR schedule -- read from the optimizer, i.e. what actually happened")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[4]
    # Train accuracy has always been logged (batch_acc) but never plotted. Drawing it on the same
    # axes as the eval curves is what makes the real failure mode legible: in the 15k baseline
    # ID accuracy climbed 77->85 while Casablanca sat at ~58 and MADIS-5 drifted DOWN, i.e. the
    # extra training was buying in-domain fit and no transfer at all.
    if "batch_acc" in df and len(df) >= 5:
        ax.plot(steps, pd.Series(df["batch_acc"].values).rolling(500, min_periods=1).mean() * 100,
                color="tab:olive", lw=1.2, alpha=0.8, label="train acc (rolling 500)")
    ev = pd.DataFrame(eval_rows) if eval_rows else pd.DataFrame()
    if len(ev):
        if "id_acc" in ev: ax.plot(ev["step"], ev["id_acc"], marker="o", label="ID acc", color="tab:blue")
        if "id_acc_macro" in ev: ax.plot(ev["step"], ev["id_acc_macro"], marker=".", ls=":", label="ID acc (macro)", color="tab:cyan")
        if "casa_acc" in ev: ax.plot(ev["step"], ev["casa_acc"], marker="o", label="Casablanca acc (select, 20-way)", color="tab:red")
        if "casa_acc_holdout" in ev: ax.plot(ev["step"], ev["casa_acc_holdout"], marker="s", ls="--", label="Casablanca acc (held out, 20-way)", color="darkred")
        # 8-way: argmax restricted to the countries Casablanca actually contains. The 20-way
        # curves above count every prediction of an absent country as an error by construction.
        if "casa_acc8_holdout" in ev: ax.plot(ev["step"], ev["casa_acc8_holdout"], marker="^", ls=":", label="Casablanca acc (held out, 8-way)", color="orangered")
        if "madis_overall" in ev: ax.plot(ev["step"], ev["madis_overall"], marker="o", label="MADIS-5 overall", color="tab:green")
        if "gap" in ev: ax.plot(ev["step"], ev["gap"], marker="x", ls="--", label="ID-OOD gap", color="gray")
    if len(ev) or "batch_acc" in df:
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("accuracy (%) / gap (pts)")
    ax.set_title("evaluation metrics -- watch train-vs-OOD divergence, not the loss curve")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    step_path = os.path.join(out_dir, f"loss_step{int(steps[-1]):06d}.png")
    latest_path = os.path.join(out_dir, "loss_latest.png")
    fig.savefig(step_path, dpi=110)
    fig.savefig(latest_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {step_path} (and loss_latest.png)")
    return step_path, latest_path


def write_spike_report(df, out_path, window=3000, k=6.0):
    """Robust-baseline outlier flagging over the trailing `window` steps: median + k*MAD. Prints
    and appends a block to `out_path` with enough context (grad norm, LR, clip length) per spike
    to tell whether it's an LR problem, a clipping problem, or a data problem.
    """
    if df is None or len(df) < 20:
        return []
    recent = df.tail(window)
    med = recent["loss_smoothed"].median()
    mad = (recent["loss_smoothed"] - med).abs().median() * 1.4826
    thresh = med + k * max(mad, 1e-6)
    spikes = recent[recent["loss_smoothed"] > thresh]

    lines = [f"-- spike report @ step {int(df['step'].iloc[-1])} "
             f"(trailing {window} steps, baseline median={med:.3f}, threshold={thresh:.3f}) --"]

    # Gradient-norm distribution, so --clip-norm is set from data rather than inherited. Clipping
    # is supposed to catch outlier steps; if clipped% is up near 100 then it is not clipping, it
    # is renormalizing every step, which under AdamW does not change the effective LR but does
    # down-weight high-norm (hard) batches relative to typical ones.
    pct = grad_norm_percentiles(df, window=window)
    if pct:
        clipped_pct = 100 * df.tail(window)["was_clipped"].mean() if "was_clipped" in df else float("nan")
        lines.append(f"  grad norm (pre-clip), trailing {window}: "
                     f"p50={pct['p50']:.2f}  p90={pct['p90']:.2f}  "
                     f"p95={pct['p95']:.2f}  p99={pct['p99']:.2f}   "
                     f"[{clipped_pct:.1f}% of steps clipped]")
        if math.isfinite(clipped_pct) and clipped_pct > 50:
            lines.append(f"  ^ clipping {clipped_pct:.0f}% of steps is normalization, not "
                         f"outlier control -- consider --clip-norm {pct['p95']:.1f} "
                         f"(= p95, clips ~5% of steps)")

    if len(spikes):
        for _, r in spikes.iterrows():
            lines.append(
                f"  step {int(r['step']):>7d}  loss={r['loss_smoothed']:.3f}  "
                f"grad_norm={r.get('grad_norm_preclip', float('nan')):.2f}  "
                f"lr_head={r.get('lr_head', float('nan')):.2e}  "
                f"lr_enc_top={r.get('lr_enc_top', float('nan')):.2e}  "
                f"clipped={bool(r.get('was_clipped', False))}"
            )
        gn = recent["grad_norm_preclip"].values[:-1] if "grad_norm_preclip" in recent else np.array([])
        next_loss = recent["loss_smoothed"].values[1:]
        if len(gn) > 5 and np.std(gn) > 0 and np.std(next_loss) > 0:
            corr = float(np.corrcoef(gn, next_loss)[0, 1])
            lines.append(f"  corr(grad_norm[t], loss[t+1]) over trailing window: {corr:.3f} "
                         "(high positive => grad spikes are PRECEDING loss spikes -- tighten "
                         "--clip-norm or lower --lr)")
    else:
        lines.append(f"  no outlier steps in the trailing window (threshold {thresh:.3f})")

    with open(out_path, "a") as f:
        f.write("\n".join(lines) + "\n\n")
    for l in lines:
        print(" ", l)
    return lines


_SELECT_WARNED = []

_DIVERGENCE_STATE = {"consecutive_high": 0, "warned": False}
def check_divergence(df, frozen_steps):
    """Warns if the 500-step rolling loss mean stays near/above chance for 3 consecutive checks
    after the encoder unfreezes -- the same failure mode that made the ORIGINAL models_bench.py
    hyperparameters (lr=1e-4, freeze_first_n=6) collapse to 0% Casablanca accuracy.
    """
    post = df[df["step"] > frozen_steps]
    if len(post) < 500:
        return
    recent_mean = post["loss_smoothed"].tail(500).mean()
    ln_chance = math.log(20)
    if recent_mean > ln_chance * 0.95:
        _DIVERGENCE_STATE["consecutive_high"] += 1
    else:
        _DIVERGENCE_STATE["consecutive_high"] = 0
    if _DIVERGENCE_STATE["consecutive_high"] >= 3 and not _DIVERGENCE_STATE["warned"]:
        print("  " + "!" * 66)
        print("  DIVERGENCE WARNING: 500-step rolling loss has stayed near/above chance level "
              "(ln 20) for 3 consecutive checks after unfreeze. Consider lowering --lr/--head-lr, "
              "or tightening --clip-norm.")
        print("  " + "!" * 66)
        _DIVERGENCE_STATE["warned"] = True


def _plot_selftest():
    """No torch, no model, no downloads: synthesize a loss history with a deliberate mid-run
    instability (the shape you described -- fine early, then fluctuating) and exercise the full
    plotting + spike-report path.
    """
    print("\n--plot-selftest: synthesizing a loss history with a deliberate mid-run instability")
    rng = np.random.default_rng(0)
    n = 6000
    steps = np.arange(1, n + 1)
    frozen_steps = 500
    # Decays toward the label-smoothing floor plus a little headroom, not toward zero -- a real
    # smoothed-CE curve can never go below the floor, so a synthetic one that does would make the
    # new panel-1 annotation report a nonsensical negative headroom.
    _floor = label_smoothing_floor(ARGS.label_smoothing, 20)
    base = (math.log(20) - _floor) * np.exp(-steps / 800) + _floor + 0.16
    noise = rng.normal(0, 0.05, n)
    instability = np.where(steps > 3500, rng.normal(0, 0.4, n) * ((steps - 3500) / 500).clip(0, 1), 0)
    loss = np.clip(base + noise + instability, _floor, None)
    grad_norm = np.abs(rng.normal(1.0, 0.3, n)) + np.where(steps > 3500, np.abs(rng.normal(0, 2.0, n)), 0)
    # Synthetic train accuracy that keeps climbing while the OOD curves below flatten -- the
    # shape the real run had, so the new panel-5 overlay is actually exercised.
    train_acc = np.clip(0.20 + 0.68 * (1 - np.exp(-steps / 1500)) + rng.normal(0, 0.03, n), 0, 1)
    rows = []
    for i, s in enumerate(steps):
        rows.append(dict(
            step=int(s), loss_smoothed=float(loss[i]), loss_raw_ce=float(loss[i]) + 0.02,
            grad_norm_preclip=float(grad_norm[i]), was_clipped=int(grad_norm[i] > 1.0),
            batch_acc=float(train_acc[i]),
            lr_head=1e-3 * max(0.05, math.cos(s / n * math.pi / 2)),
            lr_enc_top=(0.0 if s < frozen_steps else 1e-5 * min(1.0, (s - frozen_steps) / 1000)),
            lr_enc_bottom=(0.0 if s < frozen_steps else 1e-6 * min(1.0, (s - frozen_steps) / 1000)),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(ARGS.loss_csv, index=False)
    # ID climbing while the OOD curves saturate, plus the held-out Casablanca split -- i.e. the
    # divergence the panel exists to make visible.
    eval_rows = [dict(step=s, id_acc=40 + s / n * 40, id_acc_macro=36 + s / n * 38,
                       casa_acc=30 + 28 * (1 - math.exp(-s / 1200)),
                       casa_acc_holdout=29 + 27 * (1 - math.exp(-s / 1200)),
                       madis_overall=25 + s / n * 30, gap=10 - s / n * 5)
                 for s in range(1500, n + 1, 1500)]
    render_plots(df, eval_rows, ARGS.plots_dir, "plot-selftest", ARGS.clip_norm,
                 unfreeze_step=frozen_steps, label_smoothing=ARGS.label_smoothing,
                 n_classes=len(COUNTRIES) if "COUNTRIES" in globals() else 20)
    write_spike_report(df, ARGS.spike_report)
    print(f"\nplot self-test complete -> {ARGS.plots_dir}/loss_latest.png, {ARGS.loss_csv}, "
          f"{ARGS.spike_report}")


if ARGS.plot_selftest:
    _plot_selftest()
    sys.exit(0)



# ============================================================================
# Heavy imports + environment -- torch/transformers/datasets are only needed past this point,
# i.e. never for --plot-selftest (see the light-imports section above).
# ============================================================================
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time

try:
    import torch, torch.nn as nn
except ImportError:
    # Distinct from the ABI-mismatch case below: torch is simply not installed. That is a
    # one-command fix, so say the command rather than re-raising the traceback.
    print("\n" + "!" * 70)
    print("FAIL: torch is not installed.")
    if ARGS.skip_install:
        print("\n  --skip-install was passed. Re-run without it -- the dependency step now "
              "installs torch and torchaudio TOGETHER when neither is present, which is the "
              "only way to guarantee the pair matches:")
        print(f"\n    python {os.path.basename(sys.argv[0])} " +
              " ".join(a for a in sys.argv[1:] if a != "--skip-install"))
    else:
        print("\n  The install step ran but torch is still missing -- check the pip output "
              "above. To install it by hand (both in ONE command, so the pair matches):")
        print("\n    pip install torch torchaudio")
        print("    # or, for a specific CUDA build:")
        print("    pip install torch torchaudio --index-url "
              "https://download.pytorch.org/whl/cu121")
    print("!" * 70 + "\n")
    sys.exit(1)

# torchaudio is OPTIONAL. It ships a compiled C extension linked against a specific libtorch, and
# as of torch 2.12 there is no matching release at all -- torchaudio's last version is 2.11, so a
# current PyTorch image simply has no torchaudio it can legally pair with. Making it a hard
# requirement would strand the script on exactly the newest boxes.
#
# Everything the script actually needs it for has an equivalent in librosa (already a dependency)
# or in torch itself; see the _ta_* shims below. Only the `codec` aug genuinely requires
# torchaudio.io, and that one is off by default.
#
# torchaudio is imported separately and guarded: it ships a compiled C extension linked against a
# specific libtorch, so a torch upgrade that leaves torchaudio behind produces an `undefined
# symbol` OSError here that says nothing about the actual problem or its fix. Every augmentation
# added in the second pass (functional.speed, functional.fftconvolve, transforms.PitchShift,
# io.AudioEffector) leans on torchaudio far more heavily than the original resample-only usage,
# so a half-working install is worth catching up front rather than 8000 steps in.
HAS_TORCHAUDIO = False
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    torchaudio = None
    print(f"  torchaudio is not installed (torch {torch.__version__}). Continuing without it -- "
          "resampling and the noise/reverb/speed/pitch augs use librosa/torch fallbacks. Only "
          "the 'codec' aug requires torchaudio and it will be skipped if enabled.")
except OSError as e:
    # Installed but its C extension will not load: a torch<->torchaudio ABI mismatch. Degrade to
    # the fallbacks rather than dying, but say so loudly -- silently running a different resampler
    # than a previous run used is the kind of difference that quietly moves numbers.
    torchaudio = None
    _tv = torch.__version__
    print("\n" + "!" * 70)
    print("WARNING: torchaudio is installed but its C extension will not load.")
    print(f"  {type(e).__name__}: {e}")
    print(f"\n  This is a torch <-> torchaudio ABI mismatch: torch is {_tv}, and the installed")
    print("  torchaudio was compiled against a different one. Continuing WITHOUT torchaudio,")
    print("  using the librosa/torch fallbacks.")
    print("\n  To fix it properly, reinstall a torchaudio built for this torch. --no-deps is")
    print("  essential -- without it pip will move torch again and you are back where you")
    print("  started:")
    print(f"    pip install --force-reinstall --no-deps torchaudio=={_tv.split('+')[0]}")
    print("  (If no such version exists, torchaudio has been sunset past your torch version;")
    print("   the fallbacks are then the intended path, not a workaround.)")
    print("!" * 70 + "\n")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm


# ============================================================================
# Audio-op shims -- torchaudio when it is available, librosa/torch otherwise.
#
# torchaudio's last release is 2.11 while PyTorch is past 2.12, so on a current image there is no
# torchaudio to install. Every op below has an equivalent that needs only librosa (already a
# dependency, and itself soxr-backed for resampling) or torch's own FFT. The one exception is the
# `codec` aug, which needs torchaudio.io and is off by default.
#
# These are NOT bit-identical to the torchaudio versions. That matters for cross-run comparison,
# so which path is in use is printed at startup and recorded with the results.
# ============================================================================
def _ta_resample(w, orig_sr, target_sr):
    """Resample a 1-D float32 tensor. Used in the collate for the ~1% of ADI20 clips that are not
    already 16 kHz, and for loading external noise/RIR files."""
    if orig_sr == target_sr:
        return w
    if HAS_TORCHAUDIO:
        return torchaudio.functional.resample(w, orig_sr, target_sr)
    import librosa
    return torch.from_numpy(
        librosa.resample(w.detach().cpu().numpy().astype(np.float32),
                         orig_sr=orig_sr, target_sr=target_sr)).to(w.dtype)


def _ta_fftconvolve(a, b):
    """Full linear convolution via FFT -- torchaudio.functional.fftconvolve's 'full' mode. The
    fallback is a direct torch.fft implementation, so it needs no extra dependency at all."""
    if HAS_TORCHAUDIO:
        return torchaudio.functional.fftconvolve(a, b)
    n = a.shape[-1] + b.shape[-1] - 1
    nfft = 1 << (n - 1).bit_length()          # next power of two, for a fast transform
    out = torch.fft.irfft(torch.fft.rfft(a, nfft) * torch.fft.rfft(b, nfft), nfft)
    return out[..., :n]


def _ta_speed(w, sr, factor):
    """Speed perturbation: resample by 1/factor and reinterpret at the original rate, so the clip
    gets shorter for factor>1. Length-changing by design -- the collate re-pads/truncates."""
    if HAS_TORCHAUDIO:
        out, _ = torchaudio.functional.speed(w.unsqueeze(0), sr, factor)
        return out.squeeze(0)
    import librosa
    return torch.from_numpy(
        librosa.resample(w.detach().cpu().numpy().astype(np.float32),
                         orig_sr=sr, target_sr=int(round(sr / factor)))).to(w.dtype)


def _ta_pitch_shift(w, sr, n_steps, _cache={}):
    """Pitch shift by n_steps semitones. torchaudio's transform caches its resample kernels, and
    librosa's phase vocoder is the fallback -- both are slow, which is why `pitch` is in
    SLOW_AUGS and fires at --aug-prob-slow."""
    if HAS_TORCHAUDIO:
        shifter = _cache.get(n_steps)
        if shifter is None:
            shifter = _cache[n_steps] = torchaudio.transforms.PitchShift(sr, n_steps)
        return shifter(w.unsqueeze(0)).squeeze(0)
    import librosa
    return torch.from_numpy(
        librosa.effects.pitch_shift(w.detach().cpu().numpy().astype(np.float32),
                                    sr=sr, n_steps=n_steps)).to(w.dtype)


def _ta_load(path):
    """Load an audio file as (1-D float32 tensor, sample_rate). soundfile is already a dependency
    and handles wav/flac/ogg; librosa covers the rest."""
    if HAS_TORCHAUDIO:
        w, sr = torchaudio.load(path)
        return w.mean(0), sr
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.mean(axis=1)), sr
    except Exception:
        import librosa
        data, sr = librosa.load(path, sr=None, mono=True)
        return torch.from_numpy(np.asarray(data, dtype=np.float32)), sr


AUDIO_BACKEND = "torchaudio" if HAS_TORCHAUDIO else "librosa/torch fallback"
print(f"audio ops backend: {AUDIO_BACKEND}")

import datasets
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

datasets.config.TORCHVISION_AVAILABLE = False   # keeps DataLoader from importing video ops

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Train batches are now fixed-shape (crop-length padded, see collate) so cuDNN's autotuner can
# actually reuse what it picks instead of re-benchmarking every differently-shaped batch.
torch.backends.cudnn.benchmark = True

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()
SR = TARGET_SR = 16000

print(f"torch {torch.__version__} | transformers {transformers.__version__} | "
      f"gpus {N_GPU} | device {device}")
for i in range(N_GPU):
    print("  ", torch.cuda.get_device_name(i))


# --- fail-fast dependency check ---
def _transformers_version_tuple():
    parts = []
    for p in transformers.__version__.split(".")[:3]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)

if _transformers_version_tuple() < (4, 57, 0):
    print(f"FAIL: transformers=={transformers.__version__} is installed, but "
          "CohereLabs/cohere-transcribe-arabic-07-2026 needs transformers>=4.57 for its "
          "trust_remote_code modeling/processing files.")
    print('  Fix:  pip install -U "transformers>=4.57" accelerate')
    sys.exit(1)
print(f"transformers=={transformers.__version__} OK (>=4.57 required)")


# --- credentials: environment only ---
# Nothing is read from the source file. v3 shipped a live token in a constant, and this file
# gets copied to rented boxes and pasted into bug reports, so the only supported source is the
# environment. Missing credentials stop the run with the export line to copy, rather than
# failing 20 minutes later inside a dataset download.
def _require_env(env_names, label, how):
    for n in env_names:
        if os.environ.get(n):
            return os.environ[n], f"env:{n}"
    print("\n" + "!" * 70)
    print(f"FAIL: no {label} credential. Set it in the environment:\n")
    print(f"    export {env_names[0]}=...        # {how}")
    print("\n  (Or copy .env.example to .env and fill it in.)")
    print("!" * 70 + "\n")
    sys.exit(1)

# --aug-selftest exits below without touching the network or the gated model (--plot-selftest
# has already exited further up), so it must stay runnable on a laptop with no credentials.
_NEEDS_CREDENTIALS = not ARGS.aug_selftest

USE_WANDB = not ARGS.no_wandb and _NEEDS_CREDENTIALS
WANDB_GROUP = None
if USE_WANDB:
    try:
        import wandb
        import time as _time
        # Unset WANDB_API_KEY is a hard stop rather than a silent downgrade: W&B is on by
        # default, so silently continuing without it loses the run's metrics. --no-wandb is
        # the way to opt out deliberately.
        _wk, _src = _require_env(["WANDB_API_KEY"], "W&B",
                                 "https://wandb.ai/authorize -- or pass --no-wandb")
        os.environ["WANDB_API_KEY"] = _wk
        try:
            wandb.login(key=_wk, relogin=True)
            print(f"W&B login ok ({_src})")
        except Exception as e:
            print(f"W&B login failed ({type(e).__name__}) -- disabling W&B, "
                  "run continues with CSV + logs.")
            USE_WANDB = False
        if USE_WANDB:
            WANDB_GROUP = ARGS.wandb_group or f"cohere-train-{_time.strftime('%Y%m%d-%H%M%S')}"
            print(f"W&B on: project={ARGS.wandb_project} group={WANDB_GROUP}")
    except ImportError:
        print("wandb not installed (pip install wandb) -- continuing without it.")
        USE_WANDB = False

# The encoder itself is gated, so there is no useful run without this -- stop here rather
# than 404 on the first model download. (ArabicSpeech/ADI17 is public and needs no token;
# gated Casablanca and a private --vc-repo do.)
if _NEEDS_CREDENTIALS:
    _tok, _tsrc = _require_env(["HF_TOKEN", "HUGGINGFACE_TOKEN"], "Hugging Face",
                               "https://huggingface.co/settings/tokens")
    try:
        from huggingface_hub import login
        login(_tok)
        print(f"HF login ok ({_tsrc})")
    except Exception as e:
        print(f"HF login failed: {type(e).__name__}: {e}")



# ============================================================================
# Config
# ============================================================================

MODEL_REGISTRY = {"cohere-ar": dict(hf_id="CohereLabs/cohere-transcribe-arabic-07-2026")}
BENCH_ORDER = ["cohere-ar"]

# Run name carries the variant too, so results.jsonl rows stay distinguishable even if someone
# points both runs at the same --results file on purpose.
CFG = dict(run="v3" + (ARGS.variant or "_base"), seed=SEED, crop=ARGS.crop)


def _explicit_flag(*names):
    """Best-effort check for whether any of these flags were passed on the command line (vs.
    just inheriting their default) -- argparse doesn't expose this directly.
    """
    return any(a == n or a.startswith(n + "=") for a in sys.argv[1:] for n in names)


# This script's lr/head-lr defaults are sqrt(effective_batch/32)-scaled off cohere_train.py's
# (effective batch 32, lr=1e-5, head_lr=1e-3). Print the derivation so it's visible in every run
# log, and warn if that pairing has been broken by a partial override.
_lr_scale = math.sqrt(ARGS.effective_batch / 32)
print(f"LR scaling: sqrt(--effective-batch/32) = sqrt({ARGS.effective_batch}/32) = "
      f"{_lr_scale:.3g}x cohere_train.py's lr=1e-5/head_lr=1e-3 -> this run's "
      f"lr={ARGS.lr:.1e}  head_lr={ARGS.head_lr:.1e}")
if _explicit_flag("--effective-batch") and not (_explicit_flag("--lr") and _explicit_flag("--head-lr")):
    print("  WARNING: --effective-batch was overridden on the command line but --lr and "
          "--head-lr were not BOTH also overridden -- the sqrt-scaling relationship this "
          f"script assumes no longer holds. For --effective-batch {ARGS.effective_batch}, "
          f"consider --lr {1e-5*_lr_scale:.1e} --head-lr {1e-3*_lr_scale:.1e}.")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed(CFG["seed"])
print("CFG:", CFG)
print("MODEL_REGISTRY:", MODEL_REGISTRY)
print(f"precision={ARGS.precision}  pool={ARGS.pool}  bn_eval={ARGS.bn_eval}  "
      f"llrd={ARGS.llrd}  lr={ARGS.lr}  head_lr={ARGS.head_lr}")


def random_crop(wav, seconds=None, sr=SR):
    seconds = CFG["crop"] if seconds is None else seconds
    if not seconds:
        return wav
    n = int(seconds * sr)
    if len(wav) <= n:
        return wav
    s = random.randint(0, len(wav) - n)
    return wav[s:s+n]

SKIPPED = {"n": 0}



# ============================================================================
# Train-time waveform augmentation
#
# Why this exists: the 15k baseline had NO augmentation beyond random_crop, and its failure mode
# was a widening ID-OOD gap -- in-domain accuracy climbed 77->85 from step 3k to 15k while
# Casablanca stayed at ~58 and MADIS-5 drifted down. That is domain overfitting, and the
# textbook lever for it is augmentation, not schedule tuning.
#
# Ported from Model_Test/NADI_2026_Musab_aug.ipynb cell 26 (same six augs, same 1-D mono float32
# 16kHz `(T,)` contract, which is exactly what the collate's _wav() already produces), with four
# changes needed to make them safe to run on-the-fly inside DataLoader workers:
#
#   1. Tiered by CPU cost. noise/reverb/speed/gain are O(n) or one FFT and fire at --aug-prob;
#      pitch (phase vocoder, ~20-50ms/clip) and codec (libav round-trip) fire at the much lower
#      --aug-prob-slow.
#   2. aug_pitch uses cached torchaudio.transforms.PitchShift objects instead of the functional
#      form, which rebuilds its resample kernel on every single call.
#   3. aug_codec uses torchaudio.io.AudioEffector (in-process libav) instead of spawning an
#      ffmpeg subprocess plus temp-file I/O per sample -- the notebook version costs ~50-200ms
#      and a process spawn per clip, which is not viable in a dataloader.
#   4. The MUSAN/RIR file listing is cached per worker instead of re-walking the directory tree
#      on every single call.
#
# Everything runs AFTER the crop, so an aug only ever sees --crop seconds of audio (~4-12s),
# never the up-to-30s source clip.
# ============================================================================
_AUG_FILE_CACHE = {}
_SYNTH_RIR_BANK = []
_AUG_WARNED = set()
_CODEC_AVAILABLE = None


def _aug_warn_once(key, msg):
    if key not in _AUG_WARNED:
        _AUG_WARNED.add(key)
        print(f"  [aug:{key}] {msg}")


def _list_audio_files(dirpath):
    """Directory walk, cached per (worker) process. The notebook re-walked the tree on every
    single augmented sample, which for a real MUSAN checkout is thousands of stat() calls per
    clip."""
    if not dirpath:
        return []
    if dirpath not in _AUG_FILE_CACHE:
        files = []
        if os.path.isdir(dirpath):
            files = [os.path.join(r, f) for r, _, fs in os.walk(dirpath)
                     for f in fs if f.lower().endswith((".wav", ".flac", ".ogg", ".mp3"))]
        _AUG_FILE_CACHE[dirpath] = files
        if not files:
            _aug_warn_once(f"dir:{dirpath}",
                           f"no audio files under {dirpath!r} -- falling back to synthetic.")
    return _AUG_FILE_CACHE[dirpath]


def _load_random(dirpath):
    files = _list_audio_files(dirpath)
    if not files:
        return None
    try:
        w, sr = _ta_load(random.choice(files))
    except Exception:
        return None
    return _ta_resample(w, sr, SR)


def aug_noise(wav):
    """Additive background noise at SNR ~ U(5,20) dB, so the model stops relying on clean audio.
    Real noise if --musan-dir is set, synthetic Gaussian otherwise."""
    snr_db = random.uniform(5, 20)
    noise = _load_random(ARGS.musan_dir)
    if noise is None or len(noise) == 0:
        noise = torch.randn(len(wav))
    if len(noise) < len(wav):
        noise = noise.repeat(int(math.ceil(len(wav) / len(noise))))
    off = random.randint(0, max(0, len(noise) - len(wav)))
    noise = noise[off:off + len(wav)]
    ps = wav.pow(2).mean()
    pn = noise.pow(2).mean().clamp_min(1e-10)
    scale = (ps / (pn * (10 ** (snr_db / 10)))).clamp_min(0).sqrt()
    return wav + scale * noise


def _synth_rir():
    """A small bank of synthetic exponential-decay IRs, built once per process. The notebook
    generated a fresh 4800-sample randn + exp on every call."""
    if not _SYNTH_RIR_BANK:
        n = int(0.3 * SR)
        decay = torch.exp(-torch.linspace(0, 8, n))
        for _ in range(8):
            r = torch.randn(n) * decay
            _SYNTH_RIR_BANK.append(r / r.norm().clamp_min(1e-8))
    return random.choice(_SYNTH_RIR_BANK)


def aug_reverb(wav):
    """Convolve with a room impulse response so the model can't key on the recording space.
    Length-preserving. Relevant to Casablanca, which is broadcast audio."""
    rir = _load_random(ARGS.rir_dir)
    if rir is None or len(rir) == 0:
        rir = _synth_rir()
    else:
        rir = rir / rir.norm().clamp_min(1e-8)
    peak = wav.abs().max()
    out = _ta_fftconvolve(wav, rir)[:len(wav)]
    return out / out.abs().max().clamp_min(1e-8) * peak


def aug_speed(wav):
    """Speed perturbation, so the model can't key on one speaker's rate. NOTE: this CHANGES the
    waveform length (0.9x -> longer, 1.1x -> shorter); the collate re-pads/truncates to the
    batch's crop length afterwards, and reports the true length for pooling."""
    factor = random.choice([0.9, 0.95, 1.05, 1.1])
    return _ta_speed(wav, SR, factor)


def aug_gain(wav):
    """Random level change, +/- 12 dB. Nearly free, and the recording-level distribution is one
    of the more obvious things that differs between ADI20 and a broadcast corpus."""
    return wav * (10 ** (random.uniform(-12, 12) / 20))


def aug_pitch(wav):
    """Pitch shift, so the model can't key on individual speakers' voices. Expensive (phase
    vocoder) -- fires at --aug-prob-slow, and the resample kernels are cached per process."""
    steps = random.choice([-2, -1, 1, 2])
    return _ta_pitch_shift(wav, SR, steps)


def _codec_available():
    global _CODEC_AVAILABLE
    if _CODEC_AVAILABLE is None:
        if not HAS_TORCHAUDIO:
            _CODEC_AVAILABLE = False
            _aug_warn_once("codec", "torchaudio is not available, and torchaudio.io is the one "
                                    "audio op in this script with no librosa/torch substitute -- "
                                    "the 'codec' aug is disabled for this run. Every other aug "
                                    "works normally.")
            return _CODEC_AVAILABLE
        try:
            # Both names, because aug_codec needs both -- probing only AudioEffector would let a
            # torchaudio that lacks CodecConfig report the aug as available and then fail per
            # clip inside the workers.
            from torchaudio.io import AudioEffector, CodecConfig
            _CODEC_AVAILABLE = AudioEffector is not None and CodecConfig is not None
        except Exception as e:
            _CODEC_AVAILABLE = False
            _aug_warn_once("codec", f"torchaudio.io.AudioEffector unavailable "
                                    f"({type(e).__name__}) -- 'codec' aug disabled for this run. "
                                    "It is NOT falling back to an ffmpeg subprocess: spawning a "
                                    "process per sample inside a DataLoader worker costs more "
                                    "than the aug is worth.")
    return _CODEC_AVAILABLE


def aug_codec(wav):
    """Lossy-codec round-trip, faking phone/broadcast compression artifacts. In-process via
    libav -- no subprocess, no temp files."""
    if not _codec_available():
        return wav
    from torchaudio.io import AudioEffector, CodecConfig
    fmt, bitrate_k = random.choice([("mp3", 64), ("ogg", 32), ("mp3", 32)])
    effector = AudioEffector(format=fmt, codec_config=CodecConfig(bit_rate=bitrate_k * 1000))
    out = effector.apply(wav.unsqueeze(-1), SR)   # AudioEffector wants (T, C)
    return out[:, 0].to(torch.float32)


AUG_FNS = {"noise": aug_noise, "reverb": aug_reverb, "speed": aug_speed, "gain": aug_gain,
           "pitch": aug_pitch, "codec": aug_codec}


def apply_augmentations(wav, aug_list, p_cheap, p_slow):
    """Applies each enabled aug independently with its tier's probability. A failure in one aug
    degrades to the un-augmented waveform for that aug rather than killing the worker -- one bad
    clip must not take down a 24-worker dataloader mid-run."""
    for name in aug_list:
        p = p_slow if name in SLOW_AUGS else p_cheap
        if random.random() >= p:
            continue
        try:
            out = AUG_FNS[name](wav)
            if out is not None and out.numel() > 0 and torch.isfinite(out).all():
                wav = out
        except Exception as e:
            _aug_warn_once(f"fail:{name}", f"skipped -- {type(e).__name__}: {e} "
                                           "(this warning prints once per aug per process)")
    return wav


def _aug_selftest():
    """Validates + microbenchmarks every aug, and checks the per-batch crop mask arithmetic.

    The point of the timing half is the question this design turns on: the augs run inside
    DataLoader workers, so they are only free while the workers stay ahead of the GPU. This
    prints the measured per-sample cost against the actual budget rather than assuming it.
    """
    print("\n" + "=" * 66)
    print("  AUG SELF-TEST (synthetic audio -- no model, no dataset, no GPU)")
    print("=" * 66)
    rng = np.random.default_rng(0)
    n_samples = int(4.0 * SR)
    base = torch.as_tensor(rng.standard_normal(n_samples) * 0.1, dtype=torch.float32)
    reps = 20
    failures, timings = [], {}

    # Resampling is not an augmentation -- the collate needs it for every clip that is not
    # already 16 kHz (about 1% of ADI20, the 44.1 kHz TUN files), so a broken shim would drop
    # those clips silently via _wav()'s except-return-None path rather than raising.
    print(f"  audio backend: {AUDIO_BACKEND}")
    # Measure under the conditions the augs actually run in: inside a DataLoader worker, which is
    # pinned to one thread (see _worker_init). Timing them with the main process's full thread
    # count would understate contention on a busy box and overstate per-op speed.
    _prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    print(f"  CPU: {os.cpu_count()} cores visible, torch default {_prev_threads} thread(s); "
          "timing below is pinned to 1 thread to match a dataloader worker")
    try:
        _rs_in = torch.as_tensor(rng.standard_normal(44100) * 0.1, dtype=torch.float32)
        # Warm up before timing. The first call pays `import librosa` (~1s) plus any numba JIT,
        # which is a one-time per-PROCESS cost -- each persistent DataLoader worker pays it once
        # at startup, not once per clip. Timing a single cold call reported ~1000 ms/clip and made
        # a ~1 ms operation look like the most expensive thing in the pipeline.
        _ta_resample(_rs_in, 44100, SR)
        t0 = time.perf_counter()
        for _ in range(reps):
            rs = _ta_resample(_rs_in, 44100, SR)
        dt_ms = (time.perf_counter() - t0) / reps * 1000
        expected = SR                                    # 1.0s at 44.1kHz -> 1.0s at 16kHz
        if not torch.isfinite(rs).all():
            failures.append("resample: produced non-finite samples")
        elif abs(rs.shape[-1] - expected) > SR * 0.02:   # 2% tolerance on filter edge effects
            failures.append(f"resample: 44.1kHz->16kHz gave {rs.shape[-1]} samples, expected "
                            f"~{expected}")
        else:
            print(f"  OK   {'resample':<7s} {dt_ms:7.2f} ms/clip (warm)  44.1kHz -> 16kHz, "
                  f"{rs.shape[-1]} samples (expected ~{expected})")
    except Exception as e:
        failures.append(f"resample: {type(e).__name__}: {e}")
        print(f"  FAIL {'resample':<7s} {type(e).__name__}: {e}")

    for name in VALID_AUGS:
        fn = AUG_FNS[name]
        try:
            t0 = time.perf_counter()
            out = None
            for _ in range(reps):
                out = fn(base.clone())
            dt_ms = (time.perf_counter() - t0) / reps * 1000
            # An aug whose backend is missing returns its input untouched. Reporting that as "OK"
            # alongside a timing is actively misleading -- it reads as a working aug that will
            # contribute to the run, when in fact it is a no-op.
            if name == "codec" and not _codec_available():
                print(f"  SKIP {name:<7s} disabled (needs torchaudio.io) -- it will NOT augment "
                      "anything if you pass it to --aug")
                continue
            if torch.equal(out, base):
                print(f"  SKIP {name:<7s} returned its input unchanged -- backend unavailable, "
                      "this aug is a no-op for this run")
                continue
            timings[name] = dt_ms
            if out is None or out.numel() == 0:
                failures.append(f"{name}: returned empty")
            elif not torch.isfinite(out).all():
                failures.append(f"{name}: produced non-finite samples")
            elif out.dtype != torch.float32:
                failures.append(f"{name}: dtype {out.dtype}, expected float32")
            elif out.ndim != 1:
                failures.append(f"{name}: ndim {out.ndim}, expected 1-D (T,)")
            else:
                length_note = ""
                if out.shape[-1] != n_samples:
                    length_note = (f"  [length {out.shape[-1]} != input {n_samples} -- the "
                                   "collate truncates/pads to the batch crop length]")
                tier = "slow" if name in SLOW_AUGS else "cheap"
                print(f"  OK   {name:<7s} {dt_ms:7.2f} ms/clip  ({tier}){length_note}")
        except Exception as e:
            timings[name] = float("nan")
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"  FAIL {name:<7s} {type(e).__name__}: {e}")

    # Expected per-sample cost of the default cheap stack, at the configured probabilities.
    cheap_cost = sum(timings.get(a, 0.0) * ARGS.aug_prob
                     for a in CHEAP_AUGS if not math.isnan(timings.get(a, float("nan"))))
    slow_cost = sum(timings.get(a, 0.0) * ARGS.aug_prob_slow
                    for a in SLOW_AUGS if not math.isnan(timings.get(a, float("nan"))))
    print(f"\n  expected added cost per sample: {cheap_cost:.2f} ms (all cheap augs @ p="
          f"{ARGS.aug_prob}) + {slow_cost:.2f} ms (all slow augs @ p={ARGS.aug_prob_slow})")
    workers = max(1, ARGS.num_workers)
    n_cores = os.cpu_count() or 1
    # Workers beyond the core count do not add throughput -- they add contention.
    effective_workers = min(workers, n_cores)
    per_batch_ms = (cheap_cost + slow_cost) * BATCH_SIZE * GRAD_ACCUM / effective_workers
    print(f"  at batch {BATCH_SIZE} x accum {GRAD_ACCUM} across {effective_workers} effective "
          f"worker(s) that is ~{per_batch_ms:.1f} ms of added wall-time per OPTIMIZER STEP if "
          "the workers are the bottleneck.")
    if workers > n_cores:
        print(f"  WARNING: --num-workers {workers} exceeds the {n_cores} visible CPU core(s). "
              f"Extra workers only add contention -- use about {n_cores}.")
    if cheap_cost > 50:
        print(f"  WARNING: {cheap_cost:.0f} ms/sample for the cheap augs is far above the ~5 ms "
              "these ops should cost. That usually means CPU contention or a throttled "
              "container, not expensive augmentation -- check the core count above and lower "
              "--num-workers to match it before concluding the augs are too slow.")
    print("  Compare against the real per-step time with --profile-steps: if 'dataloader_wait' "
          "stays a small share of the phase breakdown, the augs are free.")

    # Crop-mask arithmetic: the riskiest edit, because a wrong denominator mis-masks pooling
    # silently rather than raising. Mirrors _derive_frame_mask's fraction for each crop length.
    print("\n  crop mask arithmetic (mirrors DialectID._derive_frame_mask):")
    crop_list = ARGS.crop_list or [ARGS.crop]
    for c in crop_list:
        padded = int(c * SR)
        T = max(1, padded // 1280)          # ~8x-subsampled 80-frame-per-second stand-in
        true_lens = torch.tensor([padded, padded // 2, max(1, padded // 8)])
        frac = (true_lens.float() / padded).clamp(max=1.0)
        valid = (frac * T).round().long().clamp(min=1, max=T)
        ok = bool((valid <= T).all() and (valid >= 1).all())
        print(f"    crop {c:>5.1f}s  padded={padded:>6d}  T={T:>4d}  "
              f"valid frames for [full, half, eighth] = {valid.tolist()}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"crop mask arithmetic out of range at crop={c}")
    if len(crop_list) > 1:
        print(f"    -> {len(crop_list)} distinct shapes, so torch.compile builds "
              f"{len(crop_list)} graphs (not one per step)")

    print("\n" + "=" * 66)
    if failures:
        print(f"  AUG SELF-TEST: {len(failures)} problem(s)")
        for f in failures:
            print(f"    - {f}")
        print("=" * 66)
        torch.set_num_threads(_prev_threads)
        return 1
    torch.set_num_threads(_prev_threads)
    print("  AUG SELF-TEST PASSED")
    print("=" * 66)
    return 0



# ============================================================================
# Auto batch-size / grad-accum from VRAM
# ============================================================================
def auto_batch_and_accum(args):
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        total_mem = 0
    # With --crop-set the batch must be sized for the LONGEST crop in the set, not the nominal
    # --crop: otherwise a run sails through the 3s batches and OOMs the first time it happens to
    # draw 12s, thousands of steps in.
    max_crop = max(args.crop_list) if args.crop_list else args.crop
    if args.batch_size is not None:
        bs = args.batch_size
    elif total_mem >= 70:
        # A short --crop gives the encoder very few frames per sample (e.g. ~50 frames at 4s on
        # an 8x-subsampling FastConformer) -- H100s sit well below 100% util because there's not
        # enough GPU work per kernel launch. Widening the micro-batch (while holding effective
        # batch, hence LRs, constant via a smaller grad_accum) puts more work behind each launch.
        # Long crops already have enough per-sample work that this isn't needed. --crop 0
        # disables cropping entirely (full clips, up to MAX_AUDIO_SECONDS_COHERE=30s) -- that's
        # the opposite case, so it must NOT qualify here despite `0 <= 8.0`.
        # Cap raised 64->256 vs. cohere_train.py: this script's --effective-batch defaults to
        # 128, and the lower cap would silently clamp that to micro-batch 64 / grad_accum 2,
        # defeating the whole point of this variant.
        bs = (min(256, max(16, args.effective_batch // max(1, N_GPU)))
              if 0 < max_crop <= 8.0 else 16)
    elif total_mem >= 35:
        # 40GB-class card (A100-PCIE-40GB and similar). 8 is inherited from the original script
        # and is very conservative for short crops -- but --crop-set 3,5,8,12 makes the longest
        # batch 3x the memory of a 4s one, and --layer-mix retains every hidden state on top of
        # that, so the conservative value is the right default here. Override with --batch-size
        # once you have a peak-VRAM number from --smoke-only or --profile-steps; every doubling
        # halves grad_accum and therefore the per-step launch overhead.
        bs = 8
    else:
        bs = 2
    if args.grad_accum is not None:
        accum = args.grad_accum
    else:
        accum = max(1, round(args.effective_batch / (bs * max(1, N_GPU))))
    return bs, accum

BATCH_SIZE, GRAD_ACCUM = auto_batch_and_accum(ARGS)
print(f"batch_size={BATCH_SIZE}  grad_accum={GRAD_ACCUM}  "
      f"effective_batch~={BATCH_SIZE * GRAD_ACCUM * max(1, N_GPU)}  "
      f"(VRAM-detected: {'n/a' if device.type!='cuda' else f'{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB'})")

# Placed here rather than beside the aug definitions because the budget arithmetic needs
# BATCH_SIZE, which is only known once the VRAM probe above has run.
if ARGS.aug_selftest:
    sys.exit(_aug_selftest())



# ============================================================================
# Backbone wrapper (Cohere only) -- mask-aware pooling, full-encoder LLRD, frozen BatchNorm
# ============================================================================
def _encoder_layer_list_with_name(encoder):
    """Returns (qualified_name, ModuleList) for the encoder's main layer stack. Cohere's
    checkpoint is a Parakeet/FastConformer-style architecture with no `.layers` attribute the
    way Whisper's encoder has, so this falls back to walking named submodules for the first
    nn.ModuleList of reasonable depth -- every layer-stack encoder architecture has one.
    """
    layers = getattr(encoder, "layers", None)
    if layers is not None and len(layers) > 0:
        return "layers", layers
    for name, module in encoder.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            return name, module
    return None, []


def _encoder_layers(encoder):
    _, layers = _encoder_layer_list_with_name(encoder)
    return layers


def _resolve_mask_kwarg(encoder):
    """Cohere's gated encoder forward signature isn't documented -- probe it for whichever mask/
    length kwarg name it accepts, so we can actually pass the padding mask through instead of
    silently pooling over padding (the bug this script exists partly to fix).
    """
    try:
        sig = inspect.signature(encoder.forward)
    except (TypeError, ValueError):
        return None
    for name in ("attention_mask", "padding_mask", "input_lengths", "lengths", "feature_lengths"):
        if name in sig.parameters:
            return name
    return None


def _supports_hidden_states(encoder):
    """Whether the encoder's forward accepts output_hidden_states. Probed, not assumed: this is a
    trust_remote_code checkpoint with an undocumented signature, and --layer-mix is useless (and
    would silently degrade to last-layer pooling) if the kwarg is ignored.
    """
    try:
        sig = inspect.signature(encoder.forward)
    except (TypeError, ValueError):
        return False
    return ("output_hidden_states" in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))


class LoRALinear(nn.Module):
    """Frozen nn.Linear + a trainable low-rank update: y = base(x) + (alpha/r) * B(A(x)).

    B is initialised to ZERO so the wrapped model is numerically identical to the base model at
    step 0 -- that is what makes it safe to bolt onto a pretrained encoder without a warmup, and
    it means a smoke test that shows a changed output before any optimizer step has found a bug.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        lx = self.lora_dropout(x)
        # Cast to the LoRA parameter dtype: base may be running under autocast in bf16 while the
        # adapters are fp32 master weights, and F.linear will not promote for us.
        upd = torch.nn.functional.linear(
            torch.nn.functional.linear(lx.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return out + self.scaling * upd.to(out.dtype)


def apply_lora(encoder, targets, r, alpha, dropout, verbose=True):
    """Wrap every nn.Linear in `encoder` whose qualified name contains one of `targets`.

    Returns the list of wrapped module paths. The caller MUST treat an empty list as fatal: this
    checkpoint's internals are undocumented, so a targets list that matches nothing would leave
    the encoder entirely frozen and quietly train the classifier head alone -- a run that looks
    healthy, costs hours, and answers nothing.
    """
    to_wrap = []
    for name, mod in encoder.named_modules():
        if isinstance(mod, nn.Linear) and any(t in name for t in targets):
            to_wrap.append((name, mod))

    for name, mod in to_wrap:
        parent_path, _, attr = name.rpartition(".")
        parent = encoder.get_submodule(parent_path) if parent_path else encoder
        setattr(parent, attr, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))

    if verbose:
        print(f"  LoRA: wrapped {len(to_wrap)} nn.Linear module(s), r={r} alpha={alpha}")
        if to_wrap:
            shown = to_wrap[:8]
            for name, mod in shown:
                print(f"    {name}  ({mod.in_features} -> {mod.out_features})")
            if len(to_wrap) > len(shown):
                print(f"    ... and {len(to_wrap) - len(shown)} more")
            # Distinct trailing names, so a wrong --lora-targets is obvious at a glance.
            leaves = sorted({n.rsplit(".", 1)[-1] for n, _ in to_wrap})
            print(f"    distinct leaf names matched: {leaves}")
    return [n for n, _ in to_wrap]


class DialectID(nn.Module):
    """Cohere ASR encoder + mask-aware pool + linear head. forward(input_features=...,
    attention_mask=...) -> logits of shape (batch, num_labels).
    """

    def __init__(self, spec, num_labels=20, pool="mean", compile_encoder=False,
                 compile_mode="default", specaug=False, specaug_f=2, specaug_t=2,
                 layer_mix=False, layer_mix_stride=1, lora=False, lora_rank=16,
                 lora_alpha=32.0, lora_dropout=0.0, lora_targets=()):
        super().__init__()
        self._specaug_on = specaug
        self._specaug_f = specaug_f
        self._specaug_t = specaug_t
        self._specaug_checked = False
        self._layer_mix = layer_mix
        self._layer_mix_stride = max(1, layer_mix_stride)
        self._layer_mix_checked = False
        self.lora_module_names = []
        hf_id = spec["hf_id"]
        full_model = AutoModelForSpeechSeq2Seq.from_pretrained(hf_id, trust_remote_code=True)
        # fp32 master weights regardless of precision: autocast (bf16/fp16) handles compute
        # precision, and this avoids a bf16-checkpoint-weight / fp32-input BatchNorm dtype clash.
        full_model = full_model.float()
        self.encoder = full_model.get_encoder()
        config = self.encoder.config
        hidden_size = getattr(config, "d_model", None) or config.hidden_size
        self.pool = pool
        self._mask_kwarg = _resolve_mask_kwarg(self.encoder)
        self._warned_no_mask = False
        print(f"  encoder mask kwarg resolved to: {self._mask_kwarg!r}")

        # torch.compile, train-forward only (see forward()) -- wrapped as a plain closure, NOT
        # assigned to an attribute torch would register as a submodule, so `self.encoder` stays
        # the one true copy of the parameters: build_param_groups's named_parameters() walk and
        # its `accounted == total_named` assert, plus trainable_report()'s counts, would silently
        # double up if an OptimizedModule alias of the encoder were registered alongside it.
        self._compile_mode = compile_mode
        self._compile_failed = False
        self._compiled_encoder_fn = None
        self._shape_debug_left = 0   # set >0 by --profile-steps to print shapes once
        if compile_encoder and device.type == "cuda" and hasattr(torch, "compile"):
            encoder_ref = self.encoder

            def _encoder_call(feats, **kw):
                return encoder_ref(feats, **kw)

            self._compiled_encoder_fn = torch.compile(_encoder_call, mode=self._compile_mode)
            print(f"  torch.compile enabled for the encoder (mode={self._compile_mode!r}); "
                  "train-only, eager fallback is automatic on first-forward failure.")

        # -- layer mixing -------------------------------------------------------------------
        # One scalar per layer, softmaxed at forward time. Initialised to zeros = a uniform mix,
        # so the model starts by averaging every layer rather than committing to the top one.
        n_enc_layers = len(_encoder_layers(self.encoder))
        self.layer_weights = None
        if layer_mix:
            if not _supports_hidden_states(self.encoder):
                raise RuntimeError(
                    "--layer-mix requires the encoder's forward to accept output_hidden_states, "
                    "and this encoder's signature does not expose it. Falling back to last-layer "
                    "pooling silently would make the flag a no-op that still reports as enabled, "
                    "so this is fatal. Run without --layer-mix."
                )
            # +1 covers the embedding/pre-encoder output that HF-style encoders prepend to
            # hidden_states; the real length is verified against this on the first forward.
            self.layer_weights = nn.Parameter(torch.zeros(n_enc_layers + 1))
            print(f"  --layer-mix: {n_enc_layers} encoder layers discovered, "
                  f"stride {self._layer_mix_stride}")

        # -- LoRA ---------------------------------------------------------------------------
        if lora:
            self.lora_module_names = apply_lora(
                self.encoder, list(lora_targets), r=lora_rank, alpha=lora_alpha,
                dropout=lora_dropout)
            if not self.lora_module_names:
                raise RuntimeError(
                    f"--lora matched ZERO nn.Linear modules with --lora-targets "
                    f"{list(lora_targets)}. The encoder would stay fully frozen and only the "
                    "classifier head would train -- a run that looks healthy and answers "
                    "nothing, so this is fatal. Inspect the names printed by "
                    "`--profile-steps 1` or pass a --lora-targets substring that matches this "
                    "architecture's projection naming."
                )

        if pool in ("mean_std", "attn_stat"):
            clf_in = hidden_size * 2
        else:
            clf_in = hidden_size
        if pool == "attn_stat":
            self.attn = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(clf_in, num_labels)

    def _is_lora_param(self, name):
        return "lora_A" in name or "lora_B" in name

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze for the main training phase.

        With --lora this unfreezes ONLY the adapter parameters and leaves the pretrained weights
        frozen -- that restriction is the entire point of the flag here (it is a regularizer
        against the 7-point train/val gap, not a memory optimization), so unfreezing everything
        would silently turn a LoRA run into a full fine-tune that merely has extra parameters.
        """
        if self.lora_module_names:
            n_lora = 0
            for name, p in self.encoder.named_parameters():
                is_lora = self._is_lora_param(name)
                p.requires_grad = is_lora
                n_lora += int(is_lora)
            print(f"  LoRA unfreeze: {n_lora} adapter tensors trainable, pretrained encoder "
                  "weights stay frozen")
        else:
            for p in self.encoder.parameters():
                p.requires_grad = True
        return len(_encoder_layers(self.encoder))

    def _derive_frame_mask(self, out, h, input_mask, input_lengths=None, padded_samples=None):
        T = h.shape[1]
        for attr in ("output_lengths", "encoder_out_lens", "lengths", "output_length"):
            lens = getattr(out, attr, None)
            if lens is not None:
                lens = lens.to(h.device)
                ar = torch.arange(T, device=h.device).unsqueeze(0)
                return ar < lens.unsqueeze(1)
        if input_mask is not None:
            Tin = input_mask.shape[1]
            if Tin == T:
                return input_mask.bool()
            m = input_mask.float().unsqueeze(1)                      # (B,1,Tin)
            m_ds = torch.nn.functional.interpolate(m, size=T, mode="nearest").squeeze(1)
            return m_ds.bool()
        if input_lengths is not None and (padded_samples is not None or CFG.get("crop")):
            # Fixed-length train padding (see collate) means every sample in the batch was
            # zero-padded up to the same raw sample count -- so with no attention_mask and no
            # output-length field, fall back to a proportional estimate: true_samples/padded_
            # samples maps linearly onto true_frames/T for a constant-frame-rate encoder. This
            # is the fallback that keeps root cause 5 (mean pooling over padding) fixed once
            # static-shape padding removes the processor's own attention_mask.
            #
            # The denominator comes from the BATCH (padded_samples), not from the global
            # CFG["crop"]: under --crop-set the padded length varies per batch, so using the
            # global would compute every non-default batch's valid-frame count against the wrong
            # total. That failure is silent -- pooling would just quietly average over the wrong
            # span -- so CFG["crop"] is only the fallback for the fixed-crop path.
            lens = input_lengths.to(h.device).float()
            if padded_samples is not None:
                # Kept as a tensor and divided on-device: reading it out with .item() would force
                # a CPU/GPU sync on every single step, which is exactly what the accumulator
                # comments in the train loop go out of their way to avoid.
                total = torch.as_tensor(padded_samples, device=h.device).reshape(-1)[0].float()
                total = total.clamp(min=1.0)
            else:
                total = torch.tensor(max(CFG["crop"] * SR, 1.0), device=h.device)
            frac = (lens / total).clamp(max=1.0)
            # .clamp(max=T) is what guarantees the mask can never exceed the encoder's output
            # length, so a wrong denominator degrades to an over-long mask rather than an
            # out-of-bounds index. The correctness check on the denominator itself is in the
            # collate (padded_samples travels with the batch) and in test_crop_mask below.
            valid = (frac * T).round().long().clamp(min=1, max=T)
            ar = torch.arange(T, device=h.device).unsqueeze(0)
            return ar < valid.unsqueeze(1)
        if not self._warned_no_mask:
            print("  WARNING: no attention mask / output-length field available from the Cohere "
                  "encoder for this batch -- pooling falls back to plain mean over all frames "
                  "(same behavior as cohere_bench.py). This warning prints once.")
            self._warned_no_mask = True
        return None

    def _pool(self, h, mask):
        if mask is None:
            mean = h.mean(dim=1)
            if self.pool == "mean":
                return mean
            std = h.std(dim=1)
            if self.pool == "mean_std":
                return torch.cat([mean, std], dim=-1)
            w = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
            wmean = (h * w).sum(1)
            wstd = torch.sqrt(((h - wmean.unsqueeze(1)) ** 2 * w).sum(1).clamp(min=1e-6))
            return torch.cat([wmean, wstd], dim=-1)

        m = mask.float().unsqueeze(-1)                                # (B,T,1)
        denom = m.sum(1).clamp(min=1.0)
        mean = (h * m).sum(1) / denom
        if self.pool == "mean":
            return mean
        if self.pool == "mean_std":
            var = ((h - mean.unsqueeze(1)) ** 2 * m).sum(1) / denom
            return torch.cat([mean, torch.sqrt(var.clamp(min=1e-6))], dim=-1)
        # attn_stat: attention weights masked to valid frames only
        logits = self.attn(h).squeeze(-1)
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        wmean = (h * w).sum(1)
        wstd = torch.sqrt(((h - wmean.unsqueeze(1)) ** 2 * w).sum(1).clamp(min=1e-6))
        return torch.cat([wmean, wstd], dim=-1)

    def _specaug(self, feats):
        """SpecAugment-style time/frequency masking, applied on the GPU to the already-extracted
        features. Costs no CPU at all, so it stacks with --aug instead of competing with it for
        the dataloader budget.

        The processor's feature layout isn't documented for this checkpoint, so the mel axis is
        inferred as the smaller of the two non-batch dims (mel bins are 80-128; frame counts for
        a >=3s crop are in the hundreds) and asserted, rather than assumed.
        """
        if feats.ndim != 3:
            return feats
        b, d1, d2 = feats.shape
        # --specaug-mel-bins pins the axis instead of guessing. The guess below is only correct
        # while frames > mel bins; with a short --crop-set that ordering flips per batch, so the
        # axis would silently change from batch to batch within one run.
        _pin = getattr(ARGS, "specaug_mel_bins", 0)
        if _pin:
            assert _pin in (d1, d2), (
                f"--specaug-mel-bins {_pin} matches neither dim of feature shape "
                f"{tuple(feats.shape)}. Check the shape printed by --profile-steps.")
            assert d1 != d2, (
                f"feature shape {tuple(feats.shape)} is square, so --specaug-mel-bins cannot "
                "disambiguate the axes.")
            mel_axis = 1 if d1 == _pin else 2
        else:
            mel_axis = 1 if d1 <= d2 else 2
        n_mel, n_frames = (d1, d2) if mel_axis == 1 else (d2, d1)
        if not self._specaug_checked:
            if not _pin and n_frames < 2 * n_mel:
                print(f"  --specaug: WARNING -- feature shape {tuple(feats.shape)} has frames "
                      f"({n_frames}) close to mel bins ({n_mel}), so the axis was GUESSED and may "
                      "be wrong.\n             Short crops invert this ordering. Pass "
                      "--specaug-mel-bins to pin it.")
            print(f"  --specaug: feature layout (B={b}, mel={n_mel}, frames={n_frames}), "
                  f"mel axis={mel_axis}")
            self._specaug_checked = True

        out = feats
        max_f = max(1, n_mel // 8)
        max_t = max(1, n_frames // 12)
        for _ in range(self._specaug_f):
            f = int(torch.randint(0, max_f + 1, (1,)).item())
            if f:
                f0 = int(torch.randint(0, max(1, n_mel - f), (1,)).item())
                idx = [slice(None)] * 3
                idx[mel_axis] = slice(f0, f0 + f)
                out = out.clone() if out is feats else out
                out[tuple(idx)] = 0.0
        for _ in range(self._specaug_t):
            t = int(torch.randint(0, max_t + 1, (1,)).item())
            if t:
                t0 = int(torch.randint(0, max(1, n_frames - t), (1,)).item())
                idx = [slice(None)] * 3
                idx[3 - mel_axis] = slice(t0, t0 + t)
                out = out.clone() if out is feats else out
                out[tuple(idx)] = 0.0
        return out

    def _mix_layers(self, out, h_last):
        """Softmax-weighted sum over encoder layer outputs, replacing last_hidden_state.

        Asserts on the first call rather than degrading quietly: if the encoder ignored
        output_hidden_states, `hidden_states` is absent or length 1 and the flag would be a no-op
        that still reports as enabled -- the single most likely way for this feature to waste a
        full training run.
        """
        hs = getattr(out, "hidden_states", None)
        if hs is None and isinstance(out, (tuple, list)) and len(out) > 1:
            hs = out[1] if isinstance(out[1], (tuple, list)) else None
        if not hs or len(hs) < 2:
            raise RuntimeError(
                "--layer-mix is on but the encoder returned "
                f"{0 if not hs else len(hs)} hidden state(s). It accepted output_hidden_states "
                "in its signature but did not populate them, so the weighted sum would silently "
                "reduce to last-layer pooling. Re-run without --layer-mix."
            )

        hs = list(hs)[::self._layer_mix_stride]
        # Always keep the final layer, whatever the stride leaves behind.
        if hs[-1] is not h_last and h_last is not None:
            hs.append(h_last)
        n = len(hs)

        if not self._layer_mix_checked:
            print(f"  --layer-mix: encoder returned {n} usable hidden states "
                  f"(stride {self._layer_mix_stride}), each {tuple(hs[-1].shape)}")
            w = self.layer_weights
            if w.numel() != n:
                # __init__ sizes this as n_layers+1, guessing that the encoder prepends an
                # embedding output the way HF encoders do. Correct it against reality here.
                #
                # This is confined to the FIRST forward on purpose. Rewriting .data changes the
                # Parameter's shape while the optimizer already holds a reference to it, and
                # Adam allocates exp_avg/exp_avg_sq lazily on the first step() -- so doing this
                # before any step is safe, and doing it after would leave those buffers at the
                # old shape and blow up on the next step. The first forward always precedes the
                # first step, in both train_one and the smoke test.
                with torch.no_grad():
                    keep = min(n, w.numel())
                    new = torch.zeros(n, device=w.device, dtype=w.dtype)
                    new[:keep] = w[:keep]
                    self.layer_weights.data = new
                print(f"  --layer-mix: resized layer weights {w.numel()} -> {n} to match the "
                      "encoder's actual hidden-state count")
            self._layer_mix_checked = True

        w = self.layer_weights
        assert w.numel() == n, (
            f"--layer-mix: layer weight count {w.numel()} != hidden-state count {n}. The "
            "hidden-state count changed after the first forward, which the one-shot resize "
            "above cannot handle safely once the optimizer has allocated state."
        )

        weights = torch.softmax(w, dim=0).to(hs[-1].dtype)
        return torch.stack(hs, dim=0).mul(weights.view(-1, 1, 1, 1)).sum(0)

    def layer_mix_weights(self):
        """Current softmax weights as a plain list, for logging. Empty when --layer-mix is off."""
        if self.layer_weights is None:
            return []
        with torch.no_grad():
            return torch.softmax(self.layer_weights.detach().float(), dim=0).cpu().tolist()

    def forward(self, input_features=None, attention_mask=None, input_lengths=None,
                padded_samples=None):
        if self._shape_debug_left > 0:
            print(f"  [profile] input_features.shape={tuple(input_features.shape)}")
        if self.training and self._specaug_on:
            input_features = self._specaug(input_features)
        kwargs = {}
        if self._mask_kwarg and attention_mask is not None:
            kwargs[self._mask_kwarg] = attention_mask
        if self._layer_mix:
            kwargs["output_hidden_states"] = True
        if self.training and self._compiled_encoder_fn is not None and not self._compile_failed:
            try:
                out = self._compiled_encoder_fn(input_features, **kwargs)
            except Exception as e:
                print(f"  torch.compile forward failed ({type(e).__name__}: {e}) -- reverting "
                      "to eager for the rest of this run.")
                self._compile_failed = True
                out = self.encoder(input_features, **kwargs)
        else:
            out = self.encoder(input_features, **kwargs)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self._layer_mix:
            h = self._mix_layers(out, h)
        if self._shape_debug_left > 0:
            print(f"  [profile] encoder output h.shape={tuple(h.shape)}  (T={h.shape[1]} frames "
                  "-- small T confirms the launch-overhead-bound GPU-util diagnosis; a T in the "
                  "thousands means that diagnosis needs revisiting)")
            self._shape_debug_left -= 1
        frame_mask = self._derive_frame_mask(out, h, attention_mask, input_lengths=input_lengths,
                                             padded_samples=padded_samples)
        pooled = self._pool(h, frame_mask)
        return self.classifier(pooled)


def set_bn_eval(module):
    """Freezes BatchNorm running-stat updates (affine weights stay trainable) -- see root cause 4
    in the module docstring: tiny-batch BN stats estimated on 4s crops otherwise get applied at
    eval time to clips up to 30s.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def trainable_report(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    print(f"  trainable {t:,} / {a:,} ({100*t/a:.2f}%)")
    return t


def _assert_aligned(logits, labels, context):
    if logits.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"[{context}] model returned {logits.shape[0]} rows for a batch of "
            f"{labels.shape[0]} labels -- the backbone likely split a long clip into multiple "
            "output rows instead of one pooled row per input sample. Lower "
            "MAX_AUDIO_SECONDS_COHERE further."
        )



# ============================================================================
# Layer-wise LR decay param groups + LR-lambda schedule -- ONE optimizer/scheduler for the
# entire run (this is the actual fix for the dead-schedule bug in cohere_bench.py).
# ============================================================================
def build_param_groups(model, args):
    head_params = list(model.classifier.parameters())
    if getattr(model, "attn", None) is not None:
        head_params += list(model.attn.parameters())
    # layer_weights is NOT inside self.encoder, so the encoder walk below will never see it. Left
    # out here it would belong to no optimizer group at all and would stay at its zero init for
    # the whole run -- a --layer-mix that reports as enabled but only ever computes a uniform
    # average. The assert further down only covers encoder params and would not catch it.
    if getattr(model, "layer_weights", None) is not None:
        head_params.append(model.layer_weights)

    encoder = model.encoder
    list_name, layer_list = _encoder_layer_list_with_name(encoder)
    n_layers = len(layer_list)
    prefix = (list_name + ".") if list_name else None

    pre, post = [], []
    lora_params = []
    layer_params = defaultdict(list)
    seen = False
    total_named = 0
    for name, p in encoder.named_parameters():
        total_named += 1
        # LoRA adapters get one flat group at --lora-lr: LLRD exists to protect PRETRAINED lower
        # layers from large updates, and adapters are freshly initialised at zero, so decaying
        # their LR by depth would just make the lower ones untrainable.
        is_lora = "lora_A" in name or "lora_B" in name
        in_stack = bool(prefix and name.startswith(prefix))
        if in_stack:
            # Set before the lora branch: a LoRA param can be the first in-stack name we see,
            # and if it did not flip `seen` the following non-stack params would be misfiled
            # into `pre` instead of `post`.
            seen = True
        if is_lora:
            lora_params.append(p)
        elif in_stack:
            idx = int(name[len(prefix):].split(".")[0])
            layer_params[idx].append(p)
        else:
            (post if seen else pre).append(p)

    accounted = (sum(len(v) for v in layer_params.values()) + len(pre) + len(post)
                 + len(lora_params))
    assert accounted == total_named, (
        f"param grouping mismatch: assigned {accounted} of {total_named} encoder parameters -- "
        "some encoder parameter would be silently left out of the optimizer entirely."
    )

    groups, idx_map = [], {}

    def add_group(name, params, lr, kind):
        w = [p for p in params if p.ndim >= 2]
        b = [p for p in params if p.ndim < 2]
        if w:
            idx_map[name] = len(groups)
            groups.append(dict(params=w, lr=lr, weight_decay=args.weight_decay,
                               name=name, kind=kind))
        if b:
            groups.append(dict(params=b, lr=lr, weight_decay=0.0, name=name + "_bias", kind=kind))

    add_group("head", head_params, args.head_lr, kind="head")
    add_group("enc_pre", pre, args.lr * (args.llrd ** n_layers), kind="enc")
    for i in range(n_layers):
        add_group(f"enc_layer_{i}", layer_params[i], args.lr * (args.llrd ** (n_layers - 1 - i)), kind="enc")
    add_group("enc_post", post, args.lr, kind="enc")
    if lora_params:
        # kind="enc" so the adapters follow the encoder's schedule -- zero while the encoder is
        # frozen, then the re-warmup at --frozen-steps. They are part of the encoder's update,
        # just a low-rank one, so ramping them in on the head's schedule instead would hit the
        # frozen backbone with full-strength adapter updates from step 0.
        add_group("lora", lora_params, args.lora_lr, kind="enc")

    info = dict(
        n_layers=n_layers,
        head_idx=idx_map.get("head"),
        top_idx=idx_map.get("lora", idx_map.get(f"enc_layer_{n_layers-1}", idx_map.get("enc_post"))),
        bottom_idx=idx_map.get("enc_layer_0", idx_map.get("enc_pre")),
        n_lora=len(lora_params),
    )
    return groups, info


def make_lr_lambdas(args, max_steps, frozen_steps, kinds):
    head_warmup = max(1, args.head_warmup)
    enc_warmup = max(1, args.enc_warmup)
    min_ratio = args.min_lr_ratio

    def cosine(progress):
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    def head_lambda(step):
        if step < head_warmup:
            return step / head_warmup
        return cosine((step - head_warmup) / max(1, max_steps - head_warmup))

    def enc_lambda(step):
        if step < frozen_steps:
            return 0.0
        s = step - frozen_steps
        if s < enc_warmup:
            return s / enc_warmup
        return cosine((s - enc_warmup) / max(1, max_steps - frozen_steps - enc_warmup))

    return [head_lambda if k == "head" else enc_lambda for k in kinds]


def _build_adamw(groups):
    """LLRD gives ~2 param groups per encoder layer (~50 total). Plain AdamW dispatches a
    separate multi-tensor launch per group per step -- fused collapses that into one kernel.
    Falls back to foreach, then the plain per-group path, for older torch/driver combos.
    """
    if device.type == "cuda":
        try:
            opt = AdamW(groups, betas=(0.9, 0.98), eps=1e-6, fused=True)
            print("  AdamW: fused=True")
            return opt
        except (RuntimeError, TypeError) as e:
            print(f"  fused AdamW unavailable ({type(e).__name__}: {e}) -- trying foreach")
        try:
            opt = AdamW(groups, betas=(0.9, 0.98), eps=1e-6, foreach=True)
            print("  AdamW: foreach=True")
            return opt
        except (RuntimeError, TypeError) as e:
            print(f"  foreach AdamW unavailable ({type(e).__name__}: {e}) -- falling back to "
                  "the per-group default")
    return AdamW(groups, betas=(0.9, 0.98), eps=1e-6)


def build_optimizer_and_schedule(model, args, max_steps, frozen_steps):
    groups, info = build_param_groups(model, args)
    opt = _build_adamw(groups)
    kinds = [g["kind"] for g in groups]
    lambdas = make_lr_lambdas(args, max_steps, frozen_steps, kinds)
    sched = LambdaLR(opt, lambdas)
    print(f"  optimizer: {len(groups)} param groups ({info['n_layers']} encoder layers + "
          f"pre/post/head), betas=(0.9,0.98) eps=1e-6")
    return opt, sched, info



# ============================================================================
# Feature extraction -- preserves the processor's attention mask (cohere_bench.py discarded it,
# then mean-pooled over the resulting padding).
# ============================================================================
MAX_AUDIO_SECONDS_COHERE = 30

def get_feature_extractor(spec):
    return AutoProcessor.from_pretrained(spec["hf_id"], trust_remote_code=True)


def _extract_features(fe, wavs):
    """Cohere's gated processor isn't documented to pin one output key name (input_features vs
    input_values) or one mask key name -- try the plausible options for both.
    """
    try:
        raw = fe(wavs, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    except TypeError:
        raw = fe(wavs, sampling_rate=TARGET_SR, return_tensors="pt")
    key = "input_features" if "input_features" in raw else "input_values"
    feats = raw[key]
    mask = None
    for mk in ("attention_mask", "input_features_mask", "feature_attention_mask"):
        if mk in raw:
            mask = raw[mk]
            break
    return feats, mask



# ============================================================================
# Result persistence -- identical durability primitive to cohere_bench.py.
# ============================================================================
def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, dict):
            out[k] = _jsonable(v)
        else:
            out[k] = v
    return out


def _load_jsonl(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def already_done(results_path):
    return {(d.get("run"), d.get("model")) for d in _load_jsonl(results_path)
            if d.get("status") == "ok"}


def _append_progress(rec):
    with open(ARGS.progress, "a") as f:
        f.write(json.dumps(_jsonable(rec)) + "\n")
        f.flush()
        os.fsync(f.fileno())


RESULT_COLS = ["run","model","status","seed","id_acc","id_acc_macro","id_cavg",
               "casa_acc","casa_acc8","casa_acc_holdout","casa_acc8_holdout",
               "casa_acc_holdout_tta","casa_acc8_holdout_tta","id_acc_tta",
               "casa_cavg20","casa_cavg20_holdout",
               "casa_cavg8","off_set","gap","gap_holdout","madis_overall",
               # ADI17 test -- report-only, never a selection metric. adi17_off_set is the share
               # predicted into MSA/BAH/TUN, i.e. a direct read on whether adding a 17-class
               # training source over-shrank those three priors.
               "adi17_acc","adi17_acc17","adi17_cavg20","adi17_cavg17","adi17_off_set",
               "lr","head_lr","llrd",
               "frozen_steps","precision","pool","batch_size","grad_accum","effective_batch",
               "clip_norm","label_smoothing","bn_eval","crop","crop_set","aug","specaug",
               "layer_mix","lora","lora_rank","lora_modules","weight_decay","tta",
               # What this row was actually trained on, and what it was selected on. Both belong
               # in the table: with three optional sources the flags alone stop being readable.
               "datasets","dataset_rows","sampler_alpha","select_metric",
               "extra_train_data","extra_train_rows","error"]


def _rewrite_table():
    rows = _load_jsonl(ARGS.results)
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = [c for c in RESULT_COLS if c in df]
    out = df[cols]
    if "casa_acc" in out:
        out = out.sort_values("casa_acc", ascending=False)
    out.to_csv("train_table.csv", index=False)


def print_running_table():
    rows = _load_jsonl(ARGS.results)
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = [c for c in ["model", "status", "id_acc", "casa_acc", "casa_acc_holdout",
                        "casa_cavg20", "madis_overall", "off_set", "gap"] if c in df]
    print("\n" + "-" * 66)
    print("  running results so far:")
    print(df[cols].round(4).to_string(index=False))
    print("-" * 66)


def record_result(rec):
    with open(ARGS.results, "a") as f:
        f.write(json.dumps(_jsonable(rec)) + "\n")
        f.flush()
        os.fsync(f.fileno())
    _rewrite_table()
    if rec.get("status") == "ok":
        print(f"\n  >> recorded {rec.get('model')}: casa_acc={rec.get('casa_acc', float('nan')):.2f} "
              f"casa_cavg20={rec.get('casa_cavg20', float('nan')):.4f} "
              f"madis_overall={rec.get('madis_overall', float('nan')):.2f} "
              f"id_acc={rec.get('id_acc', float('nan')):.2f}")
    else:
        print(f"\n  >> recorded {rec.get('model')}: status={rec.get('status')} "
              f"({rec.get('error', '')})")
    print_running_table()


def _close_dangling_wandb():
    if USE_WANDB:
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=1)
        except Exception:
            pass


def _wb_try(label, fn):
    try:
        fn()
        return True
    except Exception as e:
        print(f"  W&B {label} failed ({type(e).__name__}: {e}) -- continuing without it; "
              "training and --results/--progress are unaffected.")
        return False



# ============================================================================
# Smoke test -- runs BEFORE any dataset download, on synthetic audio only.
# ============================================================================
def _synthetic_samples():
    rng = np.random.default_rng(0)
    return [
        {"audio": {"array": rng.standard_normal(3 * TARGET_SR).astype(np.float32),
                   "sampling_rate": TARGET_SR}, "dialect": COUNTRIES[0]},
        {"audio": {"array": rng.standard_normal(5 * TARGET_SR).astype(np.float32),
                   "sampling_rate": TARGET_SR}, "dialect": COUNTRIES[1]},
    ]


def _smoke_collate(fe, samples):
    wavs, labs = [], []
    for s in samples:
        w = torch.as_tensor(s["audio"]["array"], dtype=torch.float32)
        max_samples = int(MAX_AUDIO_SECONDS_COHERE * TARGET_SR)
        w = w[..., :max_samples]
        wavs.append(w.numpy())
        labs.append(labels2id[s["dialect"]])
    labels = torch.tensor(labs)
    feats, mask = _extract_features(fe, wavs)
    f = {"input_features": feats}
    if mask is not None:
        f["attention_mask"] = mask
    return f, labels


def _autocast_ctx(precision):
    if precision == "bf16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.amp.autocast("cuda", enabled=False)


def _smoke_run(model_name, spec, feats, labels, precision):
    model = DialectID(spec, num_labels=20, pool=ARGS.pool, specaug=ARGS.specaug,
                       specaug_f=ARGS.specaug_freq_mask,
                       specaug_t=ARGS.specaug_time_mask,
                       layer_mix=ARGS.layer_mix, layer_mix_stride=ARGS.layer_mix_stride,
                       lora=ARGS.lora, lora_rank=ARGS.lora_rank,
                       lora_alpha=ARGS.lora_alpha, lora_dropout=ARGS.lora_dropout,
                       lora_targets=ARGS.lora_target_list).to(device)
    model.freeze_encoder()
    if ARGS.bn_eval:
        set_bn_eval(model)
    t0 = trainable_report(model)

    opt, sched, info = build_optimizer_and_schedule(model, ARGS, max_steps=10, frozen_steps=1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=ARGS.label_smoothing)

    # --layer-mix fails SILENTLY if layer_weights ends up in no optimizer group: the softmax of
    # an all-zero vector is a uniform average, which trains, converges, and looks entirely normal
    # while the flag does nothing. Catch it here, in seconds, rather than after a full run.
    if ARGS.layer_mix:
        assert model.layer_weights is not None, "--layer-mix on but layer_weights was never created"
        in_opt = any(any(p is model.layer_weights for p in g["params"]) for g in opt.param_groups)
        assert in_opt, (
            "--layer-mix: layer_weights is not in ANY optimizer param group, so it would stay at "
            "its zero init for the whole run and the weighted sum would be a plain uniform "
            "average. Check build_param_groups."
        )
        print(f"  --layer-mix: layer_weights ({model.layer_weights.numel()} params) confirmed in "
              "the optimizer")
    use_fp16 = precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    f = {k: v.to(device) for k, v in feats.items()}
    y = labels.to(device)

    model.train()
    if ARGS.bn_eval:
        set_bn_eval(model)
    with _autocast_ctx(precision):
        logits = model(**f)
        _assert_aligned(logits, y, f"{model_name} smoke frozen-step")
        loss = loss_fn(logits, y)
    opt.zero_grad()
    scaler.scale(loss).backward()
    if use_fp16:
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                    ARGS.clip_norm, error_if_nonfinite=False)
    scaler.step(opt); scaler.update(); sched.step()

    n_layers = model.unfreeze_encoder()
    assert n_layers > 0, (
        f"{model_name}: encoder has no discoverable layer list -- unfreeze_encoder is a no-op."
    )
    t1 = trainable_report(model)
    if ARGS.lora:
        # With --lora the pretrained weights stay frozen ON PURPOSE, so the full-encoder growth
        # check below does not apply. What must hold instead: adapters exist, they are trainable,
        # and the frozen backbone really is frozen -- a LoRA run that quietly unfroze everything
        # would be a full fine-tune wearing a LoRA label, and would answer the wrong question.
        n_lora_params = sum(p.numel() for n, p in model.encoder.named_parameters()
                            if model._is_lora_param(n))
        n_lora_trainable = sum(p.numel() for n, p in model.encoder.named_parameters()
                               if model._is_lora_param(n) and p.requires_grad)
        n_base_trainable = sum(p.numel() for n, p in model.encoder.named_parameters()
                               if not model._is_lora_param(n) and p.requires_grad)
        assert model.lora_module_names, "--lora on but no modules were wrapped"
        assert n_lora_params > 0 and n_lora_trainable == n_lora_params, (
            f"--lora: {n_lora_trainable:,} of {n_lora_params:,} adapter params are trainable -- "
            "expected all of them."
        )
        assert n_base_trainable == 0, (
            f"--lora: {n_base_trainable:,} PRETRAINED encoder params are still trainable. This "
            "run would be a full fine-tune with extra parameters, not a LoRA run."
        )
        print(f"  LoRA: {len(model.lora_module_names)} wrapped module(s), "
              f"{n_lora_params:,} adapter params trainable, backbone frozen")
    else:
        assert t1 - t0 > 1_000_000, (
            f"{model_name}: trainable params grew by only {t1 - t0:,} after unfreeze_encoder() -- "
            "encoder is effectively still frozen (layer discovery likely failed)."
        )
    print(f"  unfroze: {n_layers} encoder layers now trainable")

    lw_before = (model.layer_weights.detach().clone() if ARGS.layer_mix else None)

    with _autocast_ctx(precision):
        logits = model(**f)
        _assert_aligned(logits, y, f"{model_name} smoke unfrozen-step")
        loss = loss_fn(logits, y)
    opt.zero_grad()
    scaler.scale(loss).backward()
    if ARGS.layer_mix:
        assert model.layer_weights.grad is not None, (
            "--layer-mix: layer_weights received NO gradient from the backward pass, so the "
            "mixing weights are disconnected from the loss -- the encoder's hidden_states are "
            "probably being detached or the mix is not on the path to the classifier."
        )
    if use_fp16:
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                    ARGS.clip_norm, error_if_nonfinite=False)
    scaler.step(opt); scaler.update(); sched.step()

    if ARGS.layer_mix:
        moved = (model.layer_weights.detach() - lw_before).abs().max().item()
        assert moved > 0, (
            "--layer-mix: layer_weights did not change after an optimizer step, so they are not "
            "actually being optimized despite being in a param group."
        )
        print(f"  --layer-mix: weights moved {moved:.2e} on one step -- optimization confirmed")

    model.eval()
    with torch.no_grad():
        logits = model(**f)
    assert logits.shape == (len(labels), 20), f"unexpected logits shape {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "non-finite logits in eval forward"

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  peak VRAM: {peak:.2f} GB")

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()


PRECISION_OVERRIDE = {"cohere-ar": ARGS.precision}

def smoke_test(model_names):
    print("\n" + "=" * 66)
    print("  SMOKE TEST (synthetic audio, no dataset download yet)")
    print("=" * 66)
    samples = _synthetic_samples()
    failures = {}

    for model_name in [m for m in BENCH_ORDER if m in model_names]:
        spec = MODEL_REGISTRY[model_name]
        print(f"\n-- smoke testing {model_name} ({spec['hf_id']}) --")
        try:
            fe = get_feature_extractor(spec)
            feats, labels = _smoke_collate(fe, samples)
            assert torch.isfinite(feats["input_features"]).all(), \
                "non-finite features out of the collate/feature-extractor"

            precision = PRECISION_OVERRIDE.get(model_name, "bf16")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                _smoke_run(model_name, spec, feats, labels, precision)
            except RuntimeError as e:
                msg = str(e)
                dtype_error = any(k in msg for k in ("BFloat16", "Half", "dtype", "Float"))
                if precision != "fp32" and dtype_error:
                    print(f"  {precision} autocast failed ({msg[:160]}) -- retrying {model_name} "
                          "with fp32 + TF32 instead")
                    PRECISION_OVERRIDE[model_name] = "fp32"
                    gc.collect(); torch.cuda.empty_cache()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    _smoke_run(model_name, spec, feats, labels, "fp32")
                    print(f"  ** {model_name} will train in fp32 (TF32) for the rest of this "
                          "run -- this is expected, not a failure. **")
                else:
                    raise

            del fe
            gc.collect(); torch.cuda.empty_cache()
            print(f"{model_name}: OK")
        except Exception as e:
            status = "oom" if isinstance(e, torch.cuda.OutOfMemoryError) else "smoke_failed"
            print(f"\nFAIL {model_name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures[model_name] = (status, f"{type(e).__name__}: {e}")
            gc.collect(); torch.cuda.empty_cache()

    if failures:
        print("\n" + "=" * 66)
        print(f"  SMOKE TEST: {len(failures)} model(s) failed -- {list(failures)}")
        print("=" * 66)
        for model_name, (status, err) in failures.items():
            record_result(dict(run=CFG["run"], model=model_name, status=status, error=err))

    passed = [m for m in model_names if m not in failures]
    print(f"\nSMOKE TEST done -- {len(passed)}/{len(model_names)} model(s) passed: {passed}\n")
    return failures


smoke_failures = {} if ARGS.skip_smoke else smoke_test(list(MODEL_REGISTRY))
if ARGS.skip_smoke:
    print("\n(--skip-smoke set: skipping the pre-download smoke test -- not recommended)\n")

if smoke_failures:
    print("FAIL: cohere-ar failed the smoke test -- nothing left to train (single-model script).")
    sys.exit(1)

if ARGS.smoke_only:
    print("--smoke-only set: exiting after the smoke test.")
    sys.exit(0)



# ============================================================================
# OOD loaders (identical to cohere_bench.py, copied so this script runs standalone)
# ============================================================================
CASA_CONFIGS = {"Algeria":"ALG","Egypt":"EGY","Jordan":"JOR","Mauritania":"MAU",
                "Morocco":"MOR","Palestine":"PAL","UAE":"UAE","Yemen":"YEM"}
CASA_PRESENT = sorted(CASA_CONFIGS.values())
CASA_ABSENT  = [c for c in COUNTRIES if c not in CASA_PRESENT]

ADI20_TO_REGION = {
    "EGY":"Egyptian Arabic","SUD":"Egyptian Arabic",
    "KSA":"Gulf Arabic","UAE":"Gulf Arabic","QAT":"Gulf Arabic","KUW":"Gulf Arabic",
    "BAH":"Gulf Arabic","OMA":"Gulf Arabic","YEM":"Gulf Arabic","IRA":"Gulf Arabic",
    "SYR":"Levantine Arabic","LEB":"Levantine Arabic",
    "JOR":"Levantine Arabic","PAL":"Levantine Arabic",
    "MOR":"Maghrebi Arabic","ALG":"Maghrebi Arabic","TUN":"Maghrebi Arabic",
    "LIB":"Maghrebi Arabic","MAU":"Maghrebi Arabic",
    "MSA":"Modern Standard Arabic",
}
REGIONS   = sorted(set(ADI20_TO_REGION.values()))
REGION2ID = {r:i for i,r in enumerate(REGIONS)}
ID2REGION = torch.tensor([REGION2ID[ADI20_TO_REGION[c]] for c in COUNTRIES])

def stratified_subset(dataset, fraction=1.0, seed=SEED, label_col="dialect"):
    if fraction >= 1.0:
        return dataset
    labels = dataset[label_col]
    by_class = defaultdict(list)
    for i, l in enumerate(labels):
        by_class[l].append(i)
    rng = np.random.default_rng(seed)
    keep = []
    for l, idxs in by_class.items():
        n = max(1, round(len(idxs)*fraction))
        keep.extend(rng.choice(idxs, size=n, replace=False).tolist())
    return dataset.select(sorted(keep))

def _subsample(d, frac, seed=SEED):
    if d is None or frac >= 1.0:
        return d
    rng = np.random.default_rng(seed)
    n = max(1, int(len(d)*frac))
    return d.select(sorted(rng.choice(len(d), size=n, replace=False).tolist()))

def load_casablanca(seed=SEED, val_frac=0.5, select_frac=0.5, want_train=False):
    """Returns (select_ds, holdout_ds), split PROGRAM-disjointly.

    The baseline both picked the best checkpoint on casa_acc and reported casa_acc from the same
    clips, which biases the headline number upward -- with 10 eval points and a ~0.5pt spread,
    "best" is largely selecting on noise, and that noise then goes into the result. Splitting the
    kept programs once more gives an untouched half to report. select_frac=1.0 restores the old
    (biased) single-set behavior, with holdout_ds=None.

    The split is by PROGRAM, not by clip, for the same reason val_frac already is: clips from one
    broadcast share speakers and channel, so a clip-level split would leak.
    """
    parts = []
    for cfg, code_ in CASA_CONFIGS.items():
        try:
            d = load_dataset("UBC-NLP/Casablanca", cfg)
        except Exception as e:
            print(f"  [{cfg}] load failed -- {type(e).__name__}")
            continue
        for split in [s for s in ("validation","test") if s in d]:
            x = d[split].add_column("dialect", [code_]*len(d[split]))
            keep = [c for c in ("audio","dialect","seg_id") if c in x.column_names]
            parts.append(x.select_columns(keep))
    if not parts:
        print("  Casablanca: NOTHING LOADED")
        return None, None
    merged = concatenate_datasets(parts)
    segs = merged["seg_id"] if "seg_id" in merged.column_names else [str(i) for i in range(len(merged))]
    prog = [str(s).split("_")[0] for s in segs]
    # Kept as a real column, not just a local, so --confusion-by-program can attribute each eval
    # error to the broadcast it came from. It rides the collate's domain_col channel, which is
    # what keeps it aligned with the logits when undecodable clips are dropped.
    merged = merged.add_column("program", prog)
    rng = np.random.default_rng(seed)
    by_country = {}
    for i,(p,l) in enumerate(zip(prog, merged["dialect"])):
        by_country.setdefault(l, {}).setdefault(p, []).append(i)
    sel_idx, hold_idx, train_idx = [], [], []
    for country, progs in by_country.items():
        plist = sorted(progs); rng.shuffle(plist)
        n_kept = max(1, int(len(plist) * val_frac))
        kept = plist[:n_kept]
        # Everything past n_kept was previously DISCARDED -- roughly half the corpus, never
        # evaluated on and never trained on. --train-on-casa recovers it. Consuming no randomness
        # here is deliberate: the rng call sequence is identical to before, so `sel` and `hold`
        # come out bit-identical whether or not the flag is set, and every earlier casa_* number
        # stays directly comparable.
        for _p in plist[n_kept:]:
            train_idx.extend(progs[_p])
        # At least one program stays on each side whenever the country has >=2 to give.
        n_sel = max(1, int(round(len(kept) * select_frac)))
        if select_frac < 1.0 and len(kept) >= 2:
            n_sel = min(n_sel, len(kept) - 1)
        for p in kept[:n_sel]:
            sel_idx.extend(progs[p])
        for p in kept[n_sel:]:
            hold_idx.extend(progs[p])
    sel = merged.select(sorted(sel_idx)) if sel_idx else None
    hold = merged.select(sorted(hold_idx)) if hold_idx else None
    train = merged.select(sorted(train_idx)) if (want_train and train_idx) else None
    if want_train:
        _sp = set(sel["program"]) if sel is not None else set()
        _hp = set(hold["program"]) if hold is not None else set()
        _tp = set(train["program"]) if train is not None else set()
        _bad = _tp & (_sp | _hp)
        assert not _bad, ("Casablanca train programs overlap an eval half -- the split is broken "
                          f"and every casa_* number would be contaminated. Overlap: {sorted(_bad)[:5]}")
        print(f"  Casablanca TRAIN (recovered from the discarded half): "
              f"{0 if train is None else len(train)} clips over {len(_tp)} program(s), "
              f"program-disjoint from both eval halves (asserted)")
    print(f"  Casablanca: select {0 if sel is None else len(sel)} clips "
          f"({0 if sel is None else len(set(sel['dialect']))} countries) | "
          f"held out {0 if hold is None else len(hold)} clips "
          f"({0 if hold is None else len(set(hold['dialect']))} countries)")
    if hold is None:
        print("       (no held-out split -- selection and reporting use the SAME clips, which "
              "is optimistically biased; raise --casa-select-frac below 1.0 to separate them)")
    return (train, sel, hold) if want_train else (sel, hold)

def load_madis5(drop_tv_dramas=True):
    try:
        d = load_dataset("badrex/MADIS5-spoken-arabic-dialects")["test"]
    except Exception as e:
        print(f"  MADIS-5 load failed -- {type(e).__name__}")
        return None
    if drop_tv_dramas and "domain" in d.column_names:
        n0 = len(d)
        d = d.filter(lambda x: x["domain"] != "TV Dramas")
        print(f"  MADIS-5: dropped {n0-len(d)} TV Dramas clips")
    print(f"  MADIS-5: {len(d)} clips | domains {sorted(set(d['domain']))}")
    return d


def _stream_tiny_subset(repo_id, split, n_per_class, label_col="dialect", seed=SEED):
    ds = load_dataset(repo_id, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=200)
    counts = defaultdict(int)
    rows = []
    cap = n_per_class * 60
    for ex in ds:
        lab = ex.get(label_col)
        if lab is None or counts[lab] >= n_per_class:
            continue
        rows.append(ex)
        counts[lab] += 1
        if len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError(f"{repo_id}[{split}]: streamed 0 usable rows for --quick-data "
                            f"(label_col={label_col!r} -- does this split have that column?)")
    print(f"  [--quick-data] streamed {repo_id}[{split}]: {len(rows)} rows, "
          f"{len(counts)} distinct {label_col} value(s), {dict(counts)}")
    return Dataset.from_list(rows)


# ============================================================================
# ADI17 -- targeted subset of the 260 GB train split via parquet row-group selection
# ============================================================================
def _adi17_index_row_groups():
    """Map every train row group to its dialect, reading METADATA ONLY.

    The train split is 40 parquet files of ~6.5 GB and is sorted by dialect, so the naive
    approaches both fail: streaming sequentially would pull most of 260 GB before the last
    classes fill their per-class quota, and downloading whole files to sample inside them costs
    100+ GB.

    Parquet stores per-row-group column statistics in the footer. Because the split is
    dialect-sorted, a row group's dialect min == its max for all but the handful that straddle a
    boundary, so the entire (file, row_group) -> dialect map comes out of the footers with no
    column data read at all. Row groups whose stats are missing or non-degenerate fall back to
    reading just the `dialect` column for that group, which is a few KB (dictionary-encoded).

    Returns [(file_path, row_group_index, n_rows, byte_size, dialect_or_None), ...].
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    root = f"datasets/{ADI17_REPO}"
    split = ARGS.adi17_split
    # Several layouts are in circulation for natively-parquet repos (data/train/*.parquet,
    # train/*.parquet, */train-00000-of-*.parquet), and the repo can be re-laid-out without
    # notice. Try the specific patterns first, then fall back to filtering every parquet by path.
    files = []
    for pat in (f"{root}/**/{split}-*.parquet",     # the current layout: data/train-00000-of-N
                f"{root}/**/{split}/*.parquet",
                f"{root}/{split}/*.parquet"):
        files = sorted(fs.glob(pat))
        if files:
            break
    if not files:
        files = sorted(f for f in fs.glob(f"{root}/**/*.parquet")
                       if f"/{split}/" in f or f"/{split}-" in f)
    if not files:
        raise RuntimeError(
            f"no parquet files found for {ADI17_REPO}[{split}] under {root}. "
            "The repo layout may have changed; check https://huggingface.co/datasets/"
            f"{ADI17_REPO}/tree/main")
    # Measured against the real repo: the footer parse is ~3.1 s per file and the statistics scan
    # is ~0.01 s for all 250 row groups in it, so the whole 40-file train index costs ~2 minutes
    # and reads no audio at all. Announced with a time estimate because two silent minutes before
    # a download starts is indistinguishable from a hang.
    print(f"  indexing {len(files)} parquet file(s) from footers (no audio bytes read) -- "
          f"~{len(files) * 3.1 / 60:.0f} min")

    groups, n_from_stats, n_from_read, n_straddle = [], 0, 0, 0
    for _fi, path in enumerate(files, 1):
        if _fi == 1 or _fi % 10 == 0 or _fi == len(files):
            print(f"    [{_fi}/{len(files)}] {os.path.basename(path)}", flush=True)
        # cache_type="none" is load-bearing, not a tuning knob. HfFileSystem returns an
        # AbstractBufferedFile with readahead caching ON by default, and a parquet footer parse is
        # a handful of tiny scattered reads -- each one pulling a full readahead block. Across
        # ~250 row groups x 40 files that turns a few MB of intended reads into tens of GB of
        # transfer, which is the opposite of what "reads no audio at all" above promises.
        with fs.open(path, "rb", cache_type="none") as fh:
            pf = pq.ParquetFile(fh)
            if "dialect" not in pf.schema_arrow.names:
                raise RuntimeError(
                    f"{path} has columns {pf.schema_arrow.names}; expected a 'dialect'.")
            # The PARQUET leaf index, not the Arrow field index. Verified against the real repo:
            # arrow columns are ['id', 'audio', 'dialect'] (dialect at 2) while the parquet leaves
            # are ['id', 'audio.bytes', 'audio.path', 'dialect'] (dialect at 3), because the audio
            # struct expands to two physical columns. Using the arrow index would read statistics
            # for `audio.path` -- which also has a min and a max, both strings -- so every row
            # group would be mislabelled with no error raised anywhere.
            _leaves = [pf.metadata.schema.column(i).path
                       for i in range(pf.metadata.num_columns)]
            try:
                col_i = _leaves.index("dialect")
            except ValueError:
                raise RuntimeError(f"{path}: no leaf column named 'dialect' in {_leaves}")
            for g in range(pf.metadata.num_row_groups):
                rg = pf.metadata.row_group(g)
                lab, need_read = None, False
                try:
                    st = rg.column(col_i).statistics
                    if st is None or not st.has_min_max:
                        need_read = True            # no statistics at all -- must look
                    elif st.min == st.max:
                        lab = st.min
                        if isinstance(lab, bytes):
                            lab = lab.decode()
                        n_from_stats += 1
                    else:
                        # min != max already proves the group spans a dialect boundary, and
                        # straddling groups are DISCARDED below to keep the label guarantee
                        # exact. Reading one costs ~3.5 s -- as much as a whole footer parse --
                        # so reading it just to confirm what the statistics already said would
                        # add ~a minute across the split for data that is then thrown away.
                        n_straddle += 1
                except Exception:
                    need_read = True
                if need_read:
                    vals = set(pf.read_row_group(g, columns=["dialect"])
                                 .column("dialect").to_pylist())
                    n_from_read += 1
                    lab = vals.pop() if len(vals) == 1 else None
                groups.append((path, g, rg.num_rows, rg.total_byte_size, lab))
    print(f"  {len(groups)} row group(s): {n_from_stats} identified from statistics, "
          f"{n_straddle} span a dialect boundary, {n_from_read} needed a column read")
    if n_from_read > len(files):
        print("    NOTE: many row groups lacked usable statistics, so this index cost real "
              "reads.\n    That is a change in how the repo was written, not an error.")
    return groups


def load_adi17_subset(seed=SEED):
    """A per-dialect subset of ADI17's train split, cached to --adi17-cache-dir.

    Only ever called for --adi17-split train; dev and test are refused at argument-parse time
    because ADI17 dev IS the NADI ADI20 validation set.
    """
    cache = os.path.abspath(ARGS.adi17_cache_dir)
    alloc = ARGS.adi17_alloc_resolved

    if os.path.isdir(cache):
        try:
            ds = load_from_disk(cache)
            counts = Counter(ds["dialect"])
            print(f"  ADI17: cache hit at {cache} -- {len(ds)} clips, {len(counts)} dialect(s)")
            if dict(counts) != {k: v for k, v in alloc.items() if v}:
                print("    NOTE: the cache does not match the CURRENT allocation. It is used "
                      "as-is;\n          delete the directory to refetch under the new budget.")
            return ds
        except Exception as e:
            print(f"  ADI17: cache at {cache} is unreadable ({type(e).__name__}) -- refetching.")

    print(f"\nADI17 subset from {ADI17_REPO}[{ARGS.adi17_split}] "
          f"(requested {sum(alloc.values())} clips across {len(alloc)} dialects)")
    groups = _adi17_index_row_groups()

    by_dialect = defaultdict(list)
    straddling = 0
    for g in groups:
        if g[4] is None:
            straddling += 1
        elif g[4] in alloc:
            by_dialect[g[4]].append(g)
    unknown = {g[4] for g in groups} - set(ADI17_COUNTRIES) - {None}
    if unknown:
        raise RuntimeError(
            f"{ADI17_REPO} contains dialect code(s) {sorted(unknown)} that are not in COUNTRIES. "
            "The label vocabulary has changed and the mapping assumption in ADI17_COUNTRIES no "
            "longer holds -- fix that before training on this data.")
    if straddling:
        print(f"  {straddling} row group(s) span a dialect boundary and were skipped (they are "
              "a\n  negligible fraction and skipping keeps the label guarantee exact)")

    # Availability, before selecting anything: a dialect with fewer rows than its budget must say
    # so rather than quietly under-delivering. ADI17's dev counts suggest ALG/MAU/SUD are its
    # smallest classes, and ALG is one of the ones we most want.
    print(f"\n  {'dialect':<9s} {'available':>10s} {'requested':>10s} {'taking':>8s}")
    plan, short = {}, []
    for c in ADI17_COUNTRIES:
        avail = sum(g[2] for g in by_dialect.get(c, []))
        want = alloc[c]
        take = min(want, avail)
        plan[c] = take
        if take < want:
            short.append((c, want, avail))
        print(f"  {c:<9s} {avail:>10d} {want:>10d} {take:>8d}"
              + ("   << SHORT" if take < want else ""))
    if short:
        print(f"\n  WARNING: {len(short)} dialect(s) have less data than requested: "
              + ", ".join(f"{c} {a}<{w}" for c, w, a in short))
        print("  Taking everything available for those. The allocation table assumed ADI17 could "
              "cover\n  the budget; where it cannot, that budget is simply lost, NOT redistributed"
              " -- silently\n  moving it to another dialect would change the experiment you think "
              "you are running.")
    if not any(plan.values()):
        raise RuntimeError(f"{ADI17_REPO}[{ARGS.adi17_split}]: selected 0 clips.")

    # Pick row groups spread EVENLY across each dialect's range rather than taking a contiguous
    # head. Rows that sit next to each other in ADI17 come from the same YouTube channel, and
    # often the same speaker, so a contiguous block buys far less diversity per clip.
    rng = np.random.default_rng(seed)
    chosen, est_bytes = [], 0
    for c in ADI17_COUNTRIES:
        need, gs = plan[c], sorted(by_dialect.get(c, []), key=lambda g: (g[0], g[1]))
        if need <= 0 or not gs:
            continue
        mean_rows = max(1, sum(g[2] for g in gs) // len(gs))
        n_groups = min(len(gs), max(1, -(-need // mean_rows)))
        idx = sorted(set(np.linspace(0, len(gs) - 1, n_groups).round().astype(int).tolist()))
        # linspace can collide after rounding; top up with unused groups until the quota is met.
        got = sum(gs[i][2] for i in idx)
        if got < need:
            spare = [i for i in range(len(gs)) if i not in set(idx)]
            rng.shuffle(spare)
            for i in spare:
                idx.append(i)
                got += gs[i][2]
                if got >= need:
                    break
        for i in sorted(set(idx)):
            chosen.append((gs[i], c))
            est_bytes += gs[i][3]

    est_gb = est_bytes / 1e9
    print(f"\n  selected {len(chosen)} row group(s), ~{est_gb:.1f} GB to download "
          f"(budget --adi17-max-gb {ARGS.adi17_max_gb})")
    if est_gb > ARGS.adi17_max_gb:
        print(f"\nFAIL: the selection projects to {est_gb:.1f} GB, over the {ARGS.adi17_max_gb} GB "
              "budget.")
        print("  Row groups are the fetch quantum, so a small --adi17-alloc can still pull a lot "
              "if the\n  groups are large. Either lower the allocation or raise --adi17-max-gb "
              "deliberately.")
        sys.exit(2)
    _big = max((g[0][3] for g in chosen), default=0) / 1e9
    if _big > 1.0:
        print(f"  NOTE: the largest single row group is {_big:.1f} GB. That is the smallest unit "
              "that can be\n  fetched, so per-dialect counts round up in units of roughly that "
              "much data.")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()

    # Group by file so each parquet footer is parsed once instead of once per row group.
    by_file = defaultdict(list)
    for (path, g, _n, _b, _lab), c in chosen:
        by_file[path].append((g, c))

    tables, got = [], Counter()
    for fi, (path, wanted) in enumerate(sorted(by_file.items()), 1):
        print(f"  [{fi}/{len(by_file)}] {os.path.basename(path)}: {len(wanted)} row group(s)",
              flush=True)
        # Same readahead trap as the footer scan above -- see the comment there. The reads here
        # are large contiguous column chunks, so passing them through unbuffered is also the
        # efficient shape; PyArrow already coalesces what it asks for.
        with fs.open(path, "rb", cache_type="none") as fh:
            pf = pq.ParquetFile(fh)
            for g, c in sorted(wanted):
                if got[c] >= plan[c]:
                    continue           # an earlier group already covered this dialect's quota
                t = pf.read_row_group(g, columns=["audio", "dialect"])
                room = plan[c] - got[c]
                if t.num_rows > room:
                    t = t.slice(0, room)
                got[c] += t.num_rows
                tables.append(t)

    try:
        table = pa.concat_tables(tables)
    except pa.ArrowInvalid:
        # Row groups from different files can differ in dictionary encoding even though the
        # logical types match; normalising to the first table's schema fixes that.
        table = pa.concat_tables([t.cast(tables[0].schema) for t in tables])

    # cast_column, not a Dataset(..., features=...) constructor argument -- Dataset's __init__
    # takes (arrow_table, info, split, ...), so passing features there is silently ignored and
    # the audio column stays a raw {bytes, path} struct that _wav() cannot read. cast_column is
    # the documented path and is metadata-only for a struct that already has the right shape.
    from datasets import Audio
    ds = Dataset(table).cast_column("audio", Audio(sampling_rate=TARGET_SR))

    labs = set(ds["dialect"])
    bad = labs - set(COUNTRIES)
    assert not bad, f"ADI17 subset contains labels outside COUNTRIES: {sorted(bad)}"
    assert not (labs & set(ADI17_ABSENT)), \
        f"ADI17 subset unexpectedly contains {sorted(labs & set(ADI17_ABSENT))}"

    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    ds.save_to_disk(cache)
    print(f"\n  ADI17: {len(ds)} clips cached to {cache}")
    print("  per-dialect: " + "  ".join(f"{c} {got[c]}" for c in ADI17_COUNTRIES if got[c]))
    return ds


def load_adi17_test():
    """ADI17's TEST split, for EVALUATION only.

    Since ADI17 dev turned out to be the NADI ADI20 validation set, test is the closest available
    proxy for the hidden NADI test set -- which is exactly why it is report-only. It is never
    reachable from --adi17-split, which refuses anything but train.

    The test parquet files are resolved and passed EXPLICITLY rather than via the convenience form
    load_dataset(ADI17_REPO, split="test"). That form asks `datasets` to resolve the repo's splits
    for itself, and on a layout it cannot map cleanly it materialises every split into the HF cache
    before selecting one -- which for this repo means pulling the 260 GB train split in order to
    hand back 3 GB of test. Observed filling a 200 GB volume to 84% before anyone noticed, on a run
    whose ADI17 TRAINING data comes from targeted row groups precisely to avoid that download.
    """
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    root = f"datasets/{ADI17_REPO}"
    # Same layout patterns, in the same order, as _adi17_index_row_groups.
    files = []
    for pat in (f"{root}/**/test-*.parquet",
                f"{root}/**/test/*.parquet",
                f"{root}/test/*.parquet"):
        files = sorted(fs.glob(pat))
        if files:
            break
    if not files:
        files = sorted(f for f in fs.glob(f"{root}/**/*.parquet")
                       if "/test/" in f or "/test-" in f)
    if not files:
        print(f"  ADI17 test: no test parquet found under {root} -- skipping (report-only anyway).")
        return None
    try:
        # data_files as a bare list lands in the "train" split by construction; the name is an
        # artifact of the loader, not a claim about the data.
        d = load_dataset("parquet", data_files=["hf://" + f for f in files], split="train")
    except Exception as e:
        print(f"  ADI17 test load failed -- {type(e).__name__}: {e}")
        return None
    keep = [c for c in ("audio", "dialect") if c in d.column_names]
    if "dialect" not in keep:
        print(f"  ADI17 test has columns {d.column_names}; expected a 'dialect'. Skipping.")
        return None
    d = d.select_columns(keep)
    # Reading the parquet directly bypasses the repo's declared features, so `audio` arrives as a
    # raw {bytes, path} struct that _wav() cannot read. Same cast_column reasoning as
    # load_adi17_subset -- metadata-only for a struct that already has the right shape.
    if "audio" in keep:
        from datasets import Audio
        if not isinstance(d.features.get("audio"), Audio):
            d = d.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    labs = set(d["dialect"])
    bad = labs - set(COUNTRIES)
    if bad:
        print(f"  ADI17 test contains labels outside COUNTRIES: {sorted(bad)} -- skipping rather "
              "than scoring against a label set that does not match the head.")
        return None
    print(f"  ADI17 test: {len(d)} clips | {len(labs)} dialect(s) "
          f"(absent by construction: {list(ADI17_ABSENT)})")
    return d


# ============================================================================
# Load data
# ============================================================================

# The manifest. Every training source appends one entry as it loads, and print_dataset_manifest()
# below turns it into the DATASETS USED banner and into two results columns. The point is that a
# run log or a results row can always answer "what data was this?" without re-deriving it from
# the flags -- which matters now that there are three optional sources and they are all off by
# default.
DATASETS_USED = []      # [{name, source, split, clips, hours, classes, kind}]
TRAIN_LABELS  = []      # merged per-row dialect strings, for the sampler and the banner


def _hours(ds):
    """Total audio hours, WITHOUT decoding anything.

    Decoding to measure duration would defeat the purpose -- audio is the expensive column, and
    the whole point of the manifest is that it is free to print. Two cheap paths, one per schema
    in play:

      * vc_augment.py's plain {array, sampling_rate} struct holds raw samples, so the list length
        IS the duration. Exact.
      * an HF Audio() column holds encoded file bytes. Those are only a duration if the encoding
        is uncompressed, so the RIFF header of the first clip is checked before assuming it --
        ADI20 and ADI17 are both 16 kHz 16-bit PCM wav, where duration = (nbytes - 44) / (2*sr).
        Marked as an estimate by the caller.

    Returns (hours, exact) or (None, False) when it cannot be known cheaply. The banner prints
    '?' rather than a fabricated number.
    """
    try:
        import pyarrow.compute as pc
        aud = (getattr(ds, "features", None) or {}).get("audio")
        col = ds.data.column("audio")
        chunks = col.chunks if hasattr(col, "chunks") else [col]

        if aud is not None and not hasattr(aud, "decode"):
            total = sum(pc.sum(pc.list_value_length(ch.field("array"))).as_py() or 0
                        for ch in chunks)
            return float(total) / TARGET_SR / 3600.0, True

        # Encoded bytes. Only a wav header licenses the byte-length estimate.
        first = next((ch.field("bytes")[0].as_py() for ch in chunks if len(ch)), None)
        if not first or not first[:4] == b"RIFF":
            return None, False
        nbytes = sum(pc.sum(pc.binary_length(ch.field("bytes"))).as_py() or 0 for ch in chunks)
        n = len(ds)
        return max(0.0, (nbytes - 44 * n)) / (2.0 * TARGET_SR) / 3600.0, False
    except Exception:
        return None, False


def _register(name, source, split, parts, kind):
    """Record one source in the manifest. `parts` is a list of datasets (a shard directory can
    contribute many), so the entry aggregates over them."""
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    hrs = [_hours(p) for p in parts]
    ok = [h for h, _e in hrs if h is not None]
    DATASETS_USED.append(dict(
        name=name, source=source, split=split,
        clips=sum(len(p) for p in parts),
        hours=(sum(ok) if len(ok) == len(hrs) and hrs else None),
        hours_exact=all(e for _h, e in hrs) if hrs else False,
        classes=len({d for p in parts for d in p["dialect"]}),
        kind=kind))
    return parts[0] if len(parts) == 1 else parts


if ARGS.quick_data:
    print(f"--quick-data {ARGS.quick_data} set: streaming a tiny REAL sample instead of "
          "downloading the full ADI20 dataset. This is for pipeline validation only -- run "
          "without --quick-data for the real training run.")
    train_ds = _stream_tiny_subset("UBC-NLP/NADI_2026_ADI20_micro", "train", ARGS.quick_data)
    val_ds   = _stream_tiny_subset("UBC-NLP/NADI_2026_ADI20_micro", "validation",
                                    max(1, ARGS.quick_data // 2) or 1)
    _register("adi20", "UBC-NLP/NADI_2026_ADI20_micro", "train[quick]", train_ds, "natural")
else:
    print(f"Loading ADI20 micro (subset={ARGS.subset})...")
    _ds = load_dataset("UBC-NLP/NADI_2026_ADI20_micro")
    train_ds = stratified_subset(_ds["train"], ARGS.subset)
    val_ds   = stratified_subset(_ds["validation"], ARGS.subset)
    print(f"  train {len(train_ds)} | val {len(val_ds)}")
    _register("adi20", "UBC-NLP/NADI_2026_ADI20_micro", "train", train_ds, "natural")

# Optional TRAIN-only sources, appended below. Eval sets are NEVER touched: voice conversion is a
# training-time augmentation, and ADI17 train is training data -- augmenting or extending the eval
# sets would invalidate every comparison to earlier runs.
EXTRA_TRAIN_ROWS = 0
_parts, _n_before = [train_ds], len(train_ds)

import glob as _glob
import re as _re
from torch.utils.data import ConcatDataset


def _resolve_extra(path):
    """A path may be a single saved dataset OR a vc_augment.py output directory of shards.

    Reading shards directly is what lets vc_augment.py skip its merge step -- merging writes a
    second full copy alongside the originals, which at the full 4-voice config means needing ~2x
    the disk at the worst possible moment. A shard is just a dataset, and ConcatDataset does not
    care how many there are.
    """
    sidecars = sorted(_glob.glob(os.path.join(path, "shard_*.done.json")))
    if not sidecars:
        return [path]
    dirs, missing, unreadable = [], [], []
    for _sc in sidecars:
        _d = _sc[:-len(".done.json")]
        if not os.path.isdir(_d):
            missing.append(os.path.basename(_d))
            continue
        try:
            with open(_sc) as _fh:
                json.load(_fh)              # same completeness rule as vc_augment.valid_shards
        except Exception:
            unreadable.append(os.path.basename(_d))
            continue
        dirs.append(_d)
    if unreadable:
        print(f"  {len(unreadable)} shard(s) have an unreadable sidecar and were skipped: "
              + ", ".join(unreadable[:6]) + (" ..." if len(unreadable) > 6 else ""))
    if not dirs:
        print(f"FAIL: {path} looks like a shard directory but has no complete shards.")
        sys.exit(2)

    # A sidecar with no shard directory beside it is the fingerprint of vc_augment.py
    # --prune-local: it deletes uploaded shards from local disk and keeps every sidecar. v3
    # skipped those silently, so pointing this at a pruned directory trained on whatever fraction
    # happened to survive and printed only a shard count that looked plausible. The complete copy
    # is on the hub, so the fix is to pass the repo id, not to accept the fraction.
    if missing:
        pct = 100 * len(dirs) / len(sidecars)
        print(f"\n{'!' * 70}")
        print(f"FAIL: {path}")
        print(f"  has {len(sidecars)} shard sidecar(s) but only {len(dirs)} shard director"
              f"{'y' if len(dirs) == 1 else 'ies'} -- {len(missing)} are MISSING.")
        print(f"  Training on this directory would silently use {pct:.0f}% of the data.")
        print("\n  This is what `vc_augment.py --prune-local` leaves behind: it deletes each")
        print("  shard from local disk once the hub has it, but keeps the sidecars. The complete")
        print("  copy is on the hub -- pass the repo id instead of the local path:")
        print(f"\n      --vc-repo <owner>/<name>")
        print("\n  Or pass --allow-partial-shards if training on the fraction is what you want.")
        print(f"{'!' * 70}\n")
        if not ARGS.allow_partial_shards:
            sys.exit(2)
        print("  --allow-partial-shards given: continuing on the incomplete copy.")
    print(f"  {os.path.basename(path)}: {len(dirs)} shard(s)"
          + (f" of {len(sidecars)} (PARTIAL)" if missing else ""))
    return dirs


def _maybe_download(spec):
    """Accept a Hugging Face dataset repo id as well as a local path."""
    if os.path.exists(spec):
        return spec
    # A repo id is exactly "owner/name" -- no leading slash, no drive letter, one separator.
    if _re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", spec):
        from huggingface_hub import snapshot_download
        print(f"  {spec!r} is not a local path; downloading it as a hub dataset...")
        print("    (this lands in the HF cache -- set HF_HOME to a big volume on a rented box)")
        try:
            local = snapshot_download(repo_id=spec, repo_type="dataset")
        except Exception as e:
            # The regex matches "out/dataset" as readily as "owner/name", so a mistyped relative
            # path arrives here looking like a hub 404. Say both possibilities.
            print(f"\nFAIL: could not fetch {spec!r} from the hub ({type(e).__name__}: {e})")
            print("  If that was meant to be a LOCAL path, it does not exist -- check the "
                  "spelling.\n  If it is a private hub repo, check that $HF_TOKEN has access.")
            sys.exit(2)
        print(f"    -> {local}")
        return local
    return spec


def _load_extra_part(sub, label):
    """Validate and normalise one extra training source into something the collate can read.

    `sub` is either a save_to_disk directory or an already-materialised Dataset (which is what
    load_adi17_subset returns -- it builds its table from selected parquet row groups and has no
    directory to point at until it caches).
    """
    part = sub if isinstance(sub, Dataset) else load_from_disk(sub)
    where = f"[{label}] " + (sub if isinstance(sub, str) else f"{len(part)} rows")
    # Fail loudly on a schema the collate cannot read, rather than at step 1 of a long run.
    cols = set(getattr(part, "column_names", []) or [])
    if not {"audio", "dialect"} <= cols:
        print(f"FAIL: {where} has columns {sorted(cols)}; needs 'audio' and 'dialect'.")
        sys.exit(2)

    # Label-domain check. v3 only checked that the COLUMN existed, so an unrecognised dialect
    # string first surfaced as a KeyError inside lmap[...] in a DataLoader worker at step 1 of a
    # multi-hour run, with a traceback that named neither the dataset nor the bad value. This is
    # also the guard that makes a 17-class source safe to merge into a 20-way head.
    bad = sorted(set(part["dialect"]) - set(COUNTRIES))
    if bad:
        print(f"FAIL: {where} has dialect value(s) {bad} that are not in COUNTRIES.")
        print(f"  Expected a subset of {COUNTRIES}.")
        sys.exit(2)

    # --subset applies here too. v3 subsampled only the natural split, so --subset 0.5 halved
    # ADI20 while appending the extras whole -- silently doubling their share of the corpus in a
    # flag whose entire purpose is to hold the data mix fixed while shrinking it.
    part = _subsample(part, ARGS.subset)

    # numpy format on the audio column ONLY, and only for the plain-struct schema. Without it,
    # datasets returns Sequence(Value("float16")) as a PYTHON LIST of ~140k floats per clip, and
    # _wav()'s torch.as_tensor then walks every one: measured 40 ms/clip vs 0.14 ms/clip -- 287x,
    # or 5.1 s per batch of 128 instead of 18 ms.
    #
    # Conditional because an Audio() feature (ADI20, and the ADI17 subset built by
    # load_adi17_subset) already yields numpy from its own decode path, and forcing a format on
    # top of it is at best a no-op and at worst obscures where the decode happens.
    #
    # columns=["audio"] + output_all_columns keeps `dialect` a plain str rather than a numpy
    # scalar, so the collate's lmap[...] lookup stays an ordinary dict hit.
    _aud = (getattr(part, "features", None) or {}).get("audio")
    if _aud is not None and not hasattr(_aud, "decode"):
        part = part.with_format("numpy", columns=["audio"], output_all_columns=True)
    return part


# -- Casablanca, the discarded program half ----------------------------------------------------
# Loaded HERE rather than at the OOD block below because the training merge happens first, and the
# split must run exactly once: a second load_casablanca() call would re-shuffle with a fresh rng
# and the "program-disjoint" guarantee between train and eval would be a coincidence rather than a
# fact. _CASA_CACHE hands the same three datasets to the eval block further down.
_CASA_CACHE = None
if ARGS.train_on_casa:
    if ARGS.skip_ood:
        print("FAIL: --train-on-casa needs Casablanca loaded, but --skip-ood disables it.")
        sys.exit(2)
    _ct, _cs, _ch = load_casablanca(val_frac=ARGS.casa_val_frac,
                                    select_frac=ARGS.casa_select_frac, want_train=True)
    _CASA_CACHE = (_cs, _ch)
    if _ct is None or len(_ct) == 0:
        print("FAIL: --train-on-casa is on but the recovered training half is empty. That means "
              "--casa-val-frac is 1.0 (every program kept for eval), so there is nothing left to "
              "train on.")
        sys.exit(2)
    if ARGS.casa_train_frac < 1.0:
        _keep = sorted(set(_ct["program"]))
        _rng = np.random.default_rng(SEED)
        _rng.shuffle(_keep)
        _keep = set(_keep[:max(1, int(round(len(_keep) * ARGS.casa_train_frac)))])
        _ct = _ct.select([i for i, pr in enumerate(_ct["program"]) if pr in _keep])
        print(f"  --casa-train-frac {ARGS.casa_train_frac}: kept {len(_keep)} program(s), "
              f"{len(_ct)} clips")
    # Drop the eval-only columns so the schema matches what the collate reads.
    _ct = _ct.select_columns([c for c in ("audio", "dialect") if c in _ct.column_names])
    _ct = _load_extra_part(_ct, "casa")
    _register("casa", "UBC-NLP/Casablanca", "unused-programs", _ct, "natural")
    EXTRA_TRAIN_ROWS += len(_ct)
    _parts.append(_ct)
    print("  NOTE: casa_acc stays program-disjoint from training, so it is still a valid "
          "SELECTION metric -- but it is NO LONGER an out-of-domain one.")
    print("        Report casa_* as in-domain from here on; the model has seen Casablanca.")

# -- ADI17 train subset -----------------------------------------------------------------------
if ARGS.use_adi17:
    _adi17 = _load_extra_part(load_adi17_subset(), "adi17")
    _register("adi17", ADI17_REPO, ARGS.adi17_split, _adi17, "natural")
    EXTRA_TRAIN_ROWS += len(_adi17)
    _parts.append(_adi17)

# -- private voice-converted dataset ----------------------------------------------------------
if ARGS.use_vc:
    _vc_src = ARGS.vc_repo.strip()
    _vc_local = _maybe_download(_vc_src)
    if not os.path.exists(_vc_local):
        print(f"FAIL: --vc-repo path does not exist: {_vc_local}")
        sys.exit(2)
    _vc_parts = [_load_extra_part(_sub, "vc") for _sub in _resolve_extra(_vc_local)]
    _vc_rows = sum(len(p) for p in _vc_parts)

    # Cap the synthetic share. vc_augment.py --voices 4 emits four converted clips per source, so
    # an uncapped merge is ~80% synthetic and most of the model's updates land on kNN-VC artifacts
    # rather than on real speech. Applied by seeded subsampling so it is reproducible and so it
    # does not require regenerating the dataset at a different --voices.
    if ARGS.vc_max_frac < 1.0:
        _natural = len(train_ds) + sum(len(p) for p in _parts[1:])
        _cap = int(ARGS.vc_max_frac * _natural / (1.0 - ARGS.vc_max_frac))
        if _vc_rows > _cap:
            _keep = _cap / _vc_rows
            print(f"  --vc-max-frac {ARGS.vc_max_frac}: {_vc_rows} converted rows would be "
                  f"{100 * _vc_rows / (_vc_rows + _natural):.0f}% of the corpus; "
                  f"subsampling to {_cap} ({100 * _keep:.0f}% kept)")
            _vc_parts = [_subsample(p, _keep) for p in _vc_parts]
            _vc_rows = sum(len(p) for p in _vc_parts)
    for _p in _vc_parts:
        EXTRA_TRAIN_ROWS += len(_p)
        _parts.append(_p)
    _register("vc", _vc_src, "train", _vc_parts, "SYNTHETIC")

# -- generic escape hatch ---------------------------------------------------------------------
if ARGS.extra_train_list:
    ARGS.extra_train_list = [_maybe_download(_p) for _p in ARGS.extra_train_list]
    for _p in ARGS.extra_train_list:
        if not os.path.exists(_p):
            print(f"FAIL: --extra-train-data path does not exist: {_p}")
            sys.exit(2)
        _subs = [_load_extra_part(_sub, "extra") for _sub in _resolve_extra(_p)]
        for _s in _subs:
            EXTRA_TRAIN_ROWS += len(_s)
            _parts.append(_s)
        _register("extra", _p, "train", _subs, "natural")

if len(_parts) > 1:
    # ConcatDataset, NOT datasets.concatenate_datasets: ADI20's `audio` is an HF Audio() feature
    # while vc_augment.py writes a plain {array, sampling_rate} struct, and concatenate_datasets
    # rejects that mismatch. Casting either direction means decoding or re-encoding the whole
    # corpus. The collate only ever reads sample["audio"]["array"], ["sampling_rate"] and
    # ["dialect"], and both schemas present exactly that, so composing at the torch Dataset level
    # sidesteps the feature system entirely.
    train_ds = ConcatDataset(_parts)

# The merged label vector, read once from the string column of each part (no audio decoded). It
# feeds the sampler weights and the per-class table in the banner, and it is the only place that
# knows the merged class prior -- which is the number that actually matters once a 17-class
# source is in the mix.
for _p in _parts:
    TRAIN_LABELS.extend(list(_p["dialect"]))
assert len(TRAIN_LABELS) == len(train_ds), \
    f"label vector ({len(TRAIN_LABELS)}) and train set ({len(train_ds)}) disagree"

TRAIN_CLASS_COUNTS = Counter(TRAIN_LABELS)

# Per-row sampling weights, w_i = count(class_i) ** -alpha.
#   alpha 0.0 -> uniform (identical to plain shuffling; build_loaders skips the sampler entirely)
#   alpha 1.0 -> every class drawn equally often
#   alpha 0.5 -> halfway, the 'auto' default when any extra source is on
# See --sampler-alpha for why partial rather than full is the default. Counts, not hours, are the
# unit: the model sees one fixed-length crop per clip regardless of how long the clip is, so
# exposure is per-clip. Hours are reported in the banner because that is how corpus size is
# usually described, but they are not what the sampler balances.
TRAIN_SAMPLE_WEIGHTS = None
if ARGS.sampler_alpha > 0:
    _w = {c: float(n) ** (-ARGS.sampler_alpha) for c, n in TRAIN_CLASS_COUNTS.items()}
    _mean = sum(_w.values()) / len(_w)
    TRAIN_CLASS_WEIGHTS = {c: v / _mean for c, v in _w.items()}   # mean 1.0, for readability
    TRAIN_SAMPLE_WEIGHTS = [TRAIN_CLASS_WEIGHTS[c] for c in TRAIN_LABELS]
else:
    TRAIN_CLASS_WEIGHTS = {c: 1.0 for c in TRAIN_CLASS_COUNTS}


def print_dataset_manifest():
    """The one banner that says exactly which datasets this run trained on.

    Printed after the merge, repeated in the final report, and mirrored into the results row as
    the `datasets` / `dataset_rows` columns.
    """
    W = 78
    print("\n" + "=" * W)
    print("  DATASETS USED".ljust(W))
    print("=" * W)
    _syn = 0
    for d in DATASETS_USED:
        # '~' marks an estimate from encoded byte length rather than an exact sample count; see
        # _hours(). Never printed as if it were measured.
        _h = (f"{'' if d['hours_exact'] else '~'}{d['hours']:6.1f} h"
              if d["hours"] is not None else "     ? h")
        print(f"  {d['name']:<7s} {d['source'][:34]:<34s} [{d['split'][:11]:<11s}] "
              f"{d['clips']:>7d} clips {_h} {d['classes']:>3d} cls  {d['kind']}")
        if d["kind"] == "SYNTHETIC":
            _syn += d["clips"]
    _tot = len(train_ds)
    _th = [d["hours"] for d in DATASETS_USED]
    _tot_h = (f"{'' if all(d['hours_exact'] for d in DATASETS_USED) else '~'}{sum(_th):.1f} h"
              if _th and all(h is not None for h in _th) else "? h")
    print("  " + "-" * (W - 4))
    print(f"  merged  {_tot:>7d} clips | {_tot_h} | {100 * _syn / max(1, _tot):.1f}% synthetic"
          f" | sampler-alpha {ARGS.sampler_alpha}"
          + (" (auto)" if ARGS.sampler_alpha_auto else ""))

    # The merged per-class prior, which is the number that matters once a 17-class source is in
    # the mix -- and the one v3 never printed.
    print("  per-class clips (sampling weight):")
    _row = []
    for c in COUNTRIES:
        n = TRAIN_CLASS_COUNTS.get(c, 0)
        _row.append(f"{c} {n:>6d} ({TRAIN_CLASS_WEIGHTS.get(c, 0.0):.2f})")
        if len(_row) == 4:
            print("      " + "  ".join(_row))
            _row = []
    if _row:
        print("      " + "  ".join(_row))
    _absent = [c for c in COUNTRIES if not TRAIN_CLASS_COUNTS.get(c)]
    if _absent:
        print(f"  WARNING: no training data at all for {_absent} -- the head has 20 outputs and "
              "these\n  can only ever be predicted by accident.")

    print("  EVAL (never trained on): adi20 val"
          + ("" if ARGS.skip_ood else " | Casablanca select+holdout | MADIS-5")
          + (" | ADI17 test (report-only)" if ARGS.eval_adi17_test else ""))
    if ARGS.use_adi17:
        print("  NOT used for training: ADI17 dev (== the NADI ADI20 validation set), ADI17 test")
        print(f"  NOTE: ADI17 has no {'/'.join(ADI17_ABSENT)} -- those three could not be topped "
              "up from it.")
    print("=" * W)


print_dataset_manifest()

if ARGS.skip_ood:
    print("--skip-ood set: skipping Casablanca/MADIS-5 entirely (ID-only run -- no OOD "
          "selection metric, casa_acc/casa_cavg20/madis_overall will be absent from results).")
    casa_ds, casa_holdout_ds, madis5_ds = None, None, None
else:
    print("Loading OOD sets...")
    # Reuse the single split made above when --train-on-casa ran; calling load_casablanca again
    # would reshuffle and silently break the train/eval program disjointness it just asserted.
    _casa_sel, _casa_hold = (_CASA_CACHE if _CASA_CACHE is not None
                             else load_casablanca(val_frac=ARGS.casa_val_frac,
                                                  select_frac=ARGS.casa_select_frac))
    casa_ds         = _subsample(_casa_sel, ARGS.ood_subset)
    casa_holdout_ds = _subsample(_casa_hold, ARGS.ood_subset)
    madis5_ds = _subsample(load_madis5(), ARGS.ood_subset)

    if casa_ds is None:
        print("\nWARNING: no Casablanca -- selection falls back to in-domain val.")
        print("Results from this run are weak evidence. Fix the HF token first.")

# ADI17 test: an EVAL set, and the closest available proxy for the hidden NADI test set. Loaded
# here but scored only after training ends -- see the run_eval_and_maybe_save/final-eval split in
# train_one(). It is deliberately not subsampled by --ood-subset: it is a report-only number and
# a partial one would be harder to compare across runs than no number at all.
adi17_test_ds = None
if ARGS.eval_adi17_test:
    print("Loading ADI17 test (eval only)...")
    adi17_test_ds = load_adi17_test()
    if adi17_test_ds is None:
        print("  --eval-adi17-test was requested but the load failed; continuing without it. "
              "The adi17_* columns will be absent from the results row.")



# ============================================================================
# Dataloaders
# ============================================================================
def _worker_init(worker_id):
    """Pin each DataLoader worker to a single compute thread. Module-level so it stays picklable
    for the spawn start method. See the comment at its use site in _loader_kwargs."""
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _tta_windows(w, n, win_samples):
    """`n` evenly-spaced windows of `win_samples` across a clip, for test-time augmentation.

    Clips shorter than the window yield a single full-length view rather than being padded up to
    n copies of themselves -- averaging n identical predictions is just the single prediction, so
    the extra forward passes would be pure cost.
    """
    T = w.shape[-1]
    if n < 2 or T <= win_samples:
        return [w]
    starts = np.linspace(0, T - win_samples, n).round().astype(int)
    return [w[..., int(s):int(s) + win_samples] for s in sorted(set(starts.tolist()))]


def build_loaders(spec, tta=0):
    fe = get_feature_extractor(spec)
    eff_bs = BATCH_SIZE * max(1, N_GPU)

    # Eval clips run up to MAX_AUDIO_SECONDS_COHERE (30s) vs. the train crop -- several times
    # the frames per sample. Reusing the train micro-batch for eval, as cohere_train.py does,
    # is fine at batch 32 but risks OOMing partway through the first eval at this script's
    # much larger train batch, even though training itself fits comfortably. Sized against the
    # LONGEST train crop, so --crop-set picks the same conservative number --crop would.
    _train_crop = max(ARGS.crop_list) if ARGS.crop_list else ARGS.crop
    if ARGS.eval_batch_size is not None:
        eval_bs = ARGS.eval_batch_size
    elif _train_crop and _train_crop > 0:
        eval_bs = max(8, int(BATCH_SIZE * _train_crop / MAX_AUDIO_SECONDS_COHERE))
    else:
        eval_bs = BATCH_SIZE   # --crop 0: train clips are already full-length, same as eval
    if tta and tta > 1:
        # Each clip now becomes up to `tta` rows through the model, so the DataLoader batch has
        # to shrink by the same factor or the forward pass sees tta x the intended rows and OOMs.
        eval_bs = max(1, eval_bs // tta)
    eval_eff_bs = eval_bs * max(1, N_GPU)
    print(f"  eval batch_size={eval_bs} (train batch_size={BATCH_SIZE}) -- scaled down so eval's "
          f"up-to-{MAX_AUDIO_SECONDS_COHERE}s clips don't OOM at the training micro-batch size"
          + (f", and divided by --tta {tta}" if tta and tta > 1 else ""))

    def _loader_kwargs():
        kw = dict(num_workers=ARGS.num_workers)
        if ARGS.num_workers > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = ARGS.prefetch_factor
            # Without this every worker inherits torch's default intra-op thread count, which is
            # the core count -- so N workers each spawn N compute threads and 24 workers on a
            # 24-core box fight over 576 threads' worth of demand. The per-op slowdown from that
            # contention is severe (100x+ on small tensors) and it gets WORSE the more workers
            # you add, which makes it look like the augs are expensive when the real problem is
            # thread thrash. Each worker handles one sample at a time; it wants exactly 1 thread.
            kw["worker_init_fn"] = _worker_init
        return kw

    def make_collate(train=False, label_col="dialect", label_map=None, domain_col=None):
        lmap = labels2id if label_map is None else label_map

        def _wav(s):
            try:
                a = s["audio"]
                arr, sr = a["array"], int(a["sampling_rate"])
            except Exception:
                return None
            try:
                w = arr if torch.is_tensor(arr) else torch.as_tensor(arr)
                w = w.to(torch.float32)
            except Exception:
                return None
            if w.ndim > 1:
                w = w.mean(0)
            if w.numel() < 400:
                return None
            if not torch.isfinite(w).all():
                return None
            if sr != TARGET_SR:
                try:
                    w = _ta_resample(w, sr, TARGET_SR)
                except Exception:
                    return None
            max_samples = int(MAX_AUDIO_SECONDS_COHERE * TARGET_SR)
            if w.shape[-1] > max_samples:
                w = w[..., :max_samples]
            return w

        # Train batches are padded to a fixed crop length (not the per-batch max) so every train
        # batch has an identical shape and cuDNN autotune / torch.compile don't re-tune or
        # recompile every step. With --crop-set the length is drawn per BATCH rather than being
        # constant, which keeps that property (one shape per batch) while removing the fixed-4s
        # train vs up-to-30s eval duration mismatch -- torch.compile then builds len(crop_list)
        # graphs instead of one. Disabled entirely when --crop 0.
        crop_choices = None
        if train and CFG.get("crop"):
            crop_choices = [int(c * SR) for c in (ARGS.crop_list or [CFG["crop"]])]
        aug_list, p_cheap, p_slow = ARGS.aug_list, ARGS.aug_prob, ARGS.aug_prob_slow

        # TTA is eval-only: at train time the random crop already provides the view sampling,
        # and averaging views would defeat it.
        tta_windows = 0 if train else (tta if tta and tta > 1 else 0)
        # Default the TTA window to the LONGEST duration the model trained on -- evaluating on a
        # length the model has seen is the point, and the longest one carries the most context.
        _tta_sec = ARGS.tta_seconds or (max(ARGS.crop_list) if ARGS.crop_list else ARGS.crop)
        tta_samples = int((_tta_sec or MAX_AUDIO_SECONDS_COHERE) * SR)

        def collate(samples):
            wavs, labs, doms, lens, groups = [], [], [], [], []
            # One crop length for the whole batch -- see the comment above.
            crop_samples = random.choice(crop_choices) if crop_choices else None
            for s in samples:
                w = _wav(s)
                if w is None:
                    SKIPPED["n"] += 1
                    continue
                if train:
                    w = random_crop(w, seconds=(crop_samples / SR) if crop_samples else None)
                    if aug_list:
                        # After the crop, so an aug only ever processes crop_samples of audio.
                        w = apply_augmentations(w, aug_list, p_cheap, p_slow)
                    if crop_samples:
                        # aug_speed changes the length, so a clip can now come back LONGER than
                        # the crop. The baseline only ever padded (random_crop guaranteed
                        # w <= crop_samples); without this truncation an over-long clip would
                        # break the fixed batch shape and silently corrupt the input_lengths
                        # fraction below.
                        if w.shape[-1] > crop_samples:
                            w = w[..., :crop_samples]
                        true_len = w.shape[-1]
                        if true_len < crop_samples:
                            w = torch.nn.functional.pad(w, (0, crop_samples - true_len))
                        lens.append(true_len)
                if tta_windows:
                    # One row per window, all tagged with this clip's group id; _logits_for
                    # averages within groups so the model still yields one prediction per clip.
                    gid = len(labs)
                    for view in _tta_windows(w, tta_windows, tta_samples):
                        wavs.append(view.numpy())
                        groups.append(gid)
                else:
                    wavs.append(w.numpy())
                labs.append(lmap[s[label_col]])
                if domain_col is not None:
                    doms.append(s[domain_col])
            if not wavs:
                return None, None, None
            labels = torch.tensor(labs)
            feats, mask = _extract_features(fe, wavs)
            f = {"input_features": feats}
            if mask is not None:
                f["attention_mask"] = mask
            if crop_samples:
                # Fallback for DialectID._derive_frame_mask if fixed-length inputs mean the
                # processor no longer emits its own attention_mask (see the comment there) --
                # the true pre-pad sample count per clip, so pooling doesn't average over the
                # padding we just introduced.
                f["input_lengths"] = torch.tensor(lens, dtype=torch.long)
                # The DENOMINATOR for that fraction. Must travel with the batch: with --crop-set
                # the padded length varies per batch, so reading the global CFG["crop"] (as the
                # baseline did) would compute the valid-frame count against the wrong total and
                # silently mis-mask pooling on every batch whose crop wasn't the default.
                f["padded_samples"] = torch.tensor(crop_samples, dtype=torch.long)
            if groups:
                # Popped in _logits_for before the model call -- it is a grouping key, not a
                # model input, and feats is splatted straight into model(**feats).
                f["tta_groups"] = torch.tensor(groups, dtype=torch.long)
            return f, labels, (doms if domain_col else None)

        return collate

    # shuffle=True and sampler= are mutually exclusive in DataLoader, so the alpha 0 path stays
    # exactly the v3 loader rather than a WeightedRandomSampler that happens to be uniform --
    # same distribution, but sampling WITH replacement is not the same as a shuffled epoch and
    # the difference would quietly ride along in every "unchanged baseline" comparison.
    if TRAIN_SAMPLE_WEIGHTS is not None:
        from torch.utils.data import WeightedRandomSampler
        _sampler = WeightedRandomSampler(
            weights=torch.as_tensor(TRAIN_SAMPLE_WEIGHTS, dtype=torch.double),
            num_samples=len(train_ds), replacement=True,
            generator=torch.Generator().manual_seed(SEED))
        tl = DataLoader(train_ds, sampler=_sampler, batch_size=eff_bs,
                        collate_fn=make_collate(train=True), drop_last=True,
                        pin_memory=True, **_loader_kwargs())
        _lo = min(TRAIN_CLASS_WEIGHTS.values()); _hi = max(TRAIN_CLASS_WEIGHTS.values())
        print(f"  train sampler: weighted, alpha={ARGS.sampler_alpha} "
              f"(class weights {_lo:.2f}..{_hi:.2f}, drawn with replacement)")
    else:
        tl = DataLoader(train_ds, shuffle=True, batch_size=eff_bs,
                        collate_fn=make_collate(train=True), drop_last=True,
                        pin_memory=True, **_loader_kwargs())
        print("  train sampler: plain shuffle (--sampler-alpha 0)")
    vl = DataLoader(val_ds, shuffle=False, batch_size=eval_eff_bs,
                    collate_fn=make_collate(), **_loader_kwargs())
    # domain_col carries the program id for --confusion-by-program. Only requested when the flag
    # is on and the column survived: it costs a per-clip python string in every batch.
    _casa_dom = ("program" if (ARGS.confusion_by_program and casa_ds is not None
                               and "program" in casa_ds.column_names) else None)
    if ARGS.confusion_by_program and _casa_dom is None and casa_ds is not None:
        print("  NOTE: --confusion-by-program is on but Casablanca has no 'program' column "
              "(no seg_id?) --\n        the per-program breakdown will be skipped.")
    cl = (DataLoader(casa_ds, shuffle=False, batch_size=eval_eff_bs,
                     collate_fn=make_collate(label_map=labels2id, domain_col=_casa_dom),
                     **_loader_kwargs())
          if casa_ds is not None else None)
    chl = (DataLoader(casa_holdout_ds, shuffle=False, batch_size=eval_eff_bs,
                      collate_fn=make_collate(label_map=labels2id, domain_col=_casa_dom),
                      **_loader_kwargs())
           if casa_holdout_ds is not None else None)
    a17l = (DataLoader(adi17_test_ds, shuffle=False, batch_size=eval_eff_bs,
                       collate_fn=make_collate(label_map=labels2id), **_loader_kwargs())
            if adi17_test_ds is not None else None)
    ml = (DataLoader(madis5_ds, shuffle=False, batch_size=eval_eff_bs,
                     collate_fn=make_collate(label_map=REGION2ID, domain_col="domain"),
                     **_loader_kwargs())
          if madis5_ds is not None else None)
    print(f"  dataloaders: num_workers={ARGS.num_workers} "
          f"prefetch_factor={ARGS.prefetch_factor if ARGS.num_workers else 'n/a'} "
          f"persistent_workers={ARGS.num_workers > 0}")
    return tl, vl, cl, ml, chl, a17l



# ============================================================================
# Cavg (NIST-style average cost, identical to cohere_bench.py/models_bench.py)
# ============================================================================
def llr(logits):
    classes = logits.shape[1]
    l  = logits.unsqueeze(1).repeat(1,classes,1)
    l2 = logits.unsqueeze(1).repeat(1,classes,1).permute(0,2,1)
    e = torch.exp(l - l2)
    for i in range(len(e)):
        e[i].fill_diagonal_(0)
    return -torch.log(torch.sum(e, dim=-1)/(classes-1))

def compute_actual_cost(scores, labels, p_target, c_miss=1, c_fa=1):
    beta = c_fa*(1-p_target)/(c_miss*p_target)
    decisions = (scores >= np.log(beta)).astype("i")
    num_t = np.sum(labels); num_n = np.sum(1-labels)
    fp = np.sum(decisions*(1-labels)); fn = np.sum((1-decisions)*labels)
    fpr = fp/num_n if num_n > 0 else np.nan
    fnr = fn/num_t if num_t > 0 else np.nan
    return fnr + beta*fpr, fpr, fnr

def compute_ave_cost(logits, labels, num_l):
    llratio = llr(logits).numpy()
    labels = labels.numpy().copy()
    order = labels.argsort(); labels.sort()
    llratio = llratio[order]
    idx = np.append(np.where(labels[:-1] != labels[1:])[0] + 1, [len(labels)])
    one_hot = np.eye(num_l)[labels]
    fprs, fnrs, last = [], [], 0
    for i in idx:
        if i <= last:
            continue
        _, fpr, fnr = compute_actual_cost(llratio[last:i], one_hot[last:i], 0.5)
        fprs.append(fpr); fnrs.append(fnr); last = i
    return np.nansum(fprs)/num_l + np.nansum(fnrs)/num_l, \
           np.nansum(fprs)/num_l, np.nansum(fnrs)/num_l

_c,_,_ = compute_ave_cost(torch.randn(400,20), torch.randint(0,20,(400,)), num_l=20)
print(f"random-logit Cavg {_c:.3f}  (expect ~1.0)")



# ============================================================================
# Evaluation (identical to cohere_bench.py)
# ============================================================================
@torch.no_grad()
def _logits_for(model, loader, desc, want_domains=False, precision="bf16"):
    model.eval()
    L, Y, D = [], [], []
    n_empty = 0
    # Eval clips run up to 30s vs 4s train crops -- ~7.5x the frames -- and used to be the only
    # part of the loop with enough work per kernel launch to hit 100% GPU util. Autocast (on by
    # default, --no-eval-autocast to force the old fp32 path for exact cross-run comparability)
    # cuts that wall-time further without touching train precision handling at all.
    eval_ctx = _autocast_ctx(precision if ARGS.eval_autocast else "fp32")
    for batch in tqdm(loader, desc=desc, leave=False):
        feats, labels, doms = batch
        if feats is None:
            n_empty += 1
            continue
        feats = {k: v.to(device) for k, v in feats.items()}
        # Grouping key, not a model input -- feats is splatted into model(**feats) below.
        tta_groups = feats.pop("tta_groups", None)
        with eval_ctx:
            lg = model(**feats)
        if tta_groups is not None:
            # Average LOG-PROBABILITIES, not raw logits: the per-window logit scales are not
            # commensurable, and log_softmax also keeps the result in a valid log-probability
            # space, which compute_ave_cost's llr() needs. This is the geometric mean of the
            # per-window predictions.
            lp = torch.log_softmax(lg.float(), dim=-1)
            n_groups = int(tta_groups.max().item()) + 1
            summed = torch.zeros(n_groups, lp.shape[1], device=lp.device, dtype=lp.dtype)
            summed.index_add_(0, tta_groups, lp)
            counts = torch.zeros(n_groups, device=lp.device, dtype=lp.dtype)
            counts.index_add_(0, tta_groups, torch.ones_like(tta_groups, dtype=lp.dtype))
            lg = summed / counts.clamp(min=1).unsqueeze(1)
        _assert_aligned(lg, labels, desc)
        L.append(lg.float().cpu()); Y.append(labels)
        if want_domains and doms is not None:
            D.extend(doms)
    if not L:
        raise RuntimeError(f"{desc}: no decodable clips at all")
    if n_empty:
        print(f"       [{desc}] {n_empty} batch(es) fully skipped -- bad audio")
    return torch.cat(L), torch.cat(Y), D


def _program_report(p, t, groups, tag, out_dir=None, top_k=12):
    """Break the errors down by (true class, PROGRAM) as well as by class.

    Casablanca is split by program, and a class-level confusion cell says nothing about whether
    its errors come from one broadcast or from all of them. Those are completely different
    problems: a boundary the model has genuinely not learned is fixable with more data for that
    class, whereas one channel or one set of speakers being read as another country is a domain
    artifact that more data of the same kind will not touch.

    The case this was written for: JOR->SYR was EXACTLY 292 clips in both the --llrd 0.95 and
    --llrd 1.0 runs -- identical while every neighbouring cell moved by tens of clips. A cell that
    immovable under a change that shifted everything else is not a decision boundary.
    """
    groups = np.asarray([str(g) for g in groups])
    if len(groups) != len(t):
        print(f"    [--confusion-by-program] {len(groups)} program ids for {len(t)} clips -- "
              "skipping rather than reporting a misaligned breakdown.")
        return
    err = p != t
    rows = []
    for cls in sorted(set(t.tolist())):
        m_cls = t == cls
        for g in sorted(set(groups[m_cls].tolist())):
            m = m_cls & (groups == g)
            n, ne = int(m.sum()), int((m & err).sum())
            if not n:
                continue
            # The single label this program's clips are most often given, right or wrong.
            vals, cnts = np.unique(p[m], return_counts=True)
            top = int(vals[int(cnts.argmax())])
            rows.append((ne, n, COUNTRIES[int(cls)], g, COUNTRIES[top],
                         int(cnts.max()), 100.0 * ne / n))
    if not rows:
        return
    rows.sort(reverse=True)
    tot_err = int(err.sum())
    print(f"\n  -- per-program breakdown [{tag}] -- {len(rows)} (class, program) cells")
    print(f"    {'true':<5s} {'program':<22s} {'n':>5s} {'err':>5s} {'err%':>6s}  "
          "predicted-most-often")
    for ne, n, cls, g, top, ntop, pct in rows[:top_k]:
        print(f"    {cls:<5s} {g[:22]:<22s} {n:>5d} {ne:>5d} {pct:>5.1f}%  "
              f"{top} ({100*ntop/n:.0f}% of the program)"
              + ("   <-- ENTIRE PROGRAM MISREAD" if top != cls and ntop / n > 0.8 else ""))

    # The number that decides whether a class-level cell is a data problem or a domain artifact.
    worst = rows[0]
    print(f"\n    worst single (class, program) cell: {worst[2]} / {worst[3]} "
          f"= {worst[0]} errors, {100*worst[0]/max(1,tot_err):.1f}% of ALL {tot_err} errors")
    by_cls = defaultdict(list)
    for ne, n, cls, g, _top, _nt, _pct in rows:
        by_cls[cls].append(ne)
    print("    error concentration per class (top program's share of that class's errors):")
    for cls in sorted(by_cls, key=lambda c: -sum(by_cls[c])):
        es = sorted(by_cls[cls], reverse=True)
        tot = sum(es)
        if tot < 20:
            continue
        share = 100 * es[0] / tot
        verdict = ("ONE PROGRAM -- domain artifact, more training data will not fix it"
                   if share > 60 and len(es) > 1 else
                   "concentrated" if share > 40 else "spread across programs -- genuine boundary")
        print(f"      {cls:<5s} {tot:>5d} errors over {len(es):>3d} program(s), "
              f"top {share:5.1f}%  {verdict}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir,
                            f"by_program_{tag.replace(' ', '_').replace('/', '_')}.csv")
        pd.DataFrame([dict(true=c, program=g, n=n, errors=ne, err_pct=pct,
                           predicted_most=top, predicted_most_n=nt)
                      for ne, n, c, g, top, nt, pct in rows]).to_csv(path, index=False)
        print(f"    per-program table -> {path}")


def confusion_report(preds, y, tag, out_dir=None, top_k=15, groups=None):
    """Per-class recall/precision, the worst confusion pairs, and a within-region error breakdown.

    The question this exists to answer: 20-way COUNTRY-level Arabic dialect ID puts 6 Gulf
    countries (KSA/UAE/QAT/KUW/BAH/OMA) and 4 Levantine ones into the label set, and neighbouring
    countries genuinely share dialect features. So before spending compute chasing a higher
    number, it is worth knowing how much of the remaining error is confusion between genuinely
    adjacent dialects rather than the model being wrong in an addressable way -- a high
    within-region share means the ceiling is closer than the raw accuracy suggests.
    """
    p = preds.numpy() if torch.is_tensor(preds) else np.asarray(preds)
    t = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    K = len(COUNTRIES)
    cm = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(t, p):
        cm[int(a), int(b)] += 1

    support = cm.sum(1)
    predicted = cm.sum(0)
    correct = np.diag(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, correct / np.maximum(support, 1), np.nan)
        precision = np.where(predicted > 0, correct / np.maximum(predicted, 1), np.nan)

    print(f"\n  -- confusion report [{tag}] --")
    print(f"    {'class':<6s} {'n':>6s} {'recall':>8s} {'prec':>8s}   most-confused-with")
    order = np.argsort(recall)              # worst classes first: that is where the headroom is
    for c in order:
        if support[c] == 0:
            continue
        row = cm[c].copy(); row[c] = 0
        if row.sum():
            j = int(row.argmax())
            worst = f"{COUNTRIES[j]} ({100*row[j]/support[c]:.0f}%)"
        else:
            worst = "-"
        print(f"    {COUNTRIES[c]:<6s} {support[c]:>6d} {100*recall[c]:>7.1f}% "
              f"{100*precision[c]:>7.1f}%   {worst}")

    # Worst off-diagonal cells overall, as a share of the true class's support.
    pairs = []
    for a in range(K):
        for b in range(K):
            if a != b and cm[a, b] > 0:
                pairs.append((int(cm[a, b]), a, b))
    pairs.sort(reverse=True)
    n_err = int((p != t).sum())
    print(f"\n    top {min(top_k, len(pairs))} confusion pairs "
          f"(of {n_err} errors in {len(t)} clips):")
    for n, a, b in pairs[:top_k]:
        print(f"      {COUNTRIES[a]:>4s} -> {COUNTRIES[b]:<4s}  {n:>5d}  "
              f"({100*n/max(1,n_err):4.1f}% of all errors, {100*n/max(1,support[a]):4.1f}% of {COUNTRIES[a]})")

    # The headline number: how much error is between dialects of the same broad region.
    reg = ID2REGION.numpy()
    err = p != t
    if n_err:
        same_region = int(((reg[p] == reg[t]) & err).sum())
        print(f"\n    within-region errors: {same_region}/{n_err} ({100*same_region/n_err:.1f}%) "
              "-- confusion between neighbouring dialects of the same region")
        print(f"    cross-region errors:  {n_err-same_region}/{n_err} "
              f"({100*(n_err-same_region)/n_err:.1f}%) -- the genuinely addressable share")
        for r_i, r_name in enumerate(REGIONS):
            m = reg[t] == r_i
            if not m.sum():
                continue
            in_r = int(((reg[p] == reg[t]) & err & m).sum())
            print(f"      {r_name:<24s} n={int(m.sum()):>5d}  "
                  f"region-level acc {100*(reg[p][m]==r_i).mean():5.1f}%  "
                  f"country errors staying in-region {in_r}")
        print(f"    region-level accuracy overall: {100*(reg[p]==reg[t]).mean():.1f}% "
              f"(vs {100*(p==t).mean():.1f}% country-level)")

    if groups is not None:
        _program_report(p, t, groups, tag, out_dir=out_dir)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"confusion_{tag.replace(' ', '_').replace('/', '_')}.csv")
        pd.DataFrame(cm, index=COUNTRIES, columns=COUNTRIES).to_csv(path)
        print(f"    full 20x20 matrix -> {path}")
    return cm


def _macro_acc(preds, y, n_classes):
    """Mean of per-class recalls. The ADI20 validation split is 8.37x imbalanced (TUN is 15.26%
    of it, UAE 10.46%, SUD 1.82%), so plain micro accuracy is substantially a TUN/UAE score and
    can move without the model getting better at the other 18 dialects.
    """
    recalls = []
    for c in range(n_classes):
        m = (y == c)
        if int(m.sum()) == 0:
            continue
        recalls.append((preds[m] == c).float().mean().item())
    return float(np.mean(recalls)) * 100 if recalls else float("nan")


def eval_id(model, loader, precision="bf16", confusion=False):
    lg, y, _ = _logits_for(model, loader, "ID val", precision=precision)
    preds = lg.argmax(-1)
    acc = (preds == y).float().mean().item() * 100
    macro = _macro_acc(preds, y, len(COUNTRIES))
    cost, _, _ = compute_ave_cost(lg, y, num_l=len(COUNTRIES))
    if confusion:
        confusion_report(preds, y, "ID_val", out_dir=ARGS.plots_dir)
    return acc, macro, cost


def eval_subset_20way(model, loader, precision="bf16", confusion=False, tag="Casablanca",
                      present=None, want_groups=False):
    """Score an eval set that contains only SOME of the 20 countries.

    Was eval_casablanca, hardcoded to CASA_PRESENT. Generalised because ADI17 test is the same
    shape of problem from the other direction -- Casablanca has 8 of 20, ADI17 test has 17 of 20 --
    and it needs exactly the same pair of numbers, not an approximation of them.

    Returns (acc, acc_restricted, cavg20, cavg_restricted, off_set).
    """
    present = list(CASA_PRESENT if present is None else present)
    absent = [c for c in COUNTRIES if c not in present]
    # The program ids ride the existing domain_col channel rather than being re-derived from the
    # dataset afterwards. That matters for alignment: the collate drops undecodable clips, so a
    # separately-read column would silently shift against the logits by however many were skipped.
    lg, y, grp = _logits_for(model, loader, tag, want_domains=want_groups, precision=precision)
    preds = lg.argmax(-1)
    if confusion:
        confusion_report(preds, y, tag, out_dir=ARGS.plots_dir,
                         groups=(grp if want_groups and grp else None))
    acc = (preds == y).float().mean().item() * 100
    off = np.mean([id2labels[int(p)] in absent for p in preds]) * 100
    c20, _, _ = compute_ave_cost(lg, y, num_l=len(COUNTRIES))
    idx = torch.tensor([labels2id[c] for c in present])
    remap = {int(labels2id[c]): i for i, c in enumerate(present)}
    yk = torch.tensor([remap[int(l)] for l in y])
    ck, _, _ = compute_ave_cost(lg[:, idx], yk, num_l=len(present))
    # A full 20-way argmax counts every prediction landing on an absent country as wrong BY
    # CONSTRUCTION -- that is what `off` measures. Both numbers are reported: acc is the honest
    # open-set number (the model has to know the country is not, say, Bahraini), and the
    # restricted argmax is the number that applies when the candidate label set is known at test
    # time. They can and do move in opposite directions.
    acck = (lg[:, idx].argmax(-1) == yk).float().mean().item() * 100
    return acc, acck, c20, ck, off


def eval_casablanca(model, loader, precision="bf16", confusion=False, tag="Casablanca"):
    return eval_subset_20way(model, loader, precision=precision, confusion=confusion, tag=tag,
                             present=CASA_PRESENT, want_groups=ARGS.confusion_by_program)


def eval_adi17_test(model, loader, precision="bf16", confusion=False):
    """ADI17 test: 17 of the 20 countries, missing exactly MSA/BAH/TUN.

    REPORT-ONLY. It is the closest available proxy for the hidden NADI test set (ADI17 dev is
    demonstrably the NADI validation set), which is precisely why selecting on it is refused at
    argument-parse time and why this runs only after `best` has already been chosen.
    """
    a, a17, c20, c17, off = eval_subset_20way(
        model, loader, precision=precision, confusion=confusion, tag="ADI17_test",
        present=ADI17_COUNTRIES)
    return dict(adi17_acc=a, adi17_acc17=a17, adi17_cavg20=c20, adi17_cavg17=c17,
                adi17_off_set=off)


def eval_madis5(model, loader, precision="bf16"):
    lg, y, doms = _logits_for(model, loader, "MADIS-5", want_domains=True, precision=precision)
    preds = ID2REGION[lg.argmax(-1)].numpy()
    labs = y.numpy()
    doms = np.array(doms)
    assert len(doms) == len(labs), f"domain misalignment {len(doms)} vs {len(labs)}"
    per = {d: float((preds[doms == d] == labs[doms == d]).mean() * 100)
           for d in sorted(set(doms)) if (doms == d).sum()}
    n_by_dom = {d: int((doms == d).sum()) for d in sorted(set(doms))}
    return float((preds == labs).mean() * 100), per, n_by_dom


def _composite_select_metric(m, name):
    """Composite --select-metric values, or None if `name` is not one.

    Selecting purely on casa_acc optimises the OOD probe alone, and selecting purely on id_acc
    optimises the in-domain number alone; both matter here. Higher is always better, so the Cavg
    composite is negated (Cavg is a cost).
    """
    if name == "mean_id_casa":
        if "casa_acc" not in m:
            return None
        return (m["id_acc"] + m["casa_acc"]) / 2.0
    if name == "neg_mean_cavg":
        if "casa_cavg20" not in m:
            return None
        return -(m["id_cavg"] + m["casa_cavg20"]) / 2.0
    return None


def eval_all(model, val_loader, casa_loader, madis_loader, tag="", precision="bf16",
             casa_holdout_loader=None, confusion=False):
    print(f"\n{'='*66}\n  EVAL {tag}\n{'='*66}")
    acc, macro, cost = eval_id(model, val_loader, precision=precision, confusion=confusion)
    print(f"  [ID  val | 20-way]  ACC {acc:6.2f}  (macro {macro:6.2f})  Cavg {cost:.4f}")
    out = {"id_acc": acc, "id_acc_macro": macro, "id_cavg": cost}

    if casa_loader is not None:
        a, a8, c20, c8, off = eval_casablanca(model, casa_loader, precision=precision,
                                              confusion=confusion, tag="Casablanca_select")
        print(f"  [OOD Casa | 20-way]  ACC {a:6.2f}  Cavg20 {c20:.4f}   <-- SELECT ON THIS")
        print(f"  [OOD Casa |  8-way]  ACC {a8:6.2f}  Cavg8  {c8:.4f}   "
              f"(argmax restricted to the {len(CASA_PRESENT)} countries Casablanca contains)")
        print(f"       absent-country preds {off:.2f}% | gap {acc-a:+.2f}")
        if off > 45:
            print("       WARNING: near-chance off-set rate -- the model is guessing.")
        out.update(casa_acc=a, casa_acc8=a8, casa_cavg20=c20, casa_cavg8=c8,
                   off_set=off, gap=acc-a)

    if casa_holdout_loader is not None:
        # Program-disjoint from the selection half above, and never used to pick a checkpoint --
        # this is the number to quote externally.
        ah, a8h, c20h, c8h, offh = eval_casablanca(model, casa_holdout_loader, precision=precision,
                                                   confusion=confusion, tag="Casablanca_holdout")
        print(f"  [OOD Casa HELD OUT]  ACC {ah:6.2f} (20-way)  {a8h:6.2f} (8-way)  "
              f"Cavg20 {c20h:.4f}   <-- REPORT THESE (never selected on)")
        out.update(casa_acc_holdout=ah, casa_acc8_holdout=a8h, casa_cavg20_holdout=c20h,
                   casa_cavg8_holdout=c8h, off_set_holdout=offh, gap_holdout=acc - ah)

    if madis_loader is not None:
        try:
            ov, per, n_by = eval_madis5(model, madis_loader, precision=precision)
            print(f"  [OOD MADIS5 | diagnostic]  overall {ov:6.2f}")
            for d, v in per.items():
                print(f"       {d:<24s} {v:6.2f}  (n={n_by[d]})")
            out["madis_overall"] = ov
        except Exception as e:
            print(f"  [OOD MADIS5] failed -- {type(e).__name__}: {e}")

    print("=" * 66)
    return out



# ============================================================================
# Training
# ============================================================================
def train_one(model_name):
    spec = MODEL_REGISTRY[model_name]
    hf_id = spec["hf_id"]
    precision = PRECISION_OVERRIDE.get(model_name, ARGS.precision)
    run_name = CFG["run"]
    tag = f"{run_name}_{model_name}"
    max_steps, frozen_steps = ARGS.max_steps, ARGS.frozen_steps

    print(f"\n{'#'*66}\n#  {tag}  ({hf_id})\n#  lr={ARGS.lr}  head_lr={ARGS.head_lr}  "
          f"llrd={ARGS.llrd}  precision={precision}  batch={BATCH_SIZE} accum={GRAD_ACCUM}  "
          f"clip_norm={ARGS.clip_norm}  pool={ARGS.pool}\n{'#'*66}")

    wb = None
    if USE_WANDB:
        def _wb_init():
            nonlocal wb
            wb = wandb.init(
                project=ARGS.wandb_project, entity=ARGS.wandb_entity,
                group=WANDB_GROUP, name=tag, reinit=True,
                tags=[model_name, run_name, "llrd", "bf16" if precision == "bf16" else precision],
                config=dict(model=model_name, run=run_name, max_steps=max_steps,
                            frozen_steps=frozen_steps, batch_size=BATCH_SIZE,
                            grad_accum=GRAD_ACCUM, effective_batch=BATCH_SIZE*GRAD_ACCUM*max(1,N_GPU),
                            subset=ARGS.subset, lr=ARGS.lr, head_lr=ARGS.head_lr, llrd=ARGS.llrd,
                            precision=precision, clip_norm=ARGS.clip_norm, pool=ARGS.pool,
                            bn_eval=ARGS.bn_eval, label_smoothing=ARGS.label_smoothing,
                            seed=CFG["seed"]))
        if not _wb_try("init", _wb_init):
            wb = None

    tl, vl, cl, ml, chl, a17l = build_loaders(spec)

    try:
        model = DialectID(spec, num_labels=20, pool=ARGS.pool,
                           compile_encoder=ARGS.compile, compile_mode=ARGS.compile_mode,
                           specaug=ARGS.specaug, specaug_f=ARGS.specaug_freq_mask,
                           specaug_t=ARGS.specaug_time_mask,
                           layer_mix=ARGS.layer_mix, layer_mix_stride=ARGS.layer_mix_stride,
                           lora=ARGS.lora, lora_rank=ARGS.lora_rank,
                           lora_alpha=ARGS.lora_alpha, lora_dropout=ARGS.lora_dropout,
                           lora_targets=ARGS.lora_target_list)
    except Exception as e:
        print(f"SKIP {model_name}: {type(e).__name__}: {e}")
        if wb is not None:
            _wb_try("finish(exit_code=1)", lambda: wb.finish(exit_code=1))
        return None, [], None

    model = model.to(device)
    model.freeze_encoder()
    if ARGS.bn_eval:
        set_bn_eval(model)
    if ARGS.profile_steps > 0:
        model._shape_debug_left = 1
    trainable_report(model)

    opt, sched, ginfo = build_optimizer_and_schedule(model, ARGS, max_steps, frozen_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=ARGS.label_smoothing)
    raw_loss_fn = nn.CrossEntropyLoss()   # unsmoothed, logged only for cross-run comparability
    use_fp16 = precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    tracker = LossTracker(ARGS.loss_csv)
    eval_every = ARGS.eval_every
    plot_every = ARGS.plot_every
    step, history, best, nonfinite_count = 0, [], -1.0, 0
    # Accumulated on-GPU and only synced to Python floats once per optimizer step (not per
    # micro-batch) -- calling .item() every micro-batch forces a CPU/GPU sync that serializes
    # the pipeline and caps utilization well below 100% on a fast GPU with a short per-step
    # workload, even though nothing is actually CPU- or I/O-bound.
    micro_i, total_n = 0, 0
    crop_seconds_accum = 0.0     # CPU-side, see the read site in the loop below
    smooth_accum_t = torch.zeros((), device=device)
    raw_accum_t = torch.zeros((), device=device)
    correct_t = torch.zeros((), device=device)
    nonfinite_t = torch.zeros((), device=device)

    # requires_grad only ever changes at freeze_encoder() (already called above) and at the
    # single unfreeze at step==frozen_steps below -- cache the list instead of walking every
    # parameter in the model on every single optimizer step just to build clip_grad_norm_'s input.
    _trainable_params_cache = [p for p in model.parameters() if p.requires_grad]

    def trainable_params():
        return _trainable_params_cache

    def refresh_trainable_params():
        nonlocal _trainable_params_cache
        _trainable_params_cache = [p for p in model.parameters() if p.requires_grad]

    pbar = tqdm(total=max_steps, desc=f"train {tag}")
    model.train()
    if ARGS.bn_eval:
        set_bn_eval(model)
    opt.zero_grad(set_to_none=True)

    # --profile-steps: phase-timed breakdown of the first N optimizer steps (dataloader wait /
    # H2D / forward / backward / opt step), to verify the launch-overhead diagnosis on the actual
    # box instead of assuming it. Zero cost when disabled (ARGS.profile_steps == 0) -- every use
    # below is gated on `prof`, computed once per micro-batch from the remaining countdown.
    profile_left = ARGS.profile_steps
    if profile_left > 0:
        print(f"  --profile-steps {profile_left}: timing the first {profile_left} optimizer "
              "step(s) by phase.")
    _prof_acc = defaultdict(float)
    _prof_t_prev = time.perf_counter()

    def run_eval_and_maybe_save():
        nonlocal best, wb
        model.eval()
        m = eval_all(model, vl, cl, ml if ARGS.eval_madis else None, tag=f"{tag} step {step}",
                     precision=precision, casa_holdout_loader=chl,
                     # Always on for the final eval: it costs nothing extra (the logits are
                     # already computed) and it is the one artifact that says WHERE the error is.
                     confusion=ARGS.confusion or step >= max_steps)
        m["step"] = step
        # Which layers the task actually uses. This is the whole diagnostic payoff of --layer-mix:
        # weight concentrated near the top means last-layer pooling was fine and the flag bought
        # nothing; weight concentrated in the middle means last-layer pooling was reading the
        # wrong representation all along, and it also says whether freezing the lower layers
        # would help or would delete the features the task depends on.
        if ARGS.layer_mix:
            w = model.layer_mix_weights()
            if w:
                m["layer_mix_argmax"] = int(np.argmax(w))
                m["layer_mix_top3"] = [int(i) for i in np.argsort(w)[::-1][:3]]
                bars = "".join(" .:-=+*#%@"[min(9, int(v / max(w) * 9))] for v in w)
                print(f"  layer-mix weights (0=lowest .. {len(w)-1}=top): {bars}")
                print(f"       peak at layer {m['layer_mix_argmax']}/{len(w)-1}, "
                      f"top-3 {m['layer_mix_top3']}, max weight {max(w):.4f} "
                      f"(uniform would be {1/len(w):.4f})")
        history.append(m)
        _append_progress(dict(m, run=run_name, model=model_name))
        # Selection uses the SELECTION half only; casa_acc_holdout is never allowed to influence
        # which checkpoint is kept, which is what makes it worth reporting.
        #
        # Which metric to select ON matters more than it looks. Casablanca contains 8 of the 20
        # countries, so casa_acc (20-way) and casa_acc8 (restricted argmax) can move in OPPOSITE
        # directions -- at step 800 of the first v3 run, casa_acc rose 51.58 -> 52.02 and saved a
        # new "best" while casa_acc8 on the held-out half fell 66.21 -> 64.08. Selecting on a
        # metric you do not care about quietly hands you the wrong checkpoint.
        #
        # Two COMPOSITES are available for when in-domain and OOD both matter, which is the
        # normal case -- Casablanca is the OOD probe, not the only target. They are computed here
        # rather than added to `m` so that they never appear as if they were measured quantities
        # in the progress/results files.
        sel = _composite_select_metric(m, ARGS.select_metric)
        if sel is None:
            sel = m.get(ARGS.select_metric)
        if sel is None:
            sel = m.get("casa_acc", m["id_acc"])
            if not _SELECT_WARNED:
                print(f"  NOTE: --select-metric {ARGS.select_metric!r} is not present in the eval "
                      "metrics (OOD set missing?) -- falling back to casa_acc/id_acc.")
                _SELECT_WARNED.append(True)
        if wb is not None:
            step_metrics = {f"eval/{k}": v for k, v in m.items() if isinstance(v, (int, float))}
            if not _wb_try("log", lambda: wb.log(step_metrics, step=step)):
                wb = None
        if sel > best:
            best = sel
            torch.save(model.state_dict(), f"best_{tag}.pt")
            print(f"  >> new best OOD {sel:.2f} -- saved best_{tag}.pt")
            if wb is not None:
                if not _wb_try("summary update", lambda: wb.summary.__setitem__("best_casa_acc", best)):
                    wb = None
        model.train()
        if ARGS.bn_eval:
            set_bn_eval(model)

    while step < max_steps:
        for feats, labels, _ in tl:
            if step >= max_steps:
                break
            if feats is None:
                continue

            prof = profile_left > 0
            if prof:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _prof_acc["dataloader_wait"] += t0 - _prof_t_prev

            # Read the batch's actual crop length off the CPU tensor BEFORE the H2D move below --
            # afterwards it lives on the GPU and reading it would cost a sync every step. Logging
            # CFG["crop"] here instead (as this did originally) records a constant 4.0 for the
            # whole run regardless of --crop-set, which makes the CSV silently useless for
            # checking that the crop policy was actually in effect.
            _ps = feats.get("padded_samples")
            crop_seconds_accum += (float(_ps) / SR) if _ps is not None else float(CFG["crop"] or 0)

            feats  = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
            labels = labels.to(device, non_blocking=True)

            if prof:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                _prof_acc["h2d"] += t1 - t0

            with _autocast_ctx(precision):
                logits = model(**feats)
                _assert_aligned(logits, labels, f"{tag} train step {step}")
                loss_smoothed = loss_fn(logits, labels)
                with torch.no_grad():
                    loss_raw = raw_loss_fn(logits, labels)

            if prof:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                _prof_acc["forward"] += t2 - t1

            # A non-finite loss produces a non-finite gradient, which the grad-norm check just
            # below already catches (skipping the optimizer step, weights left unchanged) -- so
            # rather than a `.item()`-forcing sync on every micro-batch just to decide whether to
            # skip it here, tally it on-GPU and let it ride into that existing check. The only
            # behavior change is granularity: with GRAD_ACCUM==1 (the H100 default -- see
            # auto_batch_and_accum) this is exactly the old behavior; with GRAD_ACCUM>1 a bad
            # micro-batch now drops the whole accumulation window instead of just itself.
            with torch.no_grad():
                nonfinite_t += (~torch.isfinite(loss_smoothed)).float()

            scaler.scale(loss_smoothed / GRAD_ACCUM).backward()
            with torch.no_grad():
                smooth_accum_t += loss_smoothed.detach()
                raw_accum_t += loss_raw.detach()
                correct_t += (logits.argmax(-1) == labels).float().sum()
                total_n += labels.numel()
            micro_i += 1

            if prof:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                _prof_acc["backward"] += t3 - t2
                _prof_t_prev = t3

            if micro_i < GRAD_ACCUM:
                continue

            if use_fp16:
                scaler.unscale_(opt)
            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                trainable_params(), ARGS.clip_norm, error_if_nonfinite=False).item()

            if not math.isfinite(grad_norm_val):
                n_bad = int(nonfinite_t.item())
                nonfinite_count += max(n_bad, 1)
                print(f"  [step {step}] non-finite gradient norm ({n_bad} non-finite "
                      f"micro-batch loss(es) this step) -- skipping optimizer step (weights "
                      f"left unchanged); total non-finite so far: {nonfinite_count}")
                opt.zero_grad(set_to_none=True)
                micro_i, total_n, crop_seconds_accum = 0, 0, 0.0
                smooth_accum_t.zero_(); raw_accum_t.zero_(); correct_t.zero_(); nonfinite_t.zero_()
                if prof:
                    _prof_t_prev = time.perf_counter()
                continue

            was_clipped = grad_norm_val > ARGS.clip_norm
            lr_snapshot = {
                "lr_head": opt.param_groups[ginfo["head_idx"]]["lr"] if ginfo["head_idx"] is not None else float("nan"),
                "lr_enc_top": opt.param_groups[ginfo["top_idx"]]["lr"] if ginfo["top_idx"] is not None else float("nan"),
                "lr_enc_bottom": opt.param_groups[ginfo["bottom_idx"]]["lr"] if ginfo["bottom_idx"] is not None else float("nan"),
            }

            if use_fp16:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)

            # Single combined sync for all four accumulated metrics (one .item()-equivalent
            # per optimizer step, not one per micro-batch -- see the accumulator comment above).
            loss_s_v, loss_r_v, acc_v, nonf_v = torch.stack([
                smooth_accum_t / micro_i, raw_accum_t / micro_i, correct_t / total_n, nonfinite_t,
            ]).tolist()
            nonfinite_count += int(nonf_v)
            tracker.add(step=step, loss_smoothed=loss_s_v, loss_raw_ce=loss_r_v,
                        grad_norm_preclip=grad_norm_val, was_clipped=int(was_clipped),
                        batch_acc=acc_v, n_skipped_micro=nonfinite_count,
                        mean_clip_seconds=crop_seconds_accum / max(1, micro_i), **lr_snapshot)
            micro_i, total_n, crop_seconds_accum = 0, 0, 0.0
            smooth_accum_t.zero_(); raw_accum_t.zero_(); correct_t.zero_(); nonfinite_t.zero_()

            if prof:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t4 = time.perf_counter()
                _prof_acc["opt_step"] += t4 - t3
                _prof_t_prev = t4
                profile_left -= 1
                if profile_left == 0:
                    total = sum(_prof_acc.values()) or 1e-9
                    print(f"\n  [profile] phase breakdown over the first {ARGS.profile_steps} "
                          f"optimizer step(s) (wall seconds, % of total):")
                    for k, v in _prof_acc.items():
                        print(f"    {k:<16s} {v:8.4f}s  ({100*v/total:5.1f}%)")
                    print(f"    {'TOTAL':<16s} {total:8.4f}s")
                    print("  a small 'forward'+'backward' share relative to 'dataloader_wait' "
                          "means the GPU is launch-overhead-bound, not data-starved; a large "
                          "'dataloader_wait' share means the input pipeline is next to attack.\n")

            if step % 100 == 0:
                pbar.set_postfix(loss=f"{tracker.rows[-1]['loss_smoothed']:.3f}",
                                 lr=f"{lr_snapshot['lr_enc_top']:.2e}")

            if step == frozen_steps:
                n = model.unfreeze_encoder()
                if ARGS.bn_eval:
                    set_bn_eval(model)
                refresh_trainable_params()
                print(f"\n  unfroze encoder: {n} layers now trainable "
                      f"(LLRD {ARGS.llrd}, re-warming over {ARGS.enc_warmup} steps)")
                trainable_report(model)

            if step % plot_every == 0 or step == max_steps:
                torch.cuda.empty_cache() if device.type == "cuda" else None
                df = tracker.dataframe()
                fig_paths = render_plots(df, history, ARGS.plots_dir, tag, ARGS.clip_norm,
                                          unfreeze_step=frozen_steps,
                                          label_smoothing=ARGS.label_smoothing,
                                          n_classes=len(COUNTRIES))
                write_spike_report(df, ARGS.spike_report)
                check_divergence(df, frozen_steps)
                if wb is not None and fig_paths is not None:
                    _, latest_path = fig_paths
                    if not _wb_try("plot image log",
                                    lambda: wb.log({"diagnostics/loss_plot": wandb.Image(latest_path)}, step=step)):
                        wb = None

            if step % eval_every == 0 or step == max_steps:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                run_eval_and_maybe_save()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    pbar.close()
    tracker.flush()

    # One extra pass with test-time augmentation, on freshly built eval loaders. Deliberately NOT
    # folded into the per-eval-every loop: TTA multiplies eval wall-time by --tta, and with evals
    # every --eval-every steps that would dominate a short run. Recorded with a _tta suffix so it
    # never silently replaces the comparable non-TTA numbers.
    # ADI17 test, scored HERE and only here: after the training loop, after `best` is fixed, and
    # after the checkpoint has been chosen. That placement is the enforcement, not a convention --
    # a metric computed once selection is over cannot bias selection, which is the whole reason it
    # is safe to score against a proxy for the hidden test set. --select-metric additionally
    # refuses every adi17_* name at argument-parse time.
    adi17_metrics = {}
    if a17l is not None:
        try:
            print(f"\n  final eval on ADI17 test ({len(adi17_test_ds)} clips) -- REPORT ONLY, "
                  "never selected on")
            model.eval()
            adi17_metrics = eval_adi17_test(model, a17l, precision=precision, confusion=True)
            print(f"  [ADI17 test | 20-way]  ACC {adi17_metrics['adi17_acc']:6.2f}  "
                  f"Cavg20 {adi17_metrics['adi17_cavg20']:.4f}")
            print(f"  [ADI17 test | 17-way]  ACC {adi17_metrics['adi17_acc17']:6.2f}  "
                  f"Cavg17 {adi17_metrics['adi17_cavg17']:.4f}  "
                  f"(argmax restricted to the 17 countries ADI17 contains)")
            print(f"       predicted into {'/'.join(ADI17_ABSENT)}: "
                  f"{adi17_metrics['adi17_off_set']:.2f}%  -- rising here means a 17-class "
                  "training source\n       over-shrank those three priors; see --sampler-alpha")
            if history:
                _idacc = history[-1].get("id_acc")
                if _idacc is not None:
                    print(f"       id_acc {_idacc:.2f} vs adi17_acc "
                          f"{adi17_metrics['adi17_acc']:.2f} ({adi17_metrics['adi17_acc']-_idacc:+.2f})"
                          " -- both in-domain, so a large gap points at the eval, not the model")
        except Exception as e:
            print(f"  ADI17 test eval failed ({type(e).__name__}: {e}) -- every other number "
                  "above is unaffected.")

    tta_metrics = {}
    if ARGS.tta and ARGS.tta > 1 and history:
        try:
            print(f"\n  final eval with --tta {ARGS.tta} (extra pass, eval-side only)")
            _, vl_t, cl_t, ml_t, chl_t, _ = build_loaders(spec, tta=ARGS.tta)
            model.eval()
            m_tta = eval_all(model, vl_t, cl_t, ml_t if ARGS.eval_madis else None,
                             tag=f"{tag} step {step} +TTA{ARGS.tta}", precision=precision,
                             casa_holdout_loader=chl_t, confusion=False)
            tta_metrics = {f"{k}_tta": v for k, v in m_tta.items() if k != "step"}
            base = history[-1]
            for key in ("id_acc", "casa_acc", "casa_acc8", "casa_acc_holdout",
                        "casa_acc8_holdout", "madis_overall"):
                if key in base and key in m_tta:
                    print(f"    TTA delta {key:<20s} {base[key]:6.2f} -> {m_tta[key]:6.2f}  "
                          f"({m_tta[key]-base[key]:+.2f})")
        except Exception as e:
            print(f"  TTA eval failed ({type(e).__name__}: {e}) -- the non-TTA numbers above "
                  "are unaffected.")

    # final plot + spike report even if max_steps isn't a multiple of plot_every
    df = tracker.dataframe()
    render_plots(df, history, ARGS.plots_dir, tag, ARGS.clip_norm, unfreeze_step=frozen_steps,
                 label_smoothing=ARGS.label_smoothing, n_classes=len(COUNTRIES))
    write_spike_report(df, ARGS.spike_report)

    rec = None
    if history:
        rec = dict(history[-1])
        rec.update(run=run_name, model=model_name, seed=CFG["seed"], lr=ARGS.lr,
                    head_lr=ARGS.head_lr, llrd=ARGS.llrd, frozen_steps=frozen_steps,
                    precision=precision, pool=ARGS.pool, batch_size=BATCH_SIZE,
                    grad_accum=GRAD_ACCUM, effective_batch=BATCH_SIZE*GRAD_ACCUM*max(1,N_GPU),
                    clip_norm=ARGS.clip_norm, label_smoothing=ARGS.label_smoothing,
                    bn_eval=ARGS.bn_eval, crop=ARGS.crop,
                    crop_set=",".join(str(c) for c in ARGS.crop_list) or "",
                    aug=",".join(ARGS.aug_list) or "", specaug=ARGS.specaug,
                    layer_mix=ARGS.layer_mix, lora=ARGS.lora,
                    lora_rank=(ARGS.lora_rank if ARGS.lora else None),
                    lora_modules=(ginfo.get("n_lora") if ARGS.lora else None),
                    weight_decay=ARGS.weight_decay, tta=ARGS.tta,
                    datasets=",".join(d["name"] for d in DATASETS_USED),
                    dataset_rows=json.dumps({d["name"]: d["clips"] for d in DATASETS_USED}),
                    sampler_alpha=ARGS.sampler_alpha,
                    select_metric=ARGS.select_metric,
                    extra_train_data=",".join(ARGS.extra_train_list) or "",
                    extra_train_rows=EXTRA_TRAIN_ROWS,
                    layer_mix_weights=(model.layer_mix_weights() if ARGS.layer_mix else None),
                    status="ok", **adi17_metrics, **tta_metrics)
        if wb is not None:
            def _wb_final_summary():
                for k, v in rec.items():
                    if isinstance(v, (int, float)):
                        wb.summary[f"final_{k}"] = v
            _wb_try("final summary", _wb_final_summary)

    if wb is not None:
        _wb_try("finish", wb.finish)

    gc.collect(); torch.cuda.empty_cache()
    return model, history, rec



# ============================================================================
# Run loop -- single model, OOM-retry by halving batch size (grad_accum auto-adjusts to keep
# the effective batch roughly constant).
# ============================================================================
def _run_model_with_retry(model_name):
    global BATCH_SIZE, GRAD_ACCUM
    orig_batch, orig_accum = BATCH_SIZE, GRAD_ACCUM
    # Full descending ladder down to a floor of 16, rather than cohere_train.py's single
    # halving: micro-batch 128 has much further to fall before it fits.
    # Halve all the way down to 1, not to a floor of 16. The h100 script floors at 16 because it
    # starts from micro-batch 128 and has far to fall -- but that floor silently disables retry
    # ENTIRELY for any starting batch <= 16, which is exactly what --crop-set produces (a 12s max
    # crop fails auto_batch_and_accum's `max_crop <= 8.0` short-crop test, so it picks 16). An OOM
    # on the first step would then kill the whole run with no fallback, which is the single most
    # expensive way for this to fail.
    attempts = [orig_batch]
    b = orig_batch
    while b > 1:
        b = max(1, b // 2)
        if b == attempts[-1]:
            break
        attempts.append(b)

    for attempt_i, bs in enumerate(attempts):
        BATCH_SIZE = bs
        GRAD_ACCUM = max(1, round(ARGS.effective_batch / (bs * max(1, N_GPU))))
        is_last_attempt = attempt_i == len(attempts) - 1
        try:
            model, hist, rec = train_one(model_name)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on {model_name} at batch {bs}")
            _close_dangling_wandb()
            gc.collect(); torch.cuda.empty_cache()
            if is_last_attempt:
                BATCH_SIZE, GRAD_ACCUM = orig_batch, orig_accum
                record_result(dict(run=CFG["run"], model=model_name, status="oom",
                                    error=f"OOM at batch<={bs}"))
                return
            print(f"  retrying {model_name} at batch {attempts[attempt_i + 1]}")
            continue
        except Exception as e:
            print(f"  FAILED {model_name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            _close_dangling_wandb()
            BATCH_SIZE, GRAD_ACCUM = orig_batch, orig_accum
            record_result(dict(run=CFG["run"], model=model_name, status="failed",
                                error=f"{type(e).__name__}: {e}"))
            gc.collect(); torch.cuda.empty_cache()
            return
        else:
            if bs != orig_batch:
                print(f"  {model_name}: succeeded at batch {bs} (requested {orig_batch}) -- "
                      f"effective batch held at {ARGS.effective_batch} via grad_accum={GRAD_ACCUM}, "
                      "so this is NOT the batch-128 experiment that was intended; treat results "
                      "with that in mind.")
            BATCH_SIZE, GRAD_ACCUM = orig_batch, orig_accum
            if rec is not None:
                record_result(rec)
            else:
                record_result(dict(run=CFG["run"], model=model_name, status="failed",
                                    error="no history recorded (load failure)"))
            if model is not None:
                del model
            gc.collect(); torch.cuda.empty_cache()
            return


def run_bench():
    plan = list(MODEL_REGISTRY)
    done = already_done(ARGS.results)
    total = len(plan)
    print(f"\nrun plan ({total} model(s)): {plan}")
    if done:
        print(f"resume: {len(done)} run(s) already in {ARGS.results}, will skip")

    for i, model_name in enumerate(plan, 1):
        if (CFG["run"], model_name) in done:
            print(f"\n[{i}/{total}] SKIP (done): {model_name}")
            continue
        print(f"\n[{i}/{total}] {model_name}")
        _run_model_with_retry(model_name)

    _final_report()


def eval_checkpoint(path):
    """Evaluate a saved checkpoint and exit -- no optimizer, no training, one pass per eval set.

    This is the analysis step to run BEFORE spending more training compute: a headline accuracy
    says how much error there is, this says where it is, which is what decides whether the next
    run should chase optimization, regularization, or nothing at all.
    """
    model_name = BENCH_ORDER[0]
    spec = MODEL_REGISTRY[model_name]
    precision = PRECISION_OVERRIDE.get(model_name, ARGS.precision)
    print(f"\n{'#'*66}\n#  EVAL-ONLY: {path}\n{'#'*66}")
    if not os.path.exists(path):
        print(f"FAIL: no such checkpoint: {path}")
        return 1

    _, vl, cl, ml, chl, a17l = build_loaders(spec, tta=ARGS.tta)
    # compile_encoder is off: this is a single eval pass, so compilation would cost more than it
    # saves, and the train-only compile path is never exercised here anyway.
    model = DialectID(spec, num_labels=20, pool=ARGS.pool, specaug=False,
                       layer_mix=ARGS.layer_mix, layer_mix_stride=ARGS.layer_mix_stride,
                       lora=ARGS.lora, lora_rank=ARGS.lora_rank,
                       lora_alpha=ARGS.lora_alpha, lora_dropout=ARGS.lora_dropout,
                       lora_targets=ARGS.lora_target_list).to(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  WARNING: load_state_dict mismatch -- {len(missing)} missing, "
              f"{len(unexpected)} unexpected key(s).")
        for k in list(missing)[:5]:
            print(f"    missing:    {k}")
        for k in list(unexpected)[:5]:
            print(f"    unexpected: {k}")
        if len(missing) > 10:
            print("  That many missing keys means this checkpoint does not match the current "
                  "--pool / architecture flags. Re-run with the flags the checkpoint was "
                  "trained under.")
    else:
        print("  checkpoint loaded, all keys matched")
    if ARGS.bn_eval:
        set_bn_eval(model)

    m = eval_all(model, vl, cl, ml if ARGS.eval_madis else None,
                 tag=f"checkpoint {os.path.basename(path)}", precision=precision,
                 casa_holdout_loader=chl, confusion=True)
    # No selection happens in this path at all, so ADI17 test is unambiguously safe here -- and
    # this is the cheapest way to get the baseline number the data changes are measured against.
    if a17l is not None:
        try:
            print(f"\n  ADI17 test ({len(adi17_test_ds)} clips) -- report only")
            m.update(eval_adi17_test(model, a17l, precision=precision, confusion=True))
            print(f"  [ADI17 test | 20-way]  ACC {m['adi17_acc']:6.2f}  "
                  f"Cavg20 {m['adi17_cavg20']:.4f}")
            print(f"  [ADI17 test | 17-way]  ACC {m['adi17_acc17']:6.2f}  "
                  f"Cavg17 {m['adi17_cavg17']:.4f}")
            print(f"       predicted into {'/'.join(ADI17_ABSENT)}: {m['adi17_off_set']:.2f}%")
        except Exception as e:
            print(f"  ADI17 test eval failed ({type(e).__name__}: {e})")
    out = os.path.join(ARGS.plots_dir, "eval_checkpoint.json")
    os.makedirs(ARGS.plots_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(_jsonable(dict(m, checkpoint=path)), f, indent=2)
    print(f"\n  metrics -> {out}")
    print("  confusion matrices -> " + os.path.join(ARGS.plots_dir, "confusion_*.csv"))
    return 0


def preflight():
    print("\n=== PREFLIGHT ===")
    for m in list(MODEL_REGISTRY):
        spec = MODEL_REGISTRY[m]
        try:
            get_feature_extractor(spec)
            net = DialectID(spec, num_labels=20, pool=ARGS.pool,
                             layer_mix=ARGS.layer_mix, layer_mix_stride=ARGS.layer_mix_stride,
                       lora=ARGS.lora, lora_rank=ARGS.lora_rank,
                       lora_alpha=ARGS.lora_alpha, lora_dropout=ARGS.lora_dropout,
                       lora_targets=ARGS.lora_target_list)
            del net
            gc.collect(); torch.cuda.empty_cache()
            print(f"  OK  {m:16s} feature-extractor + weights load")
        except Exception as e:
            print(f"  FAIL {m}: {type(e).__name__}: {e}")
    print(f"  train {len(train_ds)} | val {len(val_ds)} | "
          f"casa-select {0 if casa_ds is None else len(casa_ds)} | "
          f"casa-holdout {0 if casa_holdout_ds is None else len(casa_holdout_ds)} | "
          f"madis {0 if madis5_ds is None else len(madis5_ds)} | "
          f"adi17-test {0 if adi17_test_ds is None else len(adi17_test_ds)}")
    print_dataset_manifest()
    print("PREFLIGHT PASSED -- safe to run training\n" if casa_ds is not None
          else "PREFLIGHT: Casablanca missing -- OOD selection will fall back\n")


def _final_report():
    rows = _load_jsonl(ARGS.results)
    if not rows:
        print("\nno results recorded.")
        return
    df = pd.DataFrame(rows)
    cols = [c for c in RESULT_COLS if c in df]
    out = df[cols]
    if "casa_acc" in out:
        out = out.sort_values("casa_acc", ascending=False)
    print("\n" + "=" * 70)
    print(f"  FINAL cohere-ar, effective batch {ARGS.effective_batch}, "
          f"layer_mix={ARGS.layer_mix} lora={ARGS.lora} (cohere_train_v4.py)")
    print("=" * 70)
    print(out.round(4).to_string(index=False))
    out.to_csv("train_table.csv", index=False)
    print("\nwrote train_table.csv")

    # Repeated at the end so the last thing in a run log answers "what data was this?" -- the
    # banner from the start of the run is thousands of lines up by now.
    print_dataset_manifest()

    for orig_path, orig_label in [("cohere_train_h100_results.jsonl", "cohere_train_h100.py (batch 128, full fine-tune)"),
                                   ("cohere_train_results.jsonl", "cohere_train.py (effective batch 32 baseline)"),
                                   ("cohere_bench_results.jsonl", "cohere_bench.py (retuned, but schedule-broken)"),
                                   ("bench_results.jsonl", "models_bench.py (original diverged run)")]:
        if not os.path.exists(orig_path):
            continue
        orig_rows = [r for r in _load_jsonl(orig_path) if r.get("model") == "cohere-ar"]
        if not orig_rows:
            continue
        print("\n" + "=" * 70)
        print(f"  BEFORE ({orig_label}) vs AFTER (this script)")
        print("=" * 70)
        before = pd.DataFrame(orig_rows)
        before_cols = [c for c in RESULT_COLS if c in before]
        print(before[before_cols].round(4).to_string(index=False))
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    if ARGS.eval_checkpoint:
        sys.exit(eval_checkpoint(ARGS.eval_checkpoint))
    elif ARGS.preflight:
        preflight()
    else:
        run_bench()
