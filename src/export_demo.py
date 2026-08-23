#!/usr/bin/env python
"""
Demo data exporter for the project landing page.

Runs a trained checkpoint over a handful of real clips and writes everything the
static page needs to run a guess-then-reveal demo with NO server and NO live model:

    <out>/demo.json          clip metadata + the full 20-way probability vector per clip
    <out>/audio/<id>.wav     the clip audio, 16 kHz mono, trimmed to --clip-seconds

The page reads demo.json, plays the audio, lets the visitor pick a country, then
reveals the model's distribution. Nothing is computed in the browser, so the demo
cannot fail live: no cold start, no GPU, no gated-token 404 in front of an audience.


WHY THIS SCRIPT IMPORTS submit.py INSTEAD OF COPYING IT
-------------------------------------------------------
The number on the page has to be the number the model actually produces. submit.py
already owns the only inference path that has been verified against the trainer
(its --verify pass re-scores ADI20 validation through its own code), and it already
solves architecture inference, the scatter-by-row-index alignment guarantee, and TTA
merging. Reimplementing any of that here would give the page a second, unverified
code path that could drift -- and a demo that disagrees with the submission is worse
than no demo.

submit.py parses sys.argv at import (`ARGS = get_args()` at module scope), so this
script builds a valid submit.py argv, imports the module, and then overrides the
handful of ARGS fields it needs. That is the whole trick.

The one thing that genuinely cannot be imported is load_casablanca: it lives in
train.py, which parses argv AND downloads the full ADI20 corpus at import time. It is
copied below with a `v5:<line>` reference, the same convention submit.py uses.


CLIP SELECTION IS DELIBERATE, NOT RANDOM
----------------------------------------
A demo that only shows wins is an advertisement. --mix picks a curated spread:
confident-correct, confident-WRONG, and genuine close calls. The wrong ones are the
reason a judge believes the right ones. Every clip's `bucket` is recorded in
demo.json so the page can label them honestly rather than hiding the failures.

Casablanca covers only 8 of the 20 countries (ALG EGY JOR MAU MOR PAL UAE YEM), so
`candidates` is written per clip: a human guessing on a broadcast clip is choosing
among 8, and the page should say so rather than implying a 1-in-20 guess.


USAGE (run from the repo root)
-----
    # broadcast holdout only -- match the --casa-val-frac the checkpoint was trained with
    python src/export_demo.py --checkpoint best_v3_lm_casatr_casavf20_6k_cohere-ar.pt \\
        --casa-val-frac 0.2 --n-casa 8 --n-adi 4 --out site/demo

    # quick smoke of the whole export on a few clips
    python src/export_demo.py --checkpoint best.pt --n-casa 2 --n-adi 2 --pool-size 40

Credentials: $HF_TOKEN, same as train.py and submit.py. Nothing is read from a file.
"""

import argparse
import json
import math
import os
import sys
import wave
from datetime import datetime, timezone


# ---------------------------------------------------------------------------------
# Args. Parsed BEFORE submit.py is imported, because importing it consumes sys.argv.
# ---------------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Export real clips + model probabilities for the landing-page demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="checkpoint to run. Use the one the page's headline number came from, "
                        "otherwise the demo and the results table describe different models.")
    p.add_argument("--out", default="site/demo", metavar="DIR",
                   help="output directory. demo.json lands here, audio in <out>/audio/.")

    p.add_argument("--n-casa", type=int, default=8, metavar="N",
                   help="clips to export from the Casablanca HELD-OUT half (broadcast audio, "
                        "program-disjoint from training). These are the interesting ones.")
    p.add_argument("--n-adi", type=int, default=4, metavar="N",
                   help="clips to export from ADI20 validation (in-domain). Included so the "
                        "page can show the domain gap as something a visitor hears, not just "
                        "a number in a table.")
    p.add_argument("--pairs", type=int, default=4, metavar="N",
                   help="matched pairs to export: the SAME country heard once in ADI-20 "
                        "(clean, read) and once in Casablanca (broadcast). Casablanca covers "
                        "8 of the 20 countries, so N is capped by that overlap. These are what "
                        "make the domain gap audible rather than merely tabulated.")
    p.add_argument("--pool-size", type=int, default=400, metavar="N",
                   help="how many clips per source to actually run the model over before "
                        "selecting from them. Larger = better spread of confident/wrong/close "
                        "cases to choose from, and more GPU time.")

    p.add_argument("--casa-val-frac", type=float, default=0.5, metavar="F",
                   help="MUST match the --casa-val-frac the checkpoint was trained with. If it "
                        "does not, the 'held-out' clips exported here may have been in that "
                        "run's training set and the demo silently becomes dishonest.")
    p.add_argument("--casa-select-frac", type=float, default=0.5, metavar="F",
                   help="MUST match the run's --casa-select-frac, for the same reason.")
    p.add_argument("--seed", type=int, default=42, metavar="N",
                   help="MUST match the run's seed -- the program-disjoint split is drawn from "
                        "it. train.py's SEED is 42.")

    p.add_argument("--clip-seconds", type=float, default=8.0, metavar="S",
                   help="trim exported audio to this length. The model still scores the clip it "
                        "was given in full; this only bounds the file the browser downloads.")
    p.add_argument("--max-total-mb", type=float, default=12.0, metavar="MB",
                   help="abort if the exported audio exceeds this. A landing page that ships "
                        "40 MB of WAV is a landing page nobody waits for.")

    p.add_argument("--tta-seconds", type=float, default=None, metavar="S",
                   help="model input window. Default: the longest --crop-set value the "
                        "checkpoint trained on (12s), which dominates activation memory. "
                        "Shortening it is the main lever on a small card; stay within the "
                        "crop range the checkpoint actually saw during training.")
    p.add_argument("--tta", type=int, default=0, metavar="N",
                   help="multi-crop test-time averaging, passed through to submit.py. Leave at "
                        "0 unless the reported number used TTA.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--pool", default=None, choices=["mean", "mean_std", "attn_stat"],
                   help="override pooling. Normally inferred from the checkpoint.")
    p.add_argument("--skip-install", action="store_true",
                   help="passed through to submit.py's dependency step.")

    return p.parse_args()


ARGS = get_args()

# --- import submit.py under an argv it will accept -------------------------------
# Everything below this point depends on it, so it happens before any other work.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
_saved_argv = sys.argv
sys.argv = ["submit.py", "--checkpoint", ARGS.checkpoint, "--tag", "demo", "--no-verify"]
if ARGS.skip_install:
    sys.argv.append("--skip-install")

import submit as S                                                    # noqa: E402

sys.argv = _saved_argv

S.ARGS.tta = ARGS.tta
if ARGS.tta_seconds is not None:
    S.ARGS.tta_seconds = ARGS.tta_seconds
S.ARGS.batch_size = ARGS.batch_size
S.ARGS.num_workers = ARGS.num_workers
S.ARGS.precision = ARGS.precision
S.ARGS.pool = ARGS.pool

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
from datasets import concatenate_datasets, load_dataset               # noqa: E402

COUNTRIES = S.COUNTRIES
SR = S.SR


# ---------------------------------------------------------------------------------
# Casablanca -- copied from train.py:3337,3377. See module docstring for why it is
# copied rather than imported. Kept byte-faithful to the split logic: if this drifts,
# "held out" stops meaning held out.
# ---------------------------------------------------------------------------------
CASA_CONFIGS = {"Algeria": "ALG", "Egypt": "EGY", "Jordan": "JOR", "Mauritania": "MAU",
                "Morocco": "MOR", "Palestine": "PAL", "UAE": "UAE", "Yemen": "YEM"}


def load_casablanca_holdout(seed, val_frac, select_frac):               # v5:3377
    """Returns the HELD-OUT half only, split program-disjointly, identical to training."""
    parts = []
    for cfg, code_ in CASA_CONFIGS.items():
        try:
            d = load_dataset("UBC-NLP/Casablanca", cfg)
        except Exception as e:
            print(f"  [{cfg}] load failed -- {type(e).__name__}")
            continue
        for split in [s for s in ("validation", "test") if s in d]:
            x = d[split].add_column("dialect", [code_] * len(d[split]))
            keep = [c for c in ("audio", "dialect", "seg_id") if c in x.column_names]
            parts.append(x.select_columns(keep))
    if not parts:
        print("FAIL: Casablanca loaded nothing. Check HF_TOKEN has gated access to "
              "UBC-NLP/Casablanca.")
        sys.exit(2)

    merged = concatenate_datasets(parts)
    segs = merged["seg_id"] if "seg_id" in merged.column_names \
        else [str(i) for i in range(len(merged))]
    prog = [str(s).split("_")[0] for s in segs]
    merged = merged.add_column("program", prog)

    rng = np.random.default_rng(seed)
    by_country = {}
    for i, (p, l) in enumerate(zip(prog, merged["dialect"])):
        by_country.setdefault(l, {}).setdefault(p, []).append(i)

    hold_idx = []
    for country, progs in by_country.items():
        plist = sorted(progs)
        rng.shuffle(plist)
        n_kept = max(1, int(len(plist) * val_frac))
        kept = plist[:n_kept]
        # The rng is NOT consumed for the discarded/recovered programs -- train.py is
        # explicit that this keeps the call sequence identical so `sel`/`hold` come out
        # bit-identical with or without --train-on-casa. Do not "tidy" this.
        n_sel = max(1, int(round(len(kept) * select_frac)))
        if select_frac < 1.0 and len(kept) >= 2:
            n_sel = min(n_sel, len(kept) - 1)
        for p in kept[n_sel:]:
            hold_idx.extend(progs[p])

    hold = merged.select(sorted(hold_idx))
    print(f"  Casablanca held-out: {len(hold)} clips, "
          f"{len(set(hold['dialect']))} countries, {len(set(hold['program']))} programs")
    return hold


# ---------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------
def pick_clips(probs, labels, n, rng):
    """Choose a spread of confident-correct, confident-wrong and close calls.

    Returns [(row_index, bucket_name)]. Buckets are recorded in demo.json so the page
    can label a failure as a failure instead of quietly omitting it.
    """
    top2 = np.argsort(-probs, axis=1)[:, :2]
    pred = top2[:, 0]
    conf = probs[np.arange(len(probs)), pred]
    margin = conf - probs[np.arange(len(probs)), top2[:, 1]]
    correct = pred == labels

    quotas = [("confident-correct", max(1, round(n * 0.40))),
              ("confident-wrong",   max(1, round(n * 0.25))),
              ("close-call",        max(1, round(n * 0.35)))]

    def rank(bucket):
        if bucket == "confident-correct":
            cand = np.flatnonzero(correct & (conf >= 0.60))
            return cand[np.argsort(-conf[cand])] if len(cand) else cand
        if bucket == "confident-wrong":
            cand = np.flatnonzero(~correct & (conf >= 0.45))
            return cand[np.argsort(-conf[cand])] if len(cand) else cand
        cand = np.flatnonzero(margin <= 0.25)
        return cand[np.argsort(margin[cand])] if len(cand) else cand

    chosen, used, seen_country = [], set(), {}
    for bucket, quota in quotas:
        taken = 0
        for i in rank(bucket):
            if taken >= quota:
                break
            i = int(i)
            if i in used:
                continue
            # Spread across countries so the demo is not four clips of the same dialect.
            c = int(labels[i])
            if seen_country.get(c, 0) >= 2:
                continue
            used.add(i)
            seen_country[c] = seen_country.get(c, 0) + 1
            chosen.append((i, bucket))
            taken += 1

    # Backfill from whatever is left if a bucket could not be filled -- a checkpoint with
    # no confident errors in the pool is a good problem, not a reason to abort.
    if len(chosen) < n:
        rest = [i for i in rng.permutation(len(probs)) if int(i) not in used]
        for i in rest[:n - len(chosen)]:
            chosen.append((int(i), "correct" if correct[int(i)] else "wrong"))

    rng.shuffle(chosen)
    return chosen[:n]


# ---------------------------------------------------------------------------------
# Audio out
# ---------------------------------------------------------------------------------
def write_wav(path, wav, sr=SR):
    """16-bit PCM mono. wave is stdlib -- no soundfile/scipy dependency for the export."""
    w = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    if peak > 0:
        w = w / peak * 0.97          # normalise: broadcast and ADI20 clips differ wildly in level
    pcm = np.clip(w * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    return os.path.getsize(path)


def _subset(ds, keep):
    """Casablanca arrives as an Arrow Dataset; ADI20 arrives as a plain list of
    already-decoded rows (submit.py._load_val_sample). Only the former has .select."""
    return ds.select(keep) if hasattr(ds, "select") else [ds[i] for i in keep]


def _field(sample, key):
    """Read a column off one row without asking the container what columns it has.
    A list has no .column_names, and neither source is guaranteed to carry `program`."""
    try:
        v = sample.get(key) if hasattr(sample, "get") else sample[key]
    except (KeyError, IndexError, TypeError):
        return None
    return None if v is None else str(v)


def run_source(model, fe, ds, name, source_label, n_want, rng, out_dir, manifest):
    if n_want <= 0:
        return
    if len(ds) > ARGS.pool_size:
        keep = sorted(rng.choice(len(ds), size=ARGS.pool_size, replace=False).tolist())
        ds = _subset(ds, keep)

    print(f"\n[{name}] scoring {len(ds)} clips to select {n_want} from")
    logp, labels, missed = S.predict(model, ds, fe, len(ds), desc=name, with_labels=True)
    if missed:
        print(f"  {len(missed)} clip(s) undecodable -- excluded from selection")

    probs = torch.exp(logp).numpy()
    labels = labels.numpy()
    ok = np.array([i not in missed and labels[i] >= 0 for i in range(len(ds))])
    idx_ok = np.flatnonzero(ok)
    probs_ok, labels_ok = probs[idx_ok], labels[idx_ok]

    acc = float((probs_ok.argmax(1) == labels_ok).mean() * 100)
    print(f"  pool accuracy: {acc:.2f}% over {len(idx_ok)} clips  "
          f"(sanity check against the run's reported number)")

    present = sorted({COUNTRIES[int(c)] for c in labels_ok})
    picks = pick_clips(probs_ok, labels_ok, n_want, rng)

    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    for k, (j, bucket) in enumerate(picks):
        row = int(idx_ok[j])
        sample = ds[row]
        wav = S._wav(sample)
        if wav is None:
            print(f"  skipping row {row}: undecodable at export time")
            continue
        wav = wav.numpy()[: int(ARGS.clip_seconds * SR)]

        cid = f"{name}-{k:02d}"
        rel = f"audio/{cid}.wav"
        size = write_wav(os.path.join(audio_dir, f"{cid}.wav"), wav)

        p = probs_ok[j]
        order = np.argsort(-p)[:5]
        true_i, pred_i = int(labels_ok[j]), int(p.argmax())

        manifest.append({
            "id": cid,
            "audio": rel,
            "bytes": size,
            "duration_s": round(len(wav) / SR, 2),
            "source": name,
            "source_label": source_label,
            "program": _field(sample, "program"),
            "true": COUNTRIES[true_i],
            "pred": COUNTRIES[pred_i],
            "correct": bool(true_i == pred_i),
            "confidence": round(float(p[pred_i]), 4),
            "bucket": bucket,
            "probs": [round(float(x), 5) for x in p],
            "top5": [[COUNTRIES[int(i)], round(float(p[int(i)]), 4)] for i in order],
            "entropy": round(float(-(p * np.log(np.clip(p, 1e-12, 1))).sum()), 3),
            "candidates": present,
        })
        print(f"  {cid}  true={COUNTRIES[true_i]:<4s} pred={COUNTRIES[pred_i]:<4s} "
              f"p={p[pred_i]:.2f}  [{bucket}]  {size/1024:.0f} KB")


def build_pairs(manifest, want):
    """Match ADI-20 and Casablanca clips of the SAME country.

    The domain gap is a claim about one dialect sounding different in two settings, so it
    is only honest to demonstrate it on the same dialect. Casablanca carries 8 of the 20
    countries, which caps how many pairs can exist at all.
    """
    if want <= 0:
        return []
    by = {}
    for c in manifest:
        by.setdefault((c["source"], c["true"]), []).append(c)

    out = []
    for country in sorted({c["true"] for c in manifest}):
        studio = by.get(("adi", country))
        broadcast = by.get(("casa", country))
        if not studio or not broadcast:
            continue
        out.append({"country": country,
                    "studio": studio[0]["id"],
                    "broadcast": broadcast[0]["id"]})
        if len(out) >= want:
            break

    if not out:
        print("\n  NOTE: no matched pairs -- no country appears in both sources at the "
              "sample sizes exported. Raise --n-adi/--n-casa or --pool-size; the page will "
              "hide the studio-vs-broadcast section rather than fake it.")
    else:
        print("\n  matched pairs: " + ", ".join(p["country"] for p in out))
    return out


def main():
    print("=" * 70)
    print("DEMO EXPORT")
    print("=" * 70)

    out_dir = os.path.abspath(ARGS.out)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(ARGS.seed)

    S._hf_login()

    print("\nLoading checkpoint")
    state = S.load_state(ARGS.checkpoint)
    arch = S.infer_arch(state)
    model, pool, layer_mix = S.build_model(state, arch)

    from transformers import AutoProcessor
    fe = AutoProcessor.from_pretrained(S.HF_ID, trust_remote_code=True)

    manifest = []

    if ARGS.n_casa > 0:
        print("\nLoading Casablanca (held-out half only)")
        casa = load_casablanca_holdout(ARGS.seed, ARGS.casa_val_frac, ARGS.casa_select_frac)
        run_source(model, fe, casa, "casa",
                   "Broadcast audio (Casablanca), held-out program", ARGS.n_casa,
                   rng, out_dir, manifest)

    if ARGS.n_adi > 0:
        print("\nLoading ADI20 validation")
        adi = S._load_val_sample(ARGS.pool_size)
        run_source(model, fe, adi, "adi",
                   "ADI-20 validation (in-domain)", ARGS.n_adi,
                   rng, out_dir, manifest)

    if not manifest:
        print("\nFAIL: nothing exported.")
        sys.exit(2)

    total_mb = sum(c["bytes"] for c in manifest) / 1e6
    if total_mb > ARGS.max_total_mb:
        print(f"\nFAIL: exported audio is {total_mb:.1f} MB, over --max-total-mb "
              f"({ARGS.max_total_mb}). Lower --clip-seconds or export fewer clips.")
        sys.exit(2)

    pairs = build_pairs(manifest, ARGS.pairs)

    n_correct = sum(1 for c in manifest if c["correct"])
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": os.path.basename(ARGS.checkpoint),
        "pool": pool,
        "layer_mix": bool(layer_mix),
        "tta": ARGS.tta,
        "countries": COUNTRIES,
        "casa_countries": sorted(set(CASA_CONFIGS.values())),
        "split_params": {"seed": ARGS.seed, "casa_val_frac": ARGS.casa_val_frac,
                         "casa_select_frac": ARGS.casa_select_frac},
        "summary": {"n": len(manifest), "n_correct": n_correct,
                    "accuracy": round(n_correct / len(manifest), 4)},
        "pairs": pairs,
        "clips": manifest,
    }

    path = os.path.join(out_dir, "demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())          # on disk before the exit below skips finalization

    print("\n" + "=" * 70)
    print(f"  {len(manifest)} clips -> {path}")
    print(f"  audio: {total_mb:.1f} MB in {os.path.join(out_dir, 'audio')}")
    print(f"  the demo set is {n_correct}/{len(manifest)} correct -- this is a CURATED "
          "spread, not an accuracy estimate. Do not quote it as one on the page.")
    print("=" * 70)


def _leave_without_finalizing():
    """Exit without running CPython's interpreter finalization.

    torch, the HF dataloader workers and the streaming reader all hold native
    thread state. On some boxes CPython's finalizer cannot release it and aborts
    with

        Fatal Python error: PyGILState_Release: thread state ... must be current
        Python runtime state: finalizing

    which is SIGABRT -- exit -6 -- raised AFTER every output file is written and
    fsynced. The work is done and correct at that point, so a crash there is
    purely cosmetic damage that makes a good export look like a failed one.

    Everything this script produces is flushed and fsynced before this runs, so
    there is nothing for finalization to do except crash. os._exit skips atexit
    handlers and native teardown entirely.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
    _leave_without_finalizing()
