#!/usr/bin/env python
"""
PLACEHOLDER demo data. Delete this file once export_demo.py has run for real.

Writes a demo.json with the exact schema export_demo.py produces, plus synthesised
audio, so the landing page can be built and reviewed without GPU access.

Every record it writes carries `"synthetic": true`, and the page renders a loud
banner whenever it sees that flag. That is the whole point of the flag: placeholder
data that looks real is how a fabricated number ends up in front of judges. The
banner disappears on its own the moment real data replaces this file -- there is
nothing to remember to turn off.

The audio is deliberately NOT speech-like enough to be mistaken for a clip: filtered
noise under a syllable-rate envelope. It sounds like muffled talking through a wall,
which is exactly the impression you want from a placeholder.

    python src/placeholder_demo.py --out site/demo

Replace with:

    python src/export_demo.py --checkpoint best_v3_lm_casatr_casavf20_6k_cohere-ar.pt \\
        --casa-val-frac 0.2 --seed 42 --n-casa 8 --n-adi 4 --out site/demo
"""

import argparse
import json
import math
import os
import wave
from datetime import datetime, timezone

import numpy as np

COUNTRIES = ['MSA', 'BAH', 'TUN', 'ALG', 'EGY', 'IRA', 'JOR', 'KSA', 'KUW', 'LEB',
             'LIB', 'MAU', 'MOR', 'OMA', 'PAL', 'QAT', 'SUD', 'SYR', 'UAE', 'YEM']
CASA_COUNTRIES = ['ALG', 'EGY', 'JOR', 'MAU', 'MOR', 'PAL', 'UAE', 'YEM']
SR = 16000

# Confusions the model plausibly makes, so the placeholder probability vectors are not
# uniform noise: Maghrebi cluster, Levantine cluster, Gulf cluster.
NEIGHBOURS = {
    'ALG': ['MOR', 'TUN', 'MAU'], 'MOR': ['ALG', 'MAU', 'TUN'], 'MAU': ['MOR', 'ALG', 'TUN'],
    'JOR': ['PAL', 'SYR', 'LEB'], 'PAL': ['JOR', 'SYR', 'LEB'], 'SYR': ['LEB', 'JOR', 'PAL'],
    'UAE': ['QAT', 'KUW', 'BAH'], 'EGY': ['SUD', 'MSA', 'LIB'], 'YEM': ['OMA', 'KSA', 'SUD'],
    'KSA': ['KUW', 'BAH', 'QAT'], 'TUN': ['ALG', 'LIB', 'MOR'], 'LEB': ['SYR', 'PAL', 'JOR'],
}


def synth_clip(seconds, rng):
    """Filtered noise under a syllable-rate envelope. Obviously not speech."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = rng.normal(0, 1, n)
    # crude 2-pole lowpass, twice, to kill the hiss and leave a vocal-ish band
    for cutoff in (900.0, 2600.0):
        a = math.exp(-2 * math.pi * cutoff / SR)
        y = np.empty_like(x)
        acc = 0.0
        for i in range(n):                      # short clips; a loop is fine and dependency-free
            acc = a * acc + (1 - a) * x[i]
            y[i] = acc
        x = x - y if cutoff > 1500 else y
    syll = 0.5 + 0.5 * np.sin(2 * math.pi * rng.uniform(3.0, 4.5) * t + rng.uniform(0, 6.28))
    phrase = (np.sin(2 * math.pi * 0.35 * t) > -0.55).astype(float)   # pauses between phrases
    return (x * syll * phrase).astype(np.float32)


def write_wav(path, wav):
    w = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(w))) or 1.0
    pcm = np.clip(w / peak * 0.9 * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes(pcm.tobytes())
    return os.path.getsize(path)


def make_probs(true, bucket, rng):
    """A plausible distribution for the given outcome. Mass goes to dialect neighbours."""
    logits = rng.normal(-2.0, 0.7, len(COUNTRIES))
    ti = COUNTRIES.index(true)
    near = [COUNTRIES.index(c) for c in NEIGHBOURS.get(true, []) if c in COUNTRIES]

    if bucket == "confident-correct":
        logits[ti] += rng.uniform(5.0, 6.5)
        for j in near[:2]: logits[j] += rng.uniform(1.5, 2.5)
    elif bucket == "confident-wrong":
        wrong = near[0] if near else (ti + 1) % len(COUNTRIES)
        logits[wrong] += rng.uniform(4.0, 5.5)
        logits[ti] += rng.uniform(2.0, 3.2)
    else:  # close-call
        other = near[0] if near else (ti + 1) % len(COUNTRIES)
        base = rng.uniform(3.5, 4.5)
        logits[ti] += base + rng.uniform(-0.35, 0.35)
        logits[other] += base + rng.uniform(-0.35, 0.35)
        for j in near[1:3]: logits[j] += rng.uniform(1.0, 2.0)

    e = np.exp(logits - logits.max())
    return e / e.sum()


PLAN = [
    # (source, true label, bucket)
    # The four ADI countries deliberately overlap the Casablanca set: the page's
    # studio-vs-broadcast section needs the SAME country from both sources, and
    # Casablanca only carries 8 of the 20.
    ("casa", "MOR", "confident-correct"),
    ("casa", "EGY", "confident-correct"),
    ("casa", "ALG", "close-call"),
    ("casa", "JOR", "confident-wrong"),
    ("casa", "PAL", "close-call"),
    ("casa", "YEM", "confident-wrong"),
    ("casa", "UAE", "confident-correct"),
    ("casa", "MAU", "close-call"),
    ("adi",  "MOR", "confident-correct"),
    ("adi",  "EGY", "confident-correct"),
    ("adi",  "ALG", "confident-correct"),
    ("adi",  "JOR", "confident-correct"),
]

LABELS = {"casa": "Broadcast audio (Casablanca), held-out program",
          "adi":  "ADI-20 validation (in-domain)"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="site/demo")
    p.add_argument("--seconds", type=float, default=4.0,
                   help="placeholder clips are short on purpose -- this is 1.5 MB of audio "
                        "nobody should be shipping to visitors")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    out = os.path.abspath(a.out)
    audio_dir = os.path.join(out, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    clips, counters = [], {}
    for source, true, bucket in PLAN:
        k = counters.get(source, 0); counters[source] = k + 1
        cid = f"{source}-{k:02d}"

        size = write_wav(os.path.join(audio_dir, f"{cid}.wav"),
                         synth_clip(a.seconds, rng))
        probs = make_probs(true, bucket, rng)
        order = np.argsort(-probs)[:5]
        pred = COUNTRIES[int(probs.argmax())]

        clips.append({
            "id": cid,
            "audio": f"audio/{cid}.wav",
            "bytes": size,
            "duration_s": round(a.seconds, 2),
            "source": source,
            "source_label": LABELS[source],
            "program": None,
            "true": true,
            "pred": pred,
            "correct": pred == true,
            "confidence": round(float(probs.max()), 4),
            "bucket": bucket,
            "probs": [round(float(x), 5) for x in probs],
            "top5": [[COUNTRIES[int(i)], round(float(probs[int(i)]), 4)] for i in order],
            "entropy": round(float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum()), 3),
            "candidates": CASA_COUNTRIES if source == "casa" else COUNTRIES,
        })
        print(f"  {cid}  true={true:<4s} pred={pred:<4s} p={probs.max():.2f}  [{bucket}]")

    by = {}
    for c in clips:
        by[(c["source"], c["true"])] = c
    pairs = []
    for country in sorted({c["true"] for c in clips}):
        if ("adi", country) in by and ("casa", country) in by:
            pairs.append({"country": country,
                          "studio": by[("adi", country)]["id"],
                          "broadcast": by[("casa", country)]["id"]})

    n_correct = sum(1 for c in clips if c["correct"])
    doc = {
        "synthetic": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": "PLACEHOLDER -- no model was run",
        "pool": None, "layer_mix": None, "tta": 0,
        "countries": COUNTRIES,
        "casa_countries": CASA_COUNTRIES,
        "split_params": {"seed": None, "casa_val_frac": None, "casa_select_frac": None},
        "summary": {"n": len(clips), "n_correct": n_correct,
                    "accuracy": round(n_correct / len(clips), 4)},
        "pairs": pairs,
        "clips": clips,
    }
    with open(os.path.join(out, "demo.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)

    mb = sum(c["bytes"] for c in clips) / 1e6
    print(f"\n  {len(clips)} SYNTHETIC clips -> {os.path.join(out, 'demo.json')}  ({mb:.1f} MB audio)")
    print("  demo.json is flagged synthetic:true, so the page shows a placeholder banner.")


if __name__ == "__main__":
    main()
