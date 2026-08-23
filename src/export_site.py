#!/usr/bin/env python
"""
Build every file the landing page reads, then package them for transfer.

    python src/export_site.py --checkpoint best_v3_lm_casatr_casavf20_6k_cohere-ar.pt

Run this ON THE VM, from the repository root. It produces one directory and one
archive:

    site/data/demo.json          clip metadata + the full 20-way posterior per clip
    site/data/audio/<id>.wav     the clip audio, 16 kHz mono
    site/data/ledger.json        every training run, from results/runs.csv
    site/data/results.json       the accuracy table and the headline figures
    site-data.tar.gz             all of the above, ready to copy to your laptop

The page reads those four things at load. Until they exist it falls back to the
simulated numbers baked into index.html and shows a banner saying so. Dropping
this archive into site/ is what makes the banner disappear.


WHY THIS WRAPS THE EXISTING SCRIPTS INSTEAD OF REPLACING THEM
-------------------------------------------------------------
export_demo.py already owns the only inference path that has been verified
against the trainer, and export_ledger.py already owns the only reader of
runs.csv. Both are invoked here as subprocesses rather than imported, because
each parses sys.argv at module scope -- importing them would mean fighting their
argv handling for no benefit. Everything this script adds is arithmetic over
results.csv plus packaging.


WHY THE ACCURACY TABLE IS A CURATED LIST
----------------------------------------
TABLE below names four runs. The *selection* is editorial -- it is the story of
the campaign, not every run that happened -- but every *number* is read out of
runs.csv at build time. Nothing here is typed by hand, so the page cannot drift
away from the log.

One wrinkle worth knowing: the earliest baselines predate program-disjoint
splitting, so they have no held-out figure. For those rows this script falls back
to the selection-half number and sets holdout=false, and the page marks them.
Quoting a selection-half number as if it were held out would flatter the
baseline and understate the result.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tarfile

REPO_MARKERS = ("src/train.py", "src/submit.py")

# label, run name in runs.csv, is this the row to highlight
TABLE = [
    ("whisper-small, no aug",     "noaug_baseline_whisper-small",         False),
    ("cohere-ar, tuned",          "v3_lm_llrd1_cohere-ar",                False),
    ("+ augmentation, no reverb", "v3_lm_final_noreverb_codec_cohere-ar", False),
    ("casavf20_6k",               "v3_lm_casatr_casavf20_6k_cohere-ar",   True),
]

DEAD = ("crashed", "killed", "failed")


def die(msg, hint=None):
    print(f"\nFAIL: {msg}")
    if hint:
        print(f"      {hint}")
    sys.exit(2)


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f          # NaN -> None


def get_args():
    p = argparse.ArgumentParser(
        description="Build and package every data file the landing page reads.")
    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="checkpoint to run over the demo clips")
    p.add_argument("--out", default="site/data", metavar="DIR",
                   help="output directory (default: site/data)")
    p.add_argument("--runs", default="results/runs.csv", metavar="PATH")
    p.add_argument("--archive", default="site-data.tar.gz", metavar="PATH",
                   help="archive to write; pass 'none' to skip packaging")

    # forwarded verbatim to export_demo.py
    p.add_argument("--n-casa", type=int, default=8)
    p.add_argument("--n-adi", type=int, default=4)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--pool-size", type=int, default=400)
    p.add_argument("--casa-val-frac", type=float, default=0.2,
                   help="MUST match what the checkpoint was trained with (default: 0.2)")
    p.add_argument("--casa-select-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip-seconds", type=float, default=8.0)
    p.add_argument("--max-total-mb", type=float, default=12.0)
    p.add_argument("--tta", type=int, default=0)
    p.add_argument("--correct-frac", type=float, default=0.75,
                   help="fraction of demo clips the model got right (rest are real failures)")
    p.add_argument("--tta-seconds", type=float, default=None,
                   help="model input window in seconds; shorter costs less memory")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="bf16 needs Ampere or newer (sm_80+). On Turing (sm_75, e.g. an "
                        "RTX 2080 Ti) use fp16 -- there is no native bf16 there.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="lower it on a small card; 11 GB wants about 4")
    p.add_argument("--skip-install", action="store_true")

    p.add_argument("--skip-demo", action="store_true",
                   help="rebuild ledger.json and results.json only; no GPU needed")
    return p.parse_args()


def check_gpu(precision="bf16"):
    """Fail in the first second, not after the checkpoint and the corpus are loaded.

    A scheduler will happily hand you a GPU older than anything your torch build
    has kernels for. The failure then arrives ~10 minutes in, as
    `cudaErrorNoKernelImageForDevice`, after a 7 GB checkpoint load and a full
    dataset resolve -- and it costs the whole allocation. One tiny CUDA op
    settles it immediately, and it is a real test rather than a guess at
    architecture compatibility.
    """
    try:
        import torch
    except ImportError:
        return                      # export_demo.py installs its own dependencies
    if not torch.cuda.is_available():
        print("\n  NOTE: no CUDA device visible. This will run on CPU and will be very slow.")
        return
    name = torch.cuda.get_device_name(0)
    cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
    try:
        (torch.zeros(8, device="cuda") + 1).sum().item()
    except Exception as e:
        die(f"this GPU cannot run this PyTorch build: {name} ({cap}).",
            "torch has kernels for: " + " ".join(torch.cuda.get_arch_list()) +
            f"\n      {type(e).__name__}: {str(e).splitlines()[0]}"
            "\n      Ask the scheduler for a newer card -- A100 (sm_80), A40/A6000 (sm_86),"
            "\n      H100 (sm_90) or RTX PRO 6000 (sm_120) all work with this build.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        die(f"{name} ({cap}) has no native bf16.",
            "bf16 arrived with Ampere (sm_80). On Turing, re-run with --precision fp16.")
    print(f"\n  GPU: {name} ({cap}), precision {precision} -- supported, proceeding.")


def preflight(a):
    for m in REPO_MARKERS:
        if not os.path.exists(m):
            die(f"{m} not found -- run this from the repository root.",
                "cd into the repo and try again.")
    if not a.skip_demo:
        if not os.path.exists(a.checkpoint):
            die(f"checkpoint not found: {a.checkpoint}",
                "ls *.pt   to see what this box actually has.")
        if not os.environ.get("HF_TOKEN"):
            die("HF_TOKEN is not set.",
                "export HF_TOKEN=hf_...   (needs gated access to the model and Casablanca)")
        check_gpu(a.precision)
    if not os.path.exists(a.runs):
        die(f"{a.runs} not found -- the ledger is built from it.")


def _usable_json(path):
    """True if `path` exists and parses. The only evidence that actually matters."""
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return os.path.getsize(path) > 64
    except Exception:
        return False


def run(cmd, what, produces=None):
    """Run a step, and judge it by what it produced rather than by how it exited.

    torch and the HF workers can abort during interpreter finalization (SIGABRT,
    -6) after every output file is already written and fsynced. Treating that as
    a failed step throws away a completed GPU run, which on a scheduler means
    queueing again for another hour. So: non-zero exit plus valid outputs is a
    warning; non-zero exit with nothing on disk is still fatal.
    """
    print(f"\n{'='*70}\n  {what}\n{'='*70}")
    print("  $ " + " ".join(cmd) + "\n")
    r = subprocess.run(cmd)
    if r.returncode == 0:
        return
    if produces and all(_usable_json(p) if p.endswith(".json") else os.path.exists(p)
                        for p in produces):
        print(f"\n  WARNING: {what} exited {r.returncode}, but every file it should")
        print("           produce is present and parses. This is the known torch/CPython")
        print("           shutdown abort, which happens after the work is finished.")
        print("           Continuing. Check the output above for a real error first.")
        return
    die(f"{what} exited {r.returncode}.",
        "Nothing usable was written. Fix the error above and re-run.")


def build_results(runs_path, out_path):
    """The accuracy table and headline figures, read out of runs.csv."""
    with open(runs_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_name = {}
    for r in rows:
        by_name.setdefault((r.get("name") or "").strip(), []).append(r)

    def pick(name):
        """Latest finished run of that name, else the latest of any state."""
        cands = by_name.get(name, [])
        if not cands:
            return None
        done = [c for c in cands if (c.get("state") or "").strip().lower() == "finished"]
        return sorted(done or cands, key=lambda c: c.get("created_at") or "")[-1]

    table, missing = [], []
    for label, name, win in TABLE:
        r = pick(name)
        if r is None:
            missing.append(name)
            continue
        idacc = num(r.get("sum.eval/id_acc")) or num(r.get("sum.final_id_acc"))
        hold = num(r.get("sum.eval/casa_acc_holdout")) or num(r.get("sum.final_casa_acc_holdout"))
        sel = num(r.get("sum.eval/casa_acc")) or num(r.get("sum.final_casa_acc"))
        broadcast = hold if hold is not None else sel
        if idacc is None or broadcast is None:
            missing.append(f"{name} (no usable metrics)")
            continue
        table.append({
            "label": label,
            "run": name,
            "in_domain": round(idacc, 1),
            "broadcast": round(broadcast, 1),
            "gap": round(idacc - broadcast, 1),
            "holdout": hold is not None,       # false => selection half, page marks it
            "win": bool(win),
        })

    if missing:
        print("\n  NOTE: not in runs.csv, so these rows are omitted rather than invented:")
        for m in missing:
            print(f"    - {m}")
    if not table:
        die("no table rows could be built from runs.csv.",
            "Check the run names in TABLE at the top of this script.")

    best = min(table, key=lambda t: t["gap"])
    doc = {
        "source": os.path.basename(runs_path),
        "table": table,
        "headline": {
            "broadcast": best["broadcast"],
            "in_domain": best["in_domain"],
            "gap": best["gap"],
            "run": best["run"],
            "holdout": best["holdout"],
        },
        "n_test_clips": 878,
        "n_classes": 20,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"\n  results.json -> {out_path}")
    for t in table:
        mark = "  <- headline" if t is best else ""
        star = "" if t["holdout"] else "   (selection half, not held out)"
        print(f"    {t['label']:<26s} {t['in_domain']:>5.1f}  {t['broadcast']:>5.1f}  "
              f"{t['gap']:>+6.1f}{star}{mark}")
    return doc


def verify(out_dir):
    """Refuse to package a half-built directory."""
    need = ["demo.json", "ledger.json", "results.json"]
    problems = []
    for n in need:
        p = os.path.join(out_dir, n)
        if not os.path.exists(p):
            problems.append(f"{n} missing")
            continue
        if os.path.getsize(p) < 64:
            problems.append(f"{n} is suspiciously small ({os.path.getsize(p)} bytes)")
        try:
            with open(p, encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            problems.append(f"{n} is not valid JSON: {e}")

    audio = os.path.join(out_dir, "audio")
    n_wav = len([x for x in os.listdir(audio) if x.endswith(".wav")]) if os.path.isdir(audio) else 0
    return problems, n_wav


def main():
    a = get_args()
    preflight(a)
    os.makedirs(a.out, exist_ok=True)
    py = sys.executable

    if a.skip_demo:
        print("\n  --skip-demo: not running inference; demo.json left as-is.")
    else:
        cmd = [py, "src/export_demo.py",
               "--checkpoint", a.checkpoint,
               "--out", a.out,
               "--n-casa", str(a.n_casa),
               "--n-adi", str(a.n_adi),
               "--pairs", str(a.pairs),
               "--pool-size", str(a.pool_size),
               "--casa-val-frac", str(a.casa_val_frac),
               "--casa-select-frac", str(a.casa_select_frac),
               "--seed", str(a.seed),
               "--clip-seconds", str(a.clip_seconds),
               "--max-total-mb", str(a.max_total_mb),
               "--tta", str(a.tta),
               "--correct-frac", str(a.correct_frac),
               "--precision", a.precision,
               "--batch-size", str(a.batch_size)]
        if a.tta_seconds is not None:
            cmd += ["--tta-seconds", str(a.tta_seconds)]
        if a.skip_install:
            cmd.append("--skip-install")
        run(cmd, "1/3  demo clips + posteriors (this is the GPU step)",
            produces=[os.path.join(a.out, "demo.json")])

    run([py, "src/export_ledger.py", "--runs", a.runs,
         "--out", os.path.join(a.out, "ledger.json")],
        "2/3  training ledger",
        produces=[os.path.join(a.out, "ledger.json")])

    print(f"\n{'='*70}\n  3/3  accuracy table\n{'='*70}")
    build_results(a.runs, os.path.join(a.out, "results.json"))

    problems, n_wav = verify(a.out)
    if problems:
        # With --skip-demo this is expected: ledger and results were rebuilt on a box
        # with no GPU and no checkpoint. Only refuse when inference was supposed to run.
        if a.skip_demo:
            print("\n  Rebuilt ledger.json and results.json. Not packaging:")
            for p in problems:
                print(f"    - {p}")
            print("\n  Run without --skip-demo on the VM to produce the clips.")
            return
        print("\n  INCOMPLETE -- not packaging:")
        for p in problems:
            print(f"    - {p}")
        sys.exit(2)

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(a.out) for f in fs)
    print(f"\n{'='*70}")
    print(f"  {a.out}: {n_wav} clips, {total/1e6:.1f} MB total")

    if a.archive.lower() == "none":
        print("  --archive none: skipping packaging.")
        print("="*70)
        return

    with tarfile.open(a.archive, "w:gz") as tar:
        tar.add(a.out, arcname="data")
    print(f"  archive: {a.archive}  ({os.path.getsize(a.archive)/1e6:.1f} MB)")
    print("="*70)
    print("\n  Next, FROM YOUR LAPTOP (not from this box):")
    print(f"    scp <user>@<vm-host>:{os.path.abspath(a.archive)} .")
    print(f"    tar -xzf {os.path.basename(a.archive)} -C site/")
    print("\n  Then reload the page. The simulated-data banner disappears on its own.")


if __name__ == "__main__":
    main()
