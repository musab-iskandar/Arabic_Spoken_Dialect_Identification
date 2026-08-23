#!/usr/bin/env python
"""
Build the ledger the landing page reads, from results/runs.csv.

    python src/export_ledger.py            # -> site/ledger.json

The page is an account of what the campaign cost, so every figure on it has to come
from the run log rather than from memory. This script is the only thing that reads
runs.csv, and it does no rounding or editorialising: it copies state, wall-clock
runtime, last step and GPU for every run, in chronological order, and lets the page
do the arithmetic.

WHY THE DEAD RUNS ARE THE POINT
-------------------------------
A run that crashed still consumed the GPU. runs.csv records those with state
crashed/killed/failed and a real runtime_s, so the wasted hours are recoverable
exactly. They are not filtered out here and must not be filtered out on the page --
they are the subject.

COST
----
No price is stored in this file. Pass --rate to stamp an hourly figure into the JSON,
or leave it unset and the page shows hours only and flags the gap. Never guess a rate:
a fabricated number on a page about wasted money is the one error that would undo it.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

DEAD = ("crashed", "killed", "failed")

# The log records full marketing names; the ledger needs something that fits a column.
GPU_SHORT = [
    ("RTX PRO 6000 Blackwell Workstation", "RTX PRO 6000 (ws)"),
    ("RTX PRO 6000 Blackwell Server",      "RTX PRO 6000 (sv)"),
    ("H100 80GB HBM3",                     "H100 80GB"),
    ("H100 NVL",                           "H100 NVL"),
    ("A100-SXM4-80GB",                     "A100 80GB"),
]


def short_gpu(name):
    for needle, out in GPU_SHORT:
        if needle in name:
            return out
    return name.replace("NVIDIA ", "").strip() or "—"


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="results/runs.csv")
    p.add_argument("--out", default="site/ledger.json")
    p.add_argument("--rate", type=float, default=None, metavar="X",
                   help="cost per GPU-hour. Omit and the page shows hours only, with the "
                        "money column flagged as missing rather than estimated.")
    p.add_argument("--total", type=float, default=None, metavar="X",
                   help="total spend across the whole campaign. The hourly rate is derived "
                        "from it and total GPU time. Use this when you know what you paid "
                        "but not the hourly price. Mutually exclusive with --rate.")
    p.add_argument("--approx", action="store_true",
                   help="mark every money figure as approximate on the page. Implied by "
                        "--total, because a rate divided out of a rounded total cannot "
                        "support per-run precision.")
    p.add_argument("--currency", default="USD")
    a = p.parse_args()

    if not os.path.exists(a.runs):
        print(f"FAIL: no such file: {a.runs}")
        sys.exit(2)

    with open(a.runs, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("FAIL: runs.csv is empty")
        sys.exit(2)

    entries = []
    for r in rows:
        created = (r.get("created_at") or "").strip()
        try:
            when = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            when = None

        state = (r.get("state") or "").strip() or "unknown"
        secs = num(r.get("runtime_s")) or 0.0
        step = num(r.get("last_step"))

        entries.append({
            "at": when.isoformat() if when else None,
            "day": when.strftime("%Y-%m-%d") if when else None,
            "time": when.strftime("%H:%M") if when else "--:--",
            "tag": (r.get("cli.tag") or r.get("cfg.run") or "").strip() or "untagged",
            "model": (r.get("cfg.model") or "").strip(),
            "gpu": short_gpu((r.get("gpu") or "").strip()),
            "state": state,
            "dead": state in DEAD,
            "seconds": round(secs, 1),
            "hours": round(secs / 3600.0, 4),
            "step": int(step) if step is not None else None,
            # the metric this run reached, when it reached one at all
            "casa_holdout": num(r.get("sum.final_casa_acc_holdout")),
            "id_acc": num(r.get("sum.final_id_acc")),
        })

    entries.sort(key=lambda e: e["at"] or "")

    live = [e for e in entries if not e["dead"]]
    dead = [e for e in entries if e["dead"]]
    h_live = sum(e["hours"] for e in live)
    h_dead = sum(e["hours"] for e in dead)

    if a.rate is not None and a.total is not None:
        print("FAIL: pass --rate or --total, not both. They are two ways of saying the same "
              "thing and disagreeing versions would put two different numbers on the page.")
        sys.exit(2)

    hours_all = h_live + h_dead
    rate, approx, derived = a.rate, a.approx, False
    if a.total is not None:
        if hours_all <= 0:
            print("FAIL: --total needs some recorded GPU time to divide by.")
            sys.exit(2)
        rate = a.total / hours_all
        approx = True          # a rate divided out of a rounded total is not exact
        derived = True

    doc = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": os.path.basename(a.runs),
        "currency": a.currency,
        "rate_per_hour": rate,            # None until a real figure is supplied
        "rate_derived_from_total": derived,
        "cost_approx": approx,
        "total_given": a.total,
        "totals": {
            "runs": len(entries),
            "runs_dead": len(dead),
            "hours_total": round(h_live + h_dead, 3),
            "hours_dead": round(h_dead, 3),
            "share_dead_runs": round(len(dead) / len(entries), 4) if entries else 0,
            "share_dead_hours": round(h_dead / (h_live + h_dead), 4) if (h_live + h_dead) else 0,
            "gpu_types": sorted({e["gpu"] for e in entries if e["gpu"] != "—"}),
            "first_day": entries[0]["day"] if entries else None,
            "last_day": entries[-1]["day"] if entries else None,
        },
        "entries": entries,
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)

    t = doc["totals"]
    print(f"  {a.out}")
    print(f"  {t['runs']} runs · {t['first_day']} -> {t['last_day']}")
    print(f"  {t['runs_dead']} died ({t['share_dead_runs']*100:.0f}% of runs)")
    print(f"  {t['hours_dead']:.2f} h burned of {t['hours_total']:.2f} h "
          f"({t['share_dead_hours']*100:.0f}% of GPU time)")
    print(f"  {len(t['gpu_types'])} GPU types: {', '.join(t['gpu_types'])}")
    if rate is None:
        print("  rate_per_hour is null -- the page will show hours and flag the money as "
              "missing. Re-run with --rate or --total to fill it in.")
    else:
        tilde = "~" if approx else ""
        print(f"  rate {tilde}{rate:.3f} {a.currency}/h"
              + (" (derived from --total)" if derived else ""))
        print(f"  burned {tilde}{h_dead*rate:.2f} {a.currency} "
              f"of {tilde}{(h_live+h_dead)*rate:.2f} {a.currency}")


if __name__ == "__main__":
    main()
