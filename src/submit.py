#!/usr/bin/env python
"""
NADI-2026 Subtask 2 (ADI-20) -- submission writer.

Loads a checkpoint trained by cohere_train_v3.py, runs it over
UBC-NLP/NADI2026_subtask2_adi_test, and writes the competition submission:

    logits.tsv       878 lines, 20 tab-separated floats each
    predictions.tsv  878 lines, one integer class index each
    submission.zip   the two files above, at the archive root

Column order is fixed by the task and matches the baseline notebook:

    ['MSA','BAH','TUN','ALG','EGY','IRA','JOR','KSA','KUW','LEB',
     'LIB','MAU','MOR','OMA','PAL','QAT','SUD','SYR','UAE','YEM']

which is byte-identical to cohere_train_v3.py's COUNTRIES, asserted at startup.


WHY THIS SCRIPT IS SHAPED THE WAY IT IS
---------------------------------------
The submission has no sample-ID column. Row i of logits.tsv is scored against test
row i, and nothing else identifies it. That makes row alignment the only thing that
can silently destroy a submission -- a model that is 3 points worse announces itself,
a submission that is off by one row scores at chance and looks like a modelling
failure. Three specific hazards, each handled explicitly below:

  1. cohere_train_v3.py's collate DROPS undecodable clips (`SKIPPED["n"] += 1;
     continue`). That is correct for training and catastrophic here: one dropped
     clip shifts every subsequent row up by one. This script therefore carries a
     global row index through the collate and SCATTERS results into a
     pre-allocated array, so a dropped clip leaves a neutral row in place instead
     of shifting its neighbours. Any drop is reported loudly and, by default, is
     fatal (--allow-undecodable to override).

  2. The notebook's write_logits/write_preds open with mode "a" (append). Running
     the submission cell twice silently produces a 1756-line file. This script
     truncates, and re-verifies the line count after writing.

  3. Architecture flags must match the checkpoint or load_state_dict(strict=False)
     quietly loads a partial model that still runs and still predicts. This script
     INFERS --layer-mix / --lora / --pool from the checkpoint's own keys, and treats
     ANY missing or unexpected key as fatal -- so a mismatch stops the run and prints
     the offending keys, rather than costing a submission.

The model, feature extraction, pooling and TTA code below is copied from
cohere_train_v3.py rather than imported, because that module parses sys.argv and
downloads the full ADI20 dataset at import time. Copied blocks carry a
`v3:<line>` reference. To guard against the two diverging, --verify (on by
default) re-scores a sample of the ADI20 validation split through this script's
own path and aborts if accuracy is far from what the checkpoint reported -- which
is what catches a feature-extraction or pooling drift that no assert would.

Credentials come from $HF_TOKEN / $HUGGINGFACE_TOKEN only -- there is no token in this file.


USAGE (run from the repo root, so submissions/ lands there rather than in src/)
-----
    # normal case: everything inferred from the checkpoint
    python src/submit.py --checkpoint best_my_run_cohere-ar.pt --tag my_run

    # skip the validation sanity pass (faster, less safe)
    python src/submit.py --checkpoint best.pt --tag my_run --no-verify

    # multi-crop test-time augmentation (measured worth ~0.01pt on this task)
    python src/submit.py --checkpoint best.pt --tag my_run --tta 3
"""

import argparse
import inspect
import itertools
import json
import math
import os
import subprocess
import sys
import zipfile

# --- credentials -------------------------------------------------------------------------
# Read from $HF_TOKEN / $HUGGINGFACE_TOKEN only. Same convention as cohere_train_v5.py.
TEST_REPO = "UBC-NLP/NADI2026_subtask2_adi_test"
TEST_SPLIT = "train"          # the test set ships its single split under this name
TEST_ROWS_EXPECTED = 878      # from the dataset card; verified against the real load
VAL_REPO = "UBC-NLP/NADI_2026_ADI20_micro"

HF_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"

COUNTRIES = ['MSA', 'BAH', 'TUN', 'ALG', 'EGY', 'IRA', 'JOR', 'KSA', 'KUW', 'LEB',
             'LIB', 'MAU', 'MOR', 'OMA', 'PAL', 'QAT', 'SUD', 'SYR', 'UAE', 'YEM']
labels2id = {k: i for i, k in enumerate(COUNTRIES)}

SR = TARGET_SR = 16000
MAX_AUDIO_SECONDS_COHERE = 30      # v3:2501 -- must match, it bounds what the encoder ever saw


# ============================================================================
# Arguments
# ============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Write a NADI-2026 subtask-2 submission from a cohere_train_v3.py checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", default="best_v3_lm_llrd1_cohere-ar.pt", metavar="PATH",
                   help="checkpoint to submit. This is the one setting that decides the score.")
    p.add_argument("--tag", default="", metavar="NAME",
                   help="names the output directory (submissions/<tag>/) and is recorded in "
                        "meta.json. Use the training run's tag so a submission can always be "
                        "traced back to the run that produced it.")
    p.add_argument("--out-dir", default=None,
                   help="output directory. Default: submissions/<tag or checkpoint stem>/")

    p.add_argument("--batch-size", type=int, default=8,
                   help="eval micro-batch. Test clips run up to 30s, so this is deliberately "
                        "small; it is halved automatically on CUDA OOM.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="autocast precision for the forward pass. bf16 matches how the reported "
                        "validation accuracy was measured.")
    p.add_argument("--tta", type=int, default=0, metavar="N",
                   help="average log-probabilities over N evenly-spaced crops per clip. 0/1 = "
                        "off, which passes the whole clip and reproduces the run's headline "
                        "id_acc. Measured worth on this task: +0.01 points, so off by default.")
    p.add_argument("--tta-seconds", type=float, default=None,
                   help="TTA window length. Default: the longest --crop-set value the checkpoint "
                        "trained on, via --crop-set below.")
    p.add_argument("--crop-set", default="3,5,8,12",
                   help="the training run's --crop-set. Used ONLY to pick the default TTA window; "
                        "it has no effect when --tta is off.")

    p.add_argument("--pool", default=None, choices=["mean", "mean_std", "attn_stat"],
                   help="pooling mode. Default: inferred from the checkpoint's classifier shape.")
    p.add_argument("--layer-mix", dest="layer_mix", action="store_true", default=None,
                   help="force layer-mix on. Default: inferred from the presence of a "
                        "'layer_weights' tensor in the checkpoint.")
    p.add_argument("--no-layer-mix", dest="layer_mix", action="store_false",
                   help="force layer-mix off.")
    p.add_argument("--lora-alpha", type=float, default=32.0,
                   help="LoRA alpha. Only consulted for a LoRA checkpoint. This value is NOT "
                        "stored in the state dict (scaling = alpha/r is computed, not a "
                        "parameter), so it must match the training run or every adapter is "
                        "mis-scaled -- the script warns when it has to rely on this.")

    p.add_argument("--verify", dest="verify", type=int, default=256, metavar="N",
                   help="before writing anything, score N ADI20 validation clips through this "
                        "script's own path. Guards against this file drifting from "
                        "cohere_train_v3.py's feature extraction or pooling, which no assert "
                        "can catch. 0 disables.")
    p.add_argument("--no-verify", dest="verify", action="store_const", const=0,
                   help="skip the validation sanity pass.")
    p.add_argument("--verify-min", type=float, default=80.0, metavar="PCT",
                   help="abort if the verification accuracy falls below this. The v3 llrd1 "
                        "checkpoint scores ~89.7 on the full validation split, so a sample of "
                        "512 landing under 80 means something in this path is wrong.")

    p.add_argument("--allow-undecodable", action="store_true",
                   help="write a submission even if some test clips could not be decoded. They "
                        "get uniform log-probabilities (argmax -> MSA). Off by default: a "
                        "decode failure on the test set is worth investigating, not papering "
                        "over, and it is silent in the output file.")
    p.add_argument("--pred-format", default="index", choices=["index", "label"],
                   help="predictions.tsv contents. 'index' matches the baseline notebook, which "
                        "writes the argmax index. A labels file is always written alongside for "
                        "human inspection regardless.")
    p.add_argument("--skip-install", action="store_true",
                   help="assume dependencies are already present and skip the pip step. Safe "
                        "and faster on a box where this script (or cohere_train_v3.py) has "
                        "already run once.")
    p.add_argument("--torch-index-url", default=None,
                   help="pip --index-url used ONLY when torch has to be installed from scratch, "
                        "e.g. https://download.pytorch.org/whl/cu121. Never used to upgrade an "
                        "existing torch.")
    p.add_argument("--selftest", action="store_true",
                   help="check the row-alignment and output-format logic and exit. Needs torch "
                        "but no GPU, no network, no checkpoint. Run this first on a new box.")
    p.add_argument("--limit", type=int, default=0,
                   help="only run the first N test clips. Pipeline debugging only -- the result "
                        "is NOT a valid submission and the script says so.")
    return p.parse_args()


ARGS = get_args()

assert COUNTRIES == ['MSA', 'BAH', 'TUN', 'ALG', 'EGY', 'IRA', 'JOR', 'KSA', 'KUW', 'LEB',
                     'LIB', 'MAU', 'MOR', 'OMA', 'PAL', 'QAT', 'SUD', 'SYR', 'UAE', 'YEM'], \
    "COUNTRIES no longer matches the order the task specifies for logits.tsv columns."
assert len(COUNTRIES) == 20


# ============================================================================
# Dependencies -- ported from cohere_train_v3.py:654-789. Same rule: never
# silently REPLACE a working torch, but do install one if none exists.
#
# The failure this prevents: pip upgrading torch transitively (accelerate
# declares a torch floor) while leaving a torchaudio C extension built against
# the old libtorch, which then dies at import with
#     OSError: ..._torchaudio.abi3.so: undefined symbol: torch_library_impl
# A constraints file binds the WHOLE resolution including transitive deps, so
# nothing can drag torch along behind our backs.
# ============================================================================
def _installed_version(pkg):
    """Version of an installed distribution WITHOUT importing it -- importing torch here would
    cost ~10s and would happen before the pin below could protect it."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _ensure_torch_pair(pip, index_args):
    """Guarantee a MATCHED torch/torchaudio pair exists, without moving an existing torch.

    torchaudio is optional for this script -- the only thing it does here is resample the ~1% of
    clips that are not already 16 kHz, and librosa covers that. So the 'no matching torchaudio
    exists' branch is a note, not an error.
    """
    torch_v, ta_v = _installed_version("torch"), _installed_version("torchaudio")
    if torch_v and ta_v:
        print(f"  torch {torch_v} + torchaudio {ta_v} already present -- pinning, not touching.")
    elif not torch_v and not ta_v:
        print("  neither torch nor torchaudio is installed. Installing them TOGETHER in one "
              "resolution so the pair is guaranteed to match.")
        subprocess.run(pip + index_args + ["torch", "torchaudio"], check=True)
        torch_v, ta_v = _installed_version("torch"), _installed_version("torchaudio")
        print(f"  installed torch {torch_v} + torchaudio {ta_v}")
    elif torch_v and not ta_v:
        base = torch_v.split("+")[0]
        print(f"  torch {torch_v} present, torchaudio absent. Trying torchaudio=={base} with "
              "--no-deps so this cannot move torch.")
        subprocess.run(pip + ["--no-deps"] + index_args + [f"torchaudio=={base}"])
        ta_v = _installed_version("torchaudio")
        if not ta_v:
            print(f"  no torchaudio matches torch {base} (torchaudio was sunset at 2.11). "
                  "Continuing WITHOUT it -- librosa covers the only thing it is used for here. "
                  "Deliberately NOT installing a mismatched one: that is what causes the "
                  "'undefined symbol: torch_library_impl' crash.")
    else:
        print(f"  torchaudio {ta_v} present but torch is not -- installing the matching torch.")
        subprocess.run(pip + index_args + ["torch", "torchaudio"], check=True)
        torch_v, ta_v = _installed_version("torch"), _installed_version("torchaudio")
    return torch_v, ta_v


def install_dependencies():
    print("\n" + "=" * 66)
    print("  Installing/upgrading dependencies")
    print("=" * 66)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    index_args = ["--index-url", ARGS.torch_index_url] if ARGS.torch_index_url else []

    torch_v, ta_v = _ensure_torch_pair(pip, index_args)

    pins = [f"{p}=={v}" for p, v in (("torch", torch_v), ("torchaudio", ta_v)) if v]
    constraint_args = []
    if pins:
        import tempfile
        with tempfile.NamedTemporaryFile("w", prefix="cohere_submit_constraints_",
                                         suffix=".txt", delete=False) as fh:
            fh.write("\n".join(pins) + "\n")
        constraint_args = ["-c", fh.name]
        print(f"  pinning {' '.join(pins)} for the rest of this install -- pip may not move them")

    # No -U: version floors already upgrade when genuinely needed, whereas -U upgrades
    # unconditionally and takes dependencies with it. Deliberately a shorter list than
    # cohere_train_v3.py's -- no wandb, matplotlib or pandas: this script trains nothing and
    # plots nothing, and every extra package is another chance to disturb torch.
    subprocess.run(pip + constraint_args +
                   ["transformers>=4.57", "accelerate", "huggingface_hub"], check=True)
    subprocess.run(pip + constraint_args +
                   ["datasets==3.5.0", "soundfile", "librosa", "numpy", "tqdm"], check=True)
    print("dependencies installed.\n")


if ARGS.skip_install:
    print("\n--skip-install set: assuming dependencies are already installed.\n")
else:
    install_dependencies()


# ============================================================================
# Imports. transformers/datasets/huggingface_hub are imported LAZILY at their
# use sites so --selftest needs only torch and numpy -- it checks index
# arithmetic and file format, and has no business pulling in a model library.
# ============================================================================
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
except ImportError as _e:
    _missing = getattr(_e, "name", None) or "a required package"
    print("\n" + "!" * 70)
    print(f"FAIL: {_missing} is not installed ({_e}).")
    if ARGS.skip_install:
        print("\n  --skip-install was passed, so the dependency step was skipped -- but the "
              "dependencies are not actually present on this machine.")
        print("  Re-run WITHOUT --skip-install once; it installs everything and pins torch so "
              "it cannot be disturbed:\n")
        print(f"    python {os.path.basename(sys.argv[0])} " +
              " ".join(a for a in sys.argv[1:] if a != "--skip-install"))
        print("\n  After that succeeds, --skip-install is safe (and faster) on every later run.")
    else:
        print("\n  The dependency install step ran but this package is still missing -- check "
              "the pip output above for the real error.")
    print("!" * 70 + "\n")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hf_login():
    # Both the gated cohere-ar encoder and the gated test repo need this, so a missing token
    # stops here with the export line rather than 404-ing mid-download.
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    src = "env"
    if not tok:
        print("\n" + "!" * 70)
        print("FAIL: no Hugging Face token. Set it in the environment:\n")
        print("    export HF_TOKEN=...        # https://huggingface.co/settings/tokens")
        print("\n  (Or copy .env.example to .env and fill it in.)")
        print("  The gated cohere-ar encoder and the gated test repo both require it.")
        print("!" * 70 + "\n")
        sys.exit(1)
    try:
        from huggingface_hub import login
        login(tok)
        print(f"HF login ok ({src})")
    except Exception as e:
        print(f"HF login failed: {type(e).__name__}: {e}")


# ============================================================================
# Model -- copied from cohere_train_v3.py, inference paths only.
# SpecAugment, torch.compile and the train-time fixed-crop mask fallback are
# omitted: none of them run under model.eval() with full-length clips, so
# carrying them would be dead code that could still drift.
# ============================================================================
def _encoder_layer_list(encoder):                                          # v3:1883
    for attr in ("layers", "encoder_layers", "blocks", "transformer_layers"):
        mod = getattr(encoder, attr, None)
        if isinstance(mod, (nn.ModuleList, nn.Sequential)):
            return list(mod)
    for m in encoder.modules():
        if isinstance(m, nn.ModuleList) and len(m) > 4:
            return list(m)
    return []


def _resolve_mask_kwarg(encoder):                                          # v3:1888
    try:
        sig = inspect.signature(encoder.forward)
    except (TypeError, ValueError):
        return None
    for name in ("attention_mask", "padding_mask", "input_lengths", "lengths", "feature_lengths"):
        if name in sig.parameters:
            return name
    return None


def _supports_hidden_states(encoder):                                      # v3:1903
    try:
        sig = inspect.signature(encoder.forward)
    except (TypeError, ValueError):
        return False
    return ("output_hidden_states" in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))


class LoRALinear(nn.Module):                                               # v3:1916
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        upd = torch.nn.functional.linear(
            torch.nn.functional.linear(self.lora_dropout(x).to(self.lora_A.dtype), self.lora_A),
            self.lora_B)
        return out + self.scaling * upd.to(out.dtype)


def apply_lora_at_paths(encoder, paths, ranks, alpha):
    """Wrap exactly the modules the checkpoint says were wrapped.

    cohere_train_v3.py's apply_lora matches --lora-targets substrings; reconstructing that here
    would mean guessing the training run's targets string. The checkpoint already names every
    wrapped module (one `<path>.lora_A` key each) and gives its rank (lora_A.shape[0]), so the
    exact set is recoverable. alpha is the one thing that is not -- see --lora-alpha.
    """
    wrapped = []
    for path in paths:
        parent_path, _, attr = path.rpartition(".")
        parent = encoder.get_submodule(parent_path) if parent_path else encoder
        base = getattr(parent, attr)
        if not isinstance(base, nn.Linear):
            raise RuntimeError(
                f"checkpoint says {path} was LoRA-wrapped, but that attribute is a "
                f"{type(base).__name__}, not nn.Linear. This checkpoint does not match the "
                f"encoder at {HF_ID}.")
        setattr(parent, attr, LoRALinear(base, r=ranks[path], alpha=alpha))
        wrapped.append(path)
    return wrapped


class DialectID(nn.Module):                                                # v3:1978
    """Cohere ASR encoder + mask-aware pool + linear head. Inference only."""

    def __init__(self, num_labels=20, pool="mean", layer_mix=False, layer_mix_stride=1):
        super().__init__()
        self._layer_mix = layer_mix
        self._layer_mix_stride = max(1, layer_mix_stride)
        self._layer_mix_checked = False
        from transformers import AutoModelForSpeechSeq2Seq
        full = AutoModelForSpeechSeq2Seq.from_pretrained(HF_ID, trust_remote_code=True).float()
        self.encoder = full.get_encoder()
        config = self.encoder.config
        hidden = getattr(config, "d_model", None) or config.hidden_size
        self.pool = pool
        self._mask_kwarg = _resolve_mask_kwarg(self.encoder)
        self._warned_no_mask = False
        print(f"  encoder mask kwarg resolved to: {self._mask_kwarg!r}")

        self.layer_weights = None
        if layer_mix:
            if not _supports_hidden_states(self.encoder):
                raise RuntimeError(
                    "checkpoint contains layer_weights but this encoder's forward does not "
                    "accept output_hidden_states -- the weighted layer sum cannot be "
                    "reproduced, so the submission would be generated by a different model "
                    "than the one that was trained.")
            n = len(_encoder_layer_list(self.encoder))
            self.layer_weights = nn.Parameter(torch.zeros(n + 1))
            print(f"  layer-mix on: {n} encoder layers discovered")

        clf_in = hidden * 2 if pool in ("mean_std", "attn_stat") else hidden
        if pool == "attn_stat":
            self.attn = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(clf_in, num_labels)

    def _derive_frame_mask(self, out, h, input_mask):                      # v3:2098
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
            m = input_mask.float().unsqueeze(1)
            m_ds = torch.nn.functional.interpolate(m, size=T, mode="nearest").squeeze(1)
            return m_ds.bool()
        if not self._warned_no_mask:
            print("  WARNING: no attention mask / output-length field from the encoder -- "
                  "pooling falls back to a plain mean over all frames, including padding. "
                  "This warning prints once.")
            self._warned_no_mask = True
        return None

    def _pool(self, h, mask):                                              # v3:2150
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

        m = mask.float().unsqueeze(-1)
        denom = m.sum(1).clamp(min=1.0)
        mean = (h * m).sum(1) / denom
        if self.pool == "mean":
            return mean
        if self.pool == "mean_std":
            var = ((h - mean.unsqueeze(1)) ** 2 * m).sum(1) / denom
            return torch.cat([mean, torch.sqrt(var.clamp(min=1e-6))], dim=-1)
        logits = self.attn(h).squeeze(-1).masked_fill(~mask.bool(), float("-inf"))
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        wmean = (h * w).sum(1)
        wstd = torch.sqrt(((h - wmean.unsqueeze(1)) ** 2 * w).sum(1).clamp(min=1e-6))
        return torch.cat([wmean, wstd], dim=-1)

    def _mix_layers(self, out, h_last):                                    # v3:2224
        hs = getattr(out, "hidden_states", None)
        if hs is None and isinstance(out, (tuple, list)) and len(out) > 1:
            hs = out[1] if isinstance(out[1], (tuple, list)) else None
        if not hs or len(hs) < 2:
            raise RuntimeError(
                f"layer-mix is on but the encoder returned {0 if not hs else len(hs)} hidden "
                "state(s), so the weighted sum would silently reduce to last-layer pooling -- "
                "a different model than the checkpoint was trained as.")
        hs = list(hs)[::self._layer_mix_stride]
        if hs[-1] is not h_last and h_last is not None:
            hs.append(h_last)
        n = len(hs)
        if not self._layer_mix_checked:
            print(f"  layer-mix: encoder returned {n} usable hidden states")
            self._layer_mix_checked = True
        w = self.layer_weights
        assert w.numel() == n, (
            f"layer weight count {w.numel()} != hidden-state count {n}. The checkpoint was "
            "trained against a different encoder layer count than the one just loaded.")
        weights = torch.softmax(w, dim=0).to(hs[-1].dtype)
        return torch.stack(hs, dim=0).mul(weights.view(-1, 1, 1, 1)).sum(0)

    def forward(self, input_features=None, attention_mask=None):           # v3:2289
        kwargs = {}
        if self._mask_kwarg and attention_mask is not None:
            kwargs[self._mask_kwarg] = attention_mask
        if self._layer_mix:
            kwargs["output_hidden_states"] = True
        out = self.encoder(input_features, **kwargs)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self._layer_mix:
            h = self._mix_layers(out, h)
        return self.classifier(self._pool(h, self._derive_frame_mask(out, h, attention_mask)))


def set_bn_eval(module):                                                   # v3:2324
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


# ============================================================================
# Checkpoint introspection
# ============================================================================
def load_state(path):
    if not os.path.exists(path):
        print(f"\nFAIL: no such checkpoint: {path}\n"
              f"      Pass --checkpoint PATH. Files in {os.getcwd()}:")
        for f in sorted(os.listdir(".")):
            if f.endswith(".pt"):
                print(f"        {f}")
        sys.exit(2)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def infer_arch(state):
    """Read the architecture back out of the checkpoint instead of trusting flags.

    Every silent-wrong-submission path runs through an architecture mismatch, so the flags that
    can be recovered are recovered, and the ones that cannot (LoRA alpha) are named explicitly.
    """
    layer_mix = "layer_weights" in state
    n_layer_w = int(state["layer_weights"].numel()) if layer_mix else 0

    clf_w = state.get("classifier.weight")
    if clf_w is None:
        print("\nFAIL: checkpoint has no 'classifier.weight'. This is not a cohere_train_v3.py "
              "checkpoint.")
        sys.exit(2)
    n_labels, clf_in = int(clf_w.shape[0]), int(clf_w.shape[1])
    has_attn = "attn.weight" in state

    lora_paths, lora_ranks = [], {}
    for k, v in state.items():
        if k.endswith(".lora_A"):
            path = k[len("encoder."):-len(".lora_A")] if k.startswith("encoder.") \
                else k[:-len(".lora_A")]
            lora_paths.append(path)
            lora_ranks[path] = int(v.shape[0])

    return dict(layer_mix=layer_mix, n_layer_weights=n_layer_w, n_labels=n_labels,
                clf_in=clf_in, has_attn=has_attn,
                lora_paths=sorted(lora_paths), lora_ranks=lora_ranks)


def build_model(state, arch):
    hidden_guess = arch["clf_in"]
    if arch["has_attn"]:
        pool = "attn_stat"
    else:
        pool = None      # mean vs mean_std needs the encoder's hidden size; resolved below
    if ARGS.pool:
        pool = ARGS.pool
    layer_mix = arch["layer_mix"] if ARGS.layer_mix is None else ARGS.layer_mix

    if pool is None:
        # Build once with pool="mean" to learn hidden_size, then decide. Cheap: the encoder is
        # already in memory either way, and this avoids hardcoding 1280.
        probe = DialectID(num_labels=arch["n_labels"], pool="mean", layer_mix=layer_mix)
        hidden = probe.classifier.in_features
        if hidden_guess == hidden:
            model = probe
            pool = "mean"
        elif hidden_guess == hidden * 2:
            del probe
            pool = "mean_std"
            model = DialectID(num_labels=arch["n_labels"], pool=pool, layer_mix=layer_mix)
        else:
            print(f"\nFAIL: checkpoint's classifier takes {hidden_guess} inputs, but this "
                  f"encoder pools to {hidden} (mean) or {hidden*2} (mean_std/attn_stat). "
                  "The checkpoint does not match this encoder.")
            sys.exit(2)
    else:
        model = DialectID(num_labels=arch["n_labels"], pool=pool, layer_mix=layer_mix)

    if arch["lora_paths"]:
        print(f"  checkpoint is LoRA: {len(arch['lora_paths'])} wrapped module(s), "
              f"r={sorted(set(arch['lora_ranks'].values()))}")
        print(f"  WARNING: LoRA alpha is not stored in a checkpoint (scaling = alpha/r is "
              f"computed, not a parameter). Using --lora-alpha {ARGS.lora_alpha}. If the "
              "training run used a different alpha, every adapter is mis-scaled and the "
              "submission will score below the run's reported accuracy.")
        apply_lora_at_paths(model.encoder, arch["lora_paths"], arch["lora_ranks"],
                            ARGS.lora_alpha)

    if layer_mix and model.layer_weights is not None:
        n_have = model.layer_weights.numel()
        if n_have != arch["n_layer_weights"]:
            # v3 resizes layer_weights on its first forward to match the encoder's real
            # hidden-state count; the checkpoint therefore holds the post-resize size.
            with torch.no_grad():
                model.layer_weights.data = torch.zeros(arch["n_layer_weights"])
            print(f"  layer weights resized {n_have} -> {arch['n_layer_weights']} to match the "
                  "checkpoint (v3 does the same on its first forward)")

    model = model.to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"\nFAIL: checkpoint does not match the model that was built.\n"
              f"      {len(missing)} missing key(s), {len(unexpected)} unexpected key(s).")
        for k in list(missing)[:8]:
            print(f"        missing:    {k}")
        for k in list(unexpected)[:8]:
            print(f"        unexpected: {k}")
        print("      Loading anyway would produce a submission from a partially-initialised "
              "model that still runs and still predicts -- so this is fatal. Check --pool / "
              "--layer-mix against the run that produced this checkpoint.")
        sys.exit(2)
    print("  checkpoint loaded, all keys matched")
    model.eval()
    set_bn_eval(model)
    return model, pool, layer_mix


# ============================================================================
# Data -- order-preserving, label-optional
# ============================================================================
class _Indexed(torch.utils.data.Dataset):
    """Attaches the global row index to every sample.

    This is the mechanism that makes a dropped clip harmless: results are scattered back to
    their original positions rather than concatenated in arrival order.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        return {"_row": i, "audio": s["audio"],
                "dialect": s.get("dialect") if isinstance(s, dict) else None}


def _worker_init(worker_id):                                               # v3:3150
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _tta_windows(w, n, win_samples):                                       # v3:3159
    T = w.shape[-1]
    if n < 2 or T <= win_samples:
        return [w]
    starts = np.linspace(0, T - win_samples, n).round().astype(int)
    return [w[..., int(s):int(s) + win_samples] for s in sorted(set(starts.tolist()))]


def _resample(w, orig_sr, target_sr):                                      # v3:1243
    if orig_sr == target_sr:
        return w
    try:
        import torchaudio
        return torchaudio.functional.resample(w, orig_sr, target_sr)
    except ImportError:
        pass
    try:
        import librosa
    except ImportError:
        raise RuntimeError(
            f"a clip needs resampling {orig_sr} -> {target_sr} Hz, but neither torchaudio nor "
            "librosa is installed. pip install librosa")
    return torch.from_numpy(
        librosa.resample(w.detach().cpu().numpy().astype(np.float32),
                         orig_sr=orig_sr, target_sr=target_sr)).to(w.dtype)


def _wav(s):                                                               # v3:3215
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
            w = _resample(w, sr, TARGET_SR)
        except Exception:
            return None
    max_samples = int(MAX_AUDIO_SECONDS_COHERE * TARGET_SR)
    if w.shape[-1] > max_samples:
        w = w[..., :max_samples]
    return w


def _extract_features(fe, wavs):                                           # v3:2507
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


def make_collate(fe, tta, tta_samples, with_labels):
    def collate(samples):
        wavs, rows, groups, labs = [], [], [], []
        for s in samples:
            w = _wav(s)
            if w is None:
                continue        # the row stays at its neutral fallback; never shifts its peers
            gid = len(rows)
            rows.append(s["_row"])
            if with_labels:
                labs.append(labels2id[s["dialect"]])
            if tta and tta > 1:
                for view in _tta_windows(w, tta, tta_samples):
                    wavs.append(view.numpy())
                    groups.append(gid)
            else:
                wavs.append(w.numpy())
                groups.append(gid)
        if not wavs:
            return None
        feats, mask = _extract_features(fe, wavs)
        out = {"input_features": feats}
        if mask is not None:
            out["attention_mask"] = mask
        return (out,
                torch.tensor(groups, dtype=torch.long),
                torch.tensor(rows, dtype=torch.long),
                torch.tensor(labs, dtype=torch.long) if with_labels else None)

    return collate


def _merge_tta_groups(lp, groups, n_groups):
    """Average log-probabilities within each clip's TTA group -- v3:3417.

    The geometric mean of the per-window predictions. With TTA off every group has exactly one
    member and this is the identity. Factored out so --selftest can exercise it directly: it is
    the step that turns `w` windows back into one row per clip, and an off-by-one here would
    misalign the whole submission.
    """
    summed = torch.zeros(n_groups, lp.shape[1], device=lp.device, dtype=lp.dtype)
    summed.index_add_(0, groups, lp)
    counts = torch.zeros(n_groups, device=lp.device, dtype=lp.dtype)
    counts.index_add_(0, groups, torch.ones_like(groups, dtype=lp.dtype))
    return summed / counts.clamp(min=1).unsqueeze(1)


def _autocast_ctx(precision):                                              # v3:2671
    if precision == "bf16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.amp.autocast("cuda", enabled=False)


@torch.no_grad()
def predict(model, ds, fe, n_rows, desc, with_labels=False, workers=None):
    """Returns (logits[n_rows, 20] as log-probs, labels or None, set of undecodable rows).

    Rows are SCATTERED by global index. Anything the collate could not decode keeps its
    pre-filled uniform row, so the output always has exactly n_rows lines in dataset order
    regardless of what failed.
    """
    uniform = math.log(1.0 / len(COUNTRIES))
    out = torch.full((n_rows, len(COUNTRIES)), uniform, dtype=torch.float32)
    seen = torch.zeros(n_rows, dtype=torch.bool)
    labels = torch.full((n_rows,), -1, dtype=torch.long) if with_labels else None

    tta_sec = ARGS.tta_seconds
    if tta_sec is None:
        crops = [float(c) for c in ARGS.crop_set.split(",") if c.strip()]
        tta_sec = max(crops) if crops else MAX_AUDIO_SECONDS_COHERE
    tta_samples = int(tta_sec * SR)

    bs = max(1, ARGS.batch_size // max(1, ARGS.tta))
    n_workers = ARGS.num_workers if workers is None else workers
    while True:
        kw = dict(num_workers=n_workers)
        if n_workers > 0:
            kw.update(persistent_workers=True, prefetch_factor=ARGS.prefetch_factor,
                      worker_init_fn=_worker_init)
        loader = DataLoader(_Indexed(ds), batch_size=bs, shuffle=False, drop_last=False,
                            collate_fn=make_collate(fe, ARGS.tta, tta_samples, with_labels), **kw)
        ctx = _autocast_ctx(ARGS.precision)
        try:
            for batch in tqdm(loader, desc=desc, leave=False):
                if batch is None:
                    continue
                feats, groups, rows, labs = batch
                feats = {k: v.to(device) for k, v in feats.items()}
                groups = groups.to(device)
                with ctx:
                    lg = model(**feats)
                lp = torch.log_softmax(lg.float(), dim=-1)
                merged = _merge_tta_groups(lp, groups, rows.shape[0]).cpu()
                if merged.shape[0] != rows.shape[0]:
                    raise RuntimeError(
                        f"[{desc}] model returned {merged.shape[0]} pooled rows for "
                        f"{rows.shape[0]} clips -- the backbone split a clip into multiple "
                        "output rows. Lower MAX_AUDIO_SECONDS_COHERE.")
                out[rows] = merged
                seen[rows] = True
                if with_labels:
                    labels[rows] = labs
            break
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            print(f"  CUDA OOM -- retrying at batch size {bs}")

    missed = set((~seen).nonzero().flatten().tolist())
    return out, labels, missed


# ============================================================================
# Verification -- catches this file drifting from cohere_train_v3.py
# ============================================================================
def _load_val_sample(n):
    """Fetch ~n ADI20 validation clips WITHOUT pulling the training corpus.

    `load_dataset(REPO, split="validation")` does not do this. `split=` filters AFTER the builder
    has run download_and_prepare() over every split, so it still drags down all 38 ADI20 train
    shards (~50 GB) to score a few hundred validation clips. The repo layout is:

        data/train-XXXXX-of-00038.parquet        <- never wanted here
        data/validation-XXXXX-of-00006.parquet   <- the only thing this needs

    Streaming resolves the validation shards alone and fetches parquet row groups on demand, so
    the transfer is bounded by the sample size rather than by the corpus size. The non-streaming
    fallback pins data_files to a single validation shard, which is bounded too -- just coarser.
    """
    from datasets import load_dataset
    # The shuffle buffer is the real transfer cost -- filling it reads that many rows of audio.
    # Kept close to the sample size; shard-order shuffling supplies most of the class diversity.
    buf = max(1000, n * 2)
    rows = None
    try:
        it = load_dataset(VAL_REPO, split="validation", streaming=True)
        try:
            # Shuffles shard order as well as within the buffer. Without it the sample is
            # whatever sits at the head of the first shard, which may be one dialect.
            it = it.shuffle(seed=42, buffer_size=buf)
        except Exception as e:
            print(f"    streaming shuffle unavailable ({type(e).__name__}) -- reading in file "
                  "order instead; the sample may be class-skewed.")
        rows = list(itertools.islice(iter(it), n))
    except Exception as e:
        # Streaming leans on dill, which breaks on some Python/datasets combinations (seen on
        # Python 3.14 + datasets 3.5.0: "Pickler._batch_setitems() takes 2 positional arguments").
        # That is a packaging problem, not a data problem, so fall back rather than fail.
        print(f"    streaming unavailable ({type(e).__name__}: {str(e)[:120]})")
        print(f"    falling back to a normal load of the validation shards only: 2.9 GB / "
              "10,806 clips, already cached if you trained on this box. Still NOT the 18.9 GB "
              "train split -- data_files makes the builder blind to it.")
    if rows is None:
        ds = load_dataset(VAL_REPO,
                          data_files={"validation": "data/validation-*.parquet"},
                          split="validation")
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        rows = [ds[i] for i in range(len(ds))]
    if not rows:
        print("FAIL: could not read any ADI20 validation clip.")
        sys.exit(3)
    seen = {}
    for r in rows:
        d = r.get("dialect")
        seen[d] = seen.get(d, 0) + 1
    print(f"    sampled {len(rows)} clips covering {len(seen)}/{len(COUNTRIES)} dialects"
          + ("" if len(seen) >= 15 else "  <- narrow sample; treat the accuracy as a "
                                        "breakage check, not an accuracy estimate"))
    return rows


def verify(model, fe, n):
    print(f"\n--- verification: {n} ADI20 validation clips through this script's own path ---")
    print("    This exists because a copied feature-extraction or pooling path can drift from")
    print("    cohere_train_v3.py without any assert firing -- it would just quietly score")
    print("    worse, and a submission is the worst place to discover that.")
    print(f"    Streams from {VAL_REPO} validation shards only -- a few hundred MB, NOT the")
    print("    ~50 GB train corpus. Skip it with --no-verify if the box is already trusted.")
    ds = _load_val_sample(n)
    # workers=0 is deliberate and NOT a performance oversight. `ds` here is a plain Python list
    # of already-decoded audio arrays, not an Arrow-backed HF Dataset. Forking N workers over a
    # list of Python objects makes each worker's refcounting touch every object, which
    # copy-on-writes the whole thing per worker -- ~1 GB x N of real RAM for a few hundred clips.
    # On a box with modest RAM that swaps and looks exactly like a hang a few percent in.
    # The decode work is already done here, so the workers would buy nothing anyway.
    # The TEST path keeps its workers: that dataset IS Arrow-backed and forks cheaply.
    logits, labels, missed = predict(model, ds, fe, len(ds), "verify", with_labels=True,
                                     workers=0)
    ok = labels >= 0
    if not ok.any():
        print("FAIL: no validation clip decoded.")
        sys.exit(3)
    acc = 100.0 * (logits[ok].argmax(-1) == labels[ok]).float().mean().item()
    print(f"    accuracy on {int(ok.sum())} clips: {acc:.2f}%"
          + (f"   ({len(missed)} undecodable)" if missed else ""))
    if acc < ARGS.verify_min:
        print(f"\nFAIL: {acc:.2f}% is below --verify-min {ARGS.verify_min}. Either the "
              "checkpoint is not what you think it is, or this script's inference path has "
              "drifted from cohere_train_v3.py's. Refusing to write a submission.\n"
              "      Re-run with --no-verify only if you are certain the number is expected.")
        sys.exit(3)
    print(f"    OK (>= {ARGS.verify_min})")
    return acc


# ============================================================================
# Output
# ============================================================================
def write_submission(logits, idxs, out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    n = logits.shape[0]
    preds = logits.argmax(-1).tolist()

    logits_p = os.path.join(out_dir, "logits.tsv")
    preds_p = os.path.join(out_dir, "predictions.tsv")
    labels_p = os.path.join(out_dir, "predictions_labels.tsv")
    manifest_p = os.path.join(out_dir, "manifest.tsv")

    # Mode "w", not the baseline notebook's "a": appending makes a second run silently
    # produce a 2n-line file that still looks plausible.
    with open(logits_p, "w", newline="\n") as f:
        for row in logits.tolist():
            f.write("\t".join(repr(float(v)) for v in row) + "\n")
    with open(preds_p, "w", newline="\n") as f:
        for p in preds:
            f.write((str(p) if ARGS.pred_format == "index" else COUNTRIES[p]) + "\n")
    with open(labels_p, "w", newline="\n") as f:
        for p in preds:
            f.write(COUNTRIES[p] + "\n")
    with open(manifest_p, "w", newline="\n") as f:
        f.write("row\tidx\tpred_index\tpred_label\n")
        for i, (ix, p) in enumerate(zip(idxs, preds)):
            f.write(f"{i}\t{ix}\t{p}\t{COUNTRIES[p]}\n")

    # Re-read: the line count is the one property that decides whether the submission scores
    # at all, so it is checked from disk rather than assumed from the loop above.
    for path in (logits_p, preds_p):
        got = sum(1 for _ in open(path))
        if got != n:
            print(f"\nFAIL: {path} has {got} lines, expected {n}.")
            sys.exit(4)
    with open(logits_p) as f:
        for i, line in enumerate(f):
            k = len(line.rstrip("\n").split("\t"))
            if k != 20:
                print(f"\nFAIL: {logits_p} line {i+1} has {k} columns, expected 20.")
                sys.exit(4)

    zip_p = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(logits_p, "logits.tsv")       # archive root, matching the baseline's `zip` call
        z.write(preds_p, "predictions.tsv")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    dist = {}
    for p in preds:
        dist[COUNTRIES[p]] = dist.get(COUNTRIES[p], 0) + 1
    return zip_p, dist


# ============================================================================
# Self-test -- no model, no network, no checkpoint. Seconds to run.
# ============================================================================
def _selftest_collate_and_scatter():
    """The submission has no ID column, so row i of logits.tsv IS the identity of test row i.

    This exercises the two places that can break that: the collate's row/group bookkeeping, and
    the scatter in predict(). A stub feature extractor stands in for the real processor -- what
    is under test is index arithmetic, not mel filterbanks.
    """
    n_rows = 25
    bad = {3, 4, 17}          # clips the collate will refuse, as _wav does for corrupt audio

    class _FakeDS:
        def __len__(self):
            return n_rows

        def __getitem__(self, i):
            # A row's true class is its own index mod 20, so a misaligned row is detectable.
            n = 8000 if i not in bad else 10      # < 400 samples -> _wav returns None
            return {"audio": {"array": np.full(n, 0.01 * (i + 1), dtype=np.float32),
                              "sampling_rate": TARGET_SR}}

    def fake_fe(wavs, sampling_rate=None, return_tensors=None, padding=None):
        m = max(len(w) for w in wavs)
        f = torch.zeros(len(wavs), 4, m)
        for j, w in enumerate(wavs):
            f[j, 0, :len(w)] = torch.as_tensor(w)
        return {"input_features": f}

    for tta in (0, 3):
        ds = _Indexed(_FakeDS())
        tta_samples = int(0.25 * SR)          # forces >1 window on the 0.5s fake clips
        collate = make_collate(fake_fe, tta, tta_samples, with_labels=False)
        uniform = math.log(1.0 / len(COUNTRIES))
        out = torch.full((n_rows, len(COUNTRIES)), uniform)
        seen = torch.zeros(n_rows, dtype=torch.bool)

        # Batch boundaries deliberately land mid-run of dropped rows.
        for start in range(0, n_rows, 4):
            batch = collate([ds[i] for i in range(start, min(start + 4, n_rows))])
            if batch is None:
                continue
            feats, groups, rows, _ = batch
            n_win = feats["input_features"].shape[0]
            assert groups.shape[0] == n_win, (
                f"tta={tta}: {groups.shape[0]} group tags for {n_win} windows")
            assert int(groups.max()) == rows.shape[0] - 1, (
                f"tta={tta}: group ids run to {int(groups.max())} for {rows.shape[0]} clips")
            # Each window carries its clip's identity, so a correct merge recovers it exactly.
            lp = torch.zeros(n_win, len(COUNTRIES))
            for w, g in enumerate(groups.tolist()):
                lp[w, int(rows[g]) % 20] = 5.0
            merged = _merge_tta_groups(lp, groups, rows.shape[0])
            out[rows] = merged
            seen[rows] = True

        missed = set((~seen).nonzero().flatten().tolist())
        assert missed == bad, f"tta={tta}: dropped {sorted(missed)}, expected {sorted(bad)}"
        for i in range(n_rows):
            if i in bad:
                assert torch.allclose(out[i], torch.full((20,), uniform)), \
                    f"tta={tta}: row {i} was undecodable but is not the neutral fallback"
            else:
                got = int(out[i].argmax())
                assert got == i % 20, (
                    f"tta={tta}: ROW MISALIGNMENT -- row {i} carries class {got}, expected "
                    f"{i % 20}. A dropped clip shifted its neighbours.")
        print(f"    tta={tta}: 25 rows, {len(bad)} undecodable -> all survivors landed on their "
              "own row, failures held the neutral fallback")


def _selftest_infer_arch():
    base = {"classifier.weight": torch.zeros(20, 1280), "classifier.bias": torch.zeros(20),
            "encoder.foo": torch.zeros(2)}
    a = infer_arch(dict(base))
    assert not a["layer_mix"] and a["clf_in"] == 1280 and not a["lora_paths"]

    lm = dict(base, **{"layer_weights": torch.zeros(49)})
    a = infer_arch(lm)
    assert a["layer_mix"] and a["n_layer_weights"] == 49, a

    lora = dict(lm, **{"encoder.layers.0.self_attn.q_proj.lora_A": torch.zeros(16, 1280),
                       "encoder.layers.0.self_attn.q_proj.lora_B": torch.zeros(1280, 16)})
    a = infer_arch(lora)
    assert a["lora_paths"] == ["layers.0.self_attn.q_proj"], a["lora_paths"]
    assert a["lora_ranks"]["layers.0.self_attn.q_proj"] == 16
    print("    layer-mix / LoRA / pooling all recovered from state-dict keys alone")


def _selftest_writer():
    import tempfile
    n = 7
    logits = torch.randn(n, 20)
    logits[3] = torch.tensor([9.0] + [0.0] * 19)         # a known argmax to check the manifest
    with tempfile.TemporaryDirectory() as td:
        zip_p, dist = write_submission(logits, [f"clip{i}" for i in range(n)], td,
                                       {"selftest": True})
        assert sum(1 for _ in open(os.path.join(td, "logits.tsv"))) == n
        assert sum(1 for _ in open(os.path.join(td, "predictions.tsv"))) == n
        with zipfile.ZipFile(zip_p) as z:
            names = sorted(z.namelist())
            assert names == ["logits.tsv", "predictions.tsv"], names
            first = z.read("logits.tsv").decode().splitlines()[0].split("\t")
            assert len(first) == 20, len(first)
        man = open(os.path.join(td, "manifest.tsv")).read().splitlines()
        assert man[4].split("\t") == ["3", "clip3", "0", "MSA"], man[4]
        # Round-trip the logits: full float repr, so re-reading must be exact.
        back = np.array([[float(v) for v in l.split("\t")]
                         for l in open(os.path.join(td, "logits.tsv"))])
        assert np.array_equal(back, logits.numpy().astype(np.float64)), \
            "logits.tsv does not round-trip exactly"
        assert sum(dist.values()) == n
    print("    logits.tsv round-trips exactly, zip holds exactly the two required files at root")


def selftest():
    print("=" * 66)
    print("SELF-TEST -- no model, no network, no checkpoint")
    print("=" * 66)
    print("\n  [1/3] collate row/group bookkeeping + scatter alignment")
    _selftest_collate_and_scatter()
    print("\n  [2/3] architecture inference from checkpoint keys")
    _selftest_infer_arch()
    print("\n  [3/3] submission file format")
    _selftest_writer()
    print("\n" + "=" * 66)
    print("ALL PASSED. This checks alignment and format only -- it says nothing about")
    print("whether the checkpoint is good. That is what --verify does, on real audio.")
    print("=" * 66)


# ============================================================================
def main():
    if ARGS.selftest:
        selftest()
        return
    _hf_login()
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        # Worth shouting about: this encoder is ~1.9B parameters and eval clips run to 30s, so a
        # CPU forward is minutes PER BATCH. The run does not fail, it just appears to hang a few
        # percent into the first progress bar -- which is a much worse failure than stopping.
        print("\n" + "!" * 70)
        print("WARNING: no CUDA device found -- running the 1.9B-parameter encoder on CPU.")
        print("  Expect minutes per batch. A progress bar that seems frozen a few percent in")
        print("  is this, not a deadlock. Check `nvidia-smi` and that torch was installed with")
        print(f"  CUDA support: torch.version.cuda = {getattr(torch.version, 'cuda', None)!r}")
        print("!" * 70 + "\n")

    print(f"\nloading checkpoint: {ARGS.checkpoint}")
    state = load_state(ARGS.checkpoint)
    arch = infer_arch(state)
    print(f"  inferred from checkpoint: layer_mix={arch['layer_mix']} "
          f"(n={arch['n_layer_weights']}), classifier {arch['clf_in']}->{arch['n_labels']}, "
          f"lora_modules={len(arch['lora_paths'])}")
    if arch["n_labels"] != 20:
        print(f"\nFAIL: checkpoint head has {arch['n_labels']} classes; the submission format "
              "requires 20.")
        sys.exit(2)

    print(f"\nbuilding model from {HF_ID} ...")
    model, pool, layer_mix = build_model(state, arch)

    from transformers import AutoProcessor
    fe = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)

    verify_acc = None
    if ARGS.verify:
        verify_acc = verify(model, fe, ARGS.verify)

    print(f"\nloading test set: {TEST_REPO} [{TEST_SPLIT}]")
    from datasets import load_dataset
    test = load_dataset(TEST_REPO, split=TEST_SPLIT)
    print(f"  {len(test)} rows, columns: {test.column_names}")
    if "idx" not in test.column_names:
        print("  NOTE: no 'idx' column; the manifest will use row numbers only.")
    if len(test) != TEST_ROWS_EXPECTED:
        print(f"  NOTE: expected {TEST_ROWS_EXPECTED} rows, got {len(test)}. The dataset was "
              "updated; the submission follows the dataset, not the constant.")
    if ARGS.limit:
        test = test.select(range(min(ARGS.limit, len(test))))
        print(f"  --limit {ARGS.limit}: NOT A VALID SUBMISSION, pipeline debugging only")

    idxs = list(test["idx"]) if "idx" in test.column_names else list(range(len(test)))

    logits, _, missed = predict(model, test, fe, len(test), "test")
    if missed:
        print(f"\n  {len(missed)} test clip(s) could not be decoded: "
              f"{sorted(missed)[:20]}{' ...' if len(missed) > 20 else ''}")
        if not ARGS.allow_undecodable:
            print("\nFAIL: refusing to write a submission with undecodable test clips. They "
                  "would carry uniform log-probabilities (argmax -> MSA), which is invisible "
                  "in the output file. Investigate, or pass --allow-undecodable to accept it.")
            sys.exit(5)
        print("  --allow-undecodable: writing uniform log-probabilities for those rows.")

    tag = ARGS.tag or os.path.splitext(os.path.basename(ARGS.checkpoint))[0]
    out_dir = ARGS.out_dir or os.path.join("submissions", tag)
    meta = dict(tag=tag, checkpoint=os.path.abspath(ARGS.checkpoint),
                test_repo=TEST_REPO, test_split=TEST_SPLIT, n_rows=int(logits.shape[0]),
                countries=COUNTRIES, pool=pool, layer_mix=bool(layer_mix),
                lora_modules=len(arch["lora_paths"]),
                lora_alpha=ARGS.lora_alpha if arch["lora_paths"] else None,
                tta=ARGS.tta, precision=ARGS.precision,
                verify_n=ARGS.verify, verify_acc=verify_acc,
                undecodable=sorted(missed), limited=bool(ARGS.limit),
                logits_are="log_softmax (log-probabilities); a per-row constant shift from raw "
                           "logits, which leaves both argmax and Cavg's LLRs unchanged")
    zip_p, dist = write_submission(logits, idxs, out_dir, meta)

    print(f"\n{'='*66}\nwrote {out_dir}/")
    print(f"  logits.tsv             {logits.shape[0]} x 20")
    print(f"  predictions.tsv        {logits.shape[0]} ({ARGS.pred_format})")
    print(f"  predictions_labels.tsv country codes, for eyeballing")
    print(f"  manifest.tsv           row -> idx -> prediction")
    print(f"  submission.zip         <- upload this")
    print(f"\npredicted class distribution:")
    for c in COUNTRIES:
        n = dist.get(c, 0)
        bar = "#" * int(40 * n / max(1, max(dist.values())))
        print(f"  {c:<4s} {n:>4d}  {bar}")
    absent = [c for c in COUNTRIES if c not in dist]
    if absent:
        print(f"\n  NOTE: {len(absent)} class(es) never predicted: {absent}")
        print("        Expected for a balanced 878-clip test set only if those dialects are "
              "genuinely absent; otherwise it suggests a collapsed head.")
    if ARGS.limit:
        print("\n  REMINDER: --limit was set. This is not a valid submission.")
    print("=" * 66)


if __name__ == "__main__":
    main()
