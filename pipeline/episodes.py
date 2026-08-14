"""Shared episode listing for the v2 pipeline. Pure stdlib (imports in all three envs).

Episode set + per-episode head/tail trim come from `<ds>/episode_offsets.csv` when it
exists. With no CSV (or `--no-offsets`), episodes are discovered from `<ds>/body_data/
episode_*.hdf5`, sorted by id, and the flat `--head`/`--tail` (default 0) apply to all.
"""
import csv, glob, os, re


def add_args(ap, head_tail=True):
    """Register the offsets flags shared by every pipeline script."""
    ap.add_argument("--offsets", default=None, help="offsets CSV (default <ds>/episode_offsets.csv)")
    ap.add_argument("--no-offsets", action="store_true",
                    help="ignore the offsets CSV: use every episode in <ds>/body_data")
    if head_tail:
        ap.add_argument("--head", type=float, default=0.0, help="flat head trim (s) when no offsets")
        ap.add_argument("--tail", type=float, default=0.0, help="flat tail trim (s) when no offsets")


def load_episodes(ds, offsets=None, no_offsets=False, head=0.0, tail=0.0):
    """-> [(orig_id, head_s, tail_s), ...] in dataset order."""
    path = offsets or os.path.join(ds, "episode_offsets.csv")
    if not no_offsets and os.path.exists(path):
        with open(path) as f:
            return [(int(r["orig_id"]), float(r["head_s"]), float(r["tail_s"])) for r in csv.DictReader(f)]
    ids = sorted(int(m.group(1)) for m in
                 (re.search(r"episode_(\d+)\.hdf5$", p)
                  for p in glob.glob(os.path.join(ds, "body_data", "episode_*.hdf5"))) if m)
    if not ids:
        raise SystemExit(f"no offsets CSV at {path} and no {ds}/body_data/episode_*.hdf5")
    return [(i, head, tail) for i in ids]


def assembled_order(ds, **kw):
    """Episodes that actually have a cl_collect rollout, in dataset (new_id) order.
    Index in this list == the episode_index written by assemble_v2.py."""
    return [e for e in load_episodes(ds, **kw)
            if os.path.exists(os.path.join(ds, "cl_collect", f"episode_{e[0]}.npz"))]


def from_args(a, assembled=False):
    """Build the episode list straight from parsed argparse args."""
    kw = dict(offsets=a.offsets, no_offsets=a.no_offsets,
              head=getattr(a, "head", 0.0), tail=getattr(a, "tail", 0.0))
    return (assembled_order if assembled else load_episodes)(a.ds, **kw)


def head_of(ds, orig, fallback=0.0):
    """Head trim actually used when cl_encode cropped this episode (stamped into the npz).
    Falls back to `fallback` for npz written before the stamp existed."""
    import numpy as np
    d = np.load(os.path.join(ds, "cl_collect", f"episode_{orig}.npz"))
    return float(d["head_s"]) if "head_s" in d.files else float(fallback)
