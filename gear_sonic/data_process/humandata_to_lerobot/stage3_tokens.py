"""Stage 3: extract aligned action.motion_token from a deploy StateLogger run.

The C++ deploy (driven offline by dummy_lowstate_publisher.py + dummy_vr_streamer.py)
runs the real SONIC smpl-mode encoder and logs token_state to CSV.  This stage
pulls out the tokens that belong to the *streamed* episode and orders them 1:1
with the motion frames.

Identification (verified on episode_0): during the streamed episode the deploy
logs `encoder_mode == 2` (smpl) and `motion_name == "streamed"`, in a single
contiguous 50 Hz run.  The startup rows (g1 mode, planner/temporary motion) are
dropped.  Because both the deploy stream and the GMR retarget sample the same
100 Hz source at 50 Hz, token k corresponds to GMR motion frame k.

Usage:
    python -m gear_sonic.data_process.humandata_to_lerobot.stage3_tokens \\
        --logs-dir /tmp/tok_ep0 --out /tmp/tokens/episode_0.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SMPL_MODE = 2.0
STREAMED_NAME = "streamed"


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        return header, [row for row in r]


def extract_tokens(logs_dir: Path) -> np.ndarray:
    """Return the streamed smpl-mode tokens as (N, 64), ordered by frame."""
    th, tok = _read_csv(logs_dir / "token_state.csv")
    _, mode = _read_csv(logs_dir / "encoder_mode.csv")
    _, play = _read_csv(logs_dir / "motion_playing.csv")
    _, name = _read_csv(logs_dir / "motion_name.csv")

    # token columns are token_0..token_63 (after 5 metadata cols)
    tok_cols = [i for i, c in enumerate(th) if c.startswith("token_")]
    assert len(tok_cols) == 64, f"expected 64 token cols, got {len(tok_cols)}"

    n = min(len(tok), len(mode), len(play), len(name))
    out = []
    for i in range(n):
        is_smpl = float(mode[i][5]) == SMPL_MODE
        is_play = float(play[i][5]) == 1.0
        is_streamed = name[i][5] == STREAMED_NAME
        if is_smpl and is_play and is_streamed:
            out.append([float(tok[i][c]) for c in tok_cols])
    tokens = np.asarray(out, dtype=np.float64)
    if len(tokens) == 0:
        raise RuntimeError(
            f"No streamed smpl-mode tokens found in {logs_dir}. "
            "Check the deploy ran the encoder in smpl mode (encoder_mode==2)."
        )
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", required=True, help="Deploy StateLogger CSV dir")
    ap.add_argument("--out", required=True, help="Output .npz (key 'motion_token')")
    args = ap.parse_args()

    tokens = extract_tokens(Path(args.logs_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, motion_token=tokens)
    print(f"Extracted {tokens.shape[0]} smpl tokens (dim {tokens.shape[1]}) -> {out}")
    print(f"  value range [{tokens.min():.4f}, {tokens.max():.4f}]")


if __name__ == "__main__":
    main()
