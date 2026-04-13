#!/usr/bin/env python3
"""Suggest a simple XY coordinate transform to align a CustomDataset to KITTI.

This is useful when evaluating a KITTI-trained checkpoint on a dataset whose
LiDAR coordinate frame differs by a rotation/reflection.

We score a small set of candidate transforms by how many GT box centers (and
optionally naive full extents) fall inside a given POINT_CLOUD_RANGE.

Usage:
  python3 tools/find_best_lidar_transform.py \
    --label_dir /OpenPCDet/erod/labels \
    --pc_range 0 -39.68 -3 69.12 39.68 1

Then set in your yaml:
  DATA_CONFIG:
    POINT_CLOUD_RANGE: [0, -39.68, -3, 69.12, 39.68, 1]
    LIDAR_COORD_TRANSFORM:
      ENABLE: True
      ROT_Z_DEG: 90
      FLIP_X: False
      FLIP_Y: False
      TRANSLATION: [0, 0, 0]

Note: This does not change weights (no retraining). It only aligns coordinates.
"""

from __future__ import annotations

import argparse
import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple


@dataclass(frozen=True)
class Candidate:
    name: str
    rot_deg: float
    flip_x: bool
    flip_y: bool


def parse_pc_range(vals: List[float]) -> Tuple[float, float, float, float, float, float]:
    if len(vals) != 6:
        raise ValueError(f"--pc_range expects 6 floats, got {len(vals)}")
    x0, y0, z0, x1, y1, z1 = map(float, vals)
    if not (x0 < x1 and y0 < y1 and z0 < z1):
        raise ValueError(f"Invalid pc_range ordering: {vals}")
    return x0, y0, z0, x1, y1, z1


def in_range_center(x: float, y: float, z: float, r: Tuple[float, float, float, float, float, float]) -> bool:
    x0, y0, z0, x1, y1, z1 = r
    return x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1


def in_range_full_naive(
    x: float,
    y: float,
    z: float,
    dx: float,
    dy: float,
    dz: float,
    r: Tuple[float, float, float, float, float, float],
) -> bool:
    x0, y0, z0, x1, y1, z1 = r
    return (
        x0 <= x - dx / 2 <= x1
        and x0 <= x + dx / 2 <= x1
        and y0 <= y - dy / 2 <= y1
        and y0 <= y + dy / 2 <= y1
        and z0 <= z - dz / 2 <= z1
        and z0 <= z + dz / 2 <= z1
    )


def make_A(rot_deg: float, flip_x: bool, flip_y: bool):
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    A = ((c, -s), (s, c))
    fx = -1.0 if flip_x else 1.0
    fy = -1.0 if flip_y else 1.0
    # F @ A
    return ((fx * A[0][0], fx * A[0][1]), (fy * A[1][0], fy * A[1][1]))


def apply_A(x: float, y: float, A) -> Tuple[float, float]:
    return (A[0][0] * x + A[0][1] * y, A[1][0] * x + A[1][1] * y)


def iter_boxes(label_dir: Path) -> Iterable[Tuple[float, float, float, float, float, float]]:
    for fp in sorted(glob.glob(str(label_dir / "*.txt"))):
        with open(fp, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 8:
                    continue
                try:
                    x, y, z, dx, dy, dz = map(float, parts[:6])
                except Exception:
                    continue
                yield x, y, z, dx, dy, dz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", required=True)
    ap.add_argument(
        "--pc_range",
        nargs=6,
        type=float,
        default=[0.0, -39.68, -3.0, 69.12, 39.68, 1.0],
    )
    args = ap.parse_args()

    label_dir = Path(args.label_dir)
    pc_range = parse_pc_range(list(args.pc_range))

    boxes = list(iter_boxes(label_dir))
    if not boxes:
        raise SystemExit(f"No boxes found in {label_dir}")

    cands: List[Candidate] = []
    for deg in [0, 90, 180, 270]:
        cands.append(Candidate(name=f"rot{deg}", rot_deg=float(deg), flip_x=False, flip_y=False))
    cands.extend(
        [
            Candidate(name="flipx", rot_deg=0.0, flip_x=True, flip_y=False),
            Candidate(name="flipy", rot_deg=0.0, flip_x=False, flip_y=True),
            Candidate(name="flipxy", rot_deg=0.0, flip_x=True, flip_y=True),
        ]
    )

    print(f"Boxes: {len(boxes)}")
    print(f"pc_range: {list(pc_range)}")
    print("\nCandidate scores (higher is better):")

    best = None
    for cand in cands:
        A = make_A(cand.rot_deg, cand.flip_x, cand.flip_y)
        center_ok = 0
        full_ok = 0
        for x, y, z, dx, dy, dz in boxes:
            xt, yt = apply_A(x, y, A)
            if in_range_center(xt, yt, z, pc_range):
                center_ok += 1
            if in_range_full_naive(xt, yt, z, dx, dy, dz, pc_range):
                full_ok += 1
        score = (center_ok, full_ok)
        if best is None or score > best[0]:
            best = (score, cand)
        print(
            f"  {cand.name:6s}  ROT_Z_DEG={cand.rot_deg:>3.0f}  FLIP_X={str(cand.flip_x):5s}  FLIP_Y={str(cand.flip_y):5s}"
            f"  centers_in={center_ok:3d}/{len(boxes)}  full_in={full_ok:3d}/{len(boxes)}"
        )

    assert best is not None
    (center_ok, full_ok), cand = best
    print("\nBest candidate:")
    print(
        f"  {cand.name} (centers_in={center_ok}/{len(boxes)}, full_in={full_ok}/{len(boxes)})"
    )
    print("\nYAML snippet:")
    print("  LIDAR_COORD_TRANSFORM:")
    print("    ENABLE: True")
    print(f"    ROT_Z_DEG: {int(cand.rot_deg)}")
    print(f"    FLIP_X: {str(cand.flip_x)}")
    print(f"    FLIP_Y: {str(cand.flip_y)}")
    print("    TRANSLATION: [0.0, 0.0, 0.0]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
