#!/usr/bin/env python3
"""Filter OpenPCDet CustomDataset label .txt files by POINT_CLOUD_RANGE.

Assumes each label line is:
  x y z dx dy dz heading [optional extra floats...] class_name

This script keeps a label if its box center (x, y, z) lies within the
provided POINT_CLOUD_RANGE [x_min, y_min, z_min, x_max, y_max, z_max].

Example:
  python3 tools/filter_labels_by_range.py \
    --input_label_dir /OpenPCDet/erod/labels \
    --output_label_dir /OpenPCDet/erod/labels_forward \
    --pc_range -32 0 -3 32 64.64 1

Notes:
- Only .txt files are processed.
- Output files always get created (possibly empty) to preserve the split.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_pc_range(values: List[float]) -> Tuple[float, float, float, float, float, float]:
    if len(values) != 6:
        raise ValueError(f"--pc_range expects 6 floats, got {len(values)}")
    x_min, y_min, z_min, x_max, y_max, z_max = map(float, values)
    if not (x_min < x_max and y_min < y_max and z_min < z_max):
        raise ValueError(f"Invalid pc_range ordering: {values}")
    return x_min, y_min, z_min, x_max, y_max, z_max


def iter_label_files(label_dir: Path) -> Iterable[Path]:
    return sorted(p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def box_center_in_range(
    x: float,
    y: float,
    z: float,
    pc_range: Tuple[float, float, float, float, float, float],
) -> bool:
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    return (x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max)


def filter_label_lines(
    lines: List[str],
    pc_range: Tuple[float, float, float, float, float, float],
) -> Tuple[List[str], int, int]:
    kept: List[str] = []
    parsed = 0
    dropped = 0

    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 8:
            # Needs at least: x y z dx dy dz heading class
            dropped += 1
            continue

        # Parse center xyz from first 3 floats.
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
        except ValueError:
            dropped += 1
            continue

        parsed += 1
        if box_center_in_range(x, y, z, pc_range):
            kept.append(s)
        else:
            dropped += 1

    return kept, parsed, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter label .txt files by POINT_CLOUD_RANGE")
    parser.add_argument(
        "--input_label_dir",
        required=True,
        help="Path to folder containing input .txt label files",
    )
    parser.add_argument(
        "--output_label_dir",
        required=True,
        help="Path to folder to write filtered .txt label files",
    )
    parser.add_argument(
        "--pc_range",
        nargs=6,
        type=float,
        default=[-32.0, 0.0, -3.0, 32.0, 64.64, 1.0],
        help="POINT_CLOUD_RANGE as 6 floats: x_min y_min z_min x_max y_max z_max (default: old forward-only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory",
    )

    args = parser.parse_args()

    in_dir = Path(args.input_label_dir)
    out_dir = Path(args.output_label_dir)
    pc_range = parse_pc_range(list(args.pc_range))

    if not in_dir.exists():
        raise FileNotFoundError(f"Input label dir does not exist: {in_dir}")
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Input label dir is not a directory: {in_dir}")

    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output dir already exists: {out_dir} (use --overwrite to allow)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_label_files(in_dir))
    if not files:
        raise FileNotFoundError(f"No .txt label files found in: {in_dir}")

    total_in_lines = 0
    total_kept = 0
    total_parsed = 0
    total_dropped = 0

    for fp in files:
        with fp.open("r") as f:
            lines = f.readlines()

        kept_lines, parsed, dropped = filter_label_lines(lines, pc_range)

        total_in_lines += sum(1 for l in lines if l.strip())
        total_kept += len(kept_lines)
        total_parsed += parsed
        total_dropped += dropped

        out_path = out_dir / fp.name
        with out_path.open("w") as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")
            else:
                # Keep file present but empty.
                f.write("")

    print("Filtered labels")
    print(f"  input_dir:  {in_dir}")
    print(f"  output_dir: {out_dir}")
    print(f"  pc_range:   {list(pc_range)}")
    print(f"  files:      {len(files)}")
    print(f"  in_lines:   {total_in_lines}")
    print(f"  parsed:     {total_parsed}")
    print(f"  kept:       {total_kept}")
    print(f"  dropped:    {total_dropped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
