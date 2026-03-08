"""
Visualize ground-truth labels on point clouds as BEV images,
so they can be compared side-by-side with demo.py model predictions.

Usage:
    python3 visualize_labels.py \
        --data_path /OpenPCDet/erod/points \
        --label_path /OpenPCDet/erod/labels \
        --ext .npy
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── BEV drawing (same as demo.py) ────────────────────────────────────────────

def get_box_corners_2d(cx, cy, dx, dy, heading):
    """Get 4 corners of a rotated 2D box."""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    half_dx = dx / 2
    half_dy = dy / 2

    corners = np.array([
        [-half_dx, -half_dy],
        [ half_dx, -half_dy],
        [ half_dx,  half_dy],
        [-half_dx,  half_dy],
    ])

    rot = np.array([[cos_h, -sin_h],
                    [sin_h,  cos_h]])
    corners = corners @ rot.T
    corners[:, 0] += cx
    corners[:, 1] += cy
    return corners


def draw_bev_image(points, gt_boxes, gt_names, save_path,
                   point_range=(-50, -50, 50, 50)):
    """
    Draw bird's-eye view of point cloud with ground-truth boxes.
    """
    # Filter points within range
    mask = ((points[:, 0] > point_range[0]) & (points[:, 0] < point_range[2]) &
            (points[:, 1] > point_range[1]) & (points[:, 1] < point_range[3]))
    points = points[mask]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)

    # Plot points (BEV: x=forward, y=left)
    ax.scatter(points[:, 1], points[:, 0], s=0.1, c='white', alpha=0.5)

    # Class → colour mapping
    class_colors = {
        'Car': 'lime', 'Vehicle': 'lime',
        'Pedestrian': 'cyan',
        'Cyclist': 'yellow',
    }
    default_color = 'magenta'

    # Draw GT boxes
    if gt_boxes is not None and len(gt_boxes) > 0:
        for i in range(len(gt_boxes)):
            box = gt_boxes[i]
            x, y, z, dx, dy, dz, heading = box[:7]
            name = gt_names[i] if i < len(gt_names) else '?'
            color = class_colors.get(name, default_color)

            corners = get_box_corners_2d(x, y, dx, dy, heading)
            polygon = plt.Polygon(corners[:, [1, 0]], fill=False,
                                  edgecolor=color, linewidth=1.5)
            ax.add_patch(polygon)

            ax.text(y, x, f'{name}\n(GT)',
                    color=color, fontsize=5, ha='center', va='bottom')

    ax.set_xlim(point_range[1], point_range[3])
    ax.set_ylim(point_range[0], point_range[2])
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    ax.set_title("Bird's Eye View  —  Ground Truth")
    fig.tight_layout()
    fig.savefig(str(save_path), facecolor='black')
    plt.close(fig)


# ── Label parsing ────────────────────────────────────────────────────────────

def parse_label_file(label_path):
    """
    Parse a label file in either format:
      • Custom (8 cols):  x y z dx dy dz heading ClassName
      • KITTI  (15 cols): ClassName trunc occ alpha bb1 bb2 bb3 bb4 h w l x y z ry
    Returns:
        gt_boxes  : np.ndarray  (N, 7) — x y z dx dy dz heading
        gt_names  : list[str]
    """
    gt_boxes = []
    gt_names = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            # Detect format by checking whether the first token is a number
            first_is_number = True
            try:
                float(parts[0])
            except ValueError:
                first_is_number = False

            if first_is_number and len(parts) == 8:
                # Custom format: x y z dx dy dz heading ClassName
                x, y, z, dx, dy, dz, heading = [float(v) for v in parts[:7]]
                name = parts[7]
            elif not first_is_number and len(parts) >= 15:
                # KITTI format: class trunc occ alpha bb(4) h w l x y z ry
                name = parts[0]
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                heading = float(parts[14])
                dx, dy, dz = l, w, h  # KITTI h/w/l → dx/dy/dz
            else:
                print(f'  [WARN] Skipping unrecognised label line: {line.strip()}')
                continue

            gt_boxes.append([x, y, z, dx, dy, dz, heading])
            gt_names.append(name)

    if gt_boxes:
        return np.array(gt_boxes, dtype=np.float32), gt_names
    return np.zeros((0, 7), dtype=np.float32), []


# ── Filename → label matching ────────────────────────────────────────────────

def extract_frame_id(filename_stem):
    """
    Try to extract a numeric frame ID from a file stem.
    Examples:
        '192'  →  '192'
        '2025-11-18-13-51-31_Velodyne-VLP-16-Data (Frame 193)'  →  '193'
    """
    # If the stem is purely digits, use it directly
    if filename_stem.isdigit():
        return filename_stem

    # Look for "Frame NNN" pattern
    m = re.search(r'Frame\s+(\d+)', filename_stem)
    if m:
        return m.group(1)

    # Fall back to the last number in the string
    nums = re.findall(r'\d+', filename_stem)
    if nums:
        return nums[-1]

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualise ground-truth labels as BEV images')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Directory containing point cloud files')
    parser.add_argument('--label_path', type=str, required=True,
                        help='Directory containing label .txt files')
    parser.add_argument('--ext', type=str, default='.npy',
                        help='Point cloud file extension (.npy or .bin)')
    parser.add_argument('--output_dir', type=str,
                        default='/OpenPCDet/output/gt_images',
                        help='Where to save the BEV images')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    label_path = Path(args.label_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather point cloud files
    pc_files = sorted(glob.glob(str(data_path / f'*{args.ext}')))
    print(f'Found {len(pc_files)} point cloud files in {data_path}')

    for pc_file in pc_files:
        stem = Path(pc_file).stem
        frame_id = extract_frame_id(stem)

        # Try to find a matching label file
        label_file = label_path / f'{frame_id}.txt' if frame_id else None
        if label_file is None or not label_file.exists():
            print(f'[{stem}]  No label file found (frame_id={frame_id}), skipping.')
            continue
        
        gt_boxes, gt_names = parse_label_file(label_file)
        print(f'[{stem}]  Loaded {len(gt_names)} GT box(es) from {label_file.name}')

        # Load points
        if args.ext == '.bin':
            points = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
        elif args.ext == '.npy':
            points = np.load(pc_file)
        else:
            raise ValueError(f'Unsupported extension: {args.ext}')

        save_path = output_dir / f'{stem}.png'
        draw_bev_image(points, gt_boxes, gt_names, save_path)
        print(f'  → Saved {save_path}')

    print(f'\nDone. GT images saved to {output_dir}')


if __name__ == '__main__':
    main()
