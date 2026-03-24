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
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from visual_utils.multiview_renderer import MultiViewRenderer


RENDERER = MultiViewRenderer()


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
                   point_range=(-50, -50, 50, 50), z_range=None):
    """
    Draw a two-panel image:
      1) BEV (X-Y)
      2) Front view (X-Z)
    z_range: optional (z_min, z_max) tuple to filter points by height.
    """
    # Filter points within XY range for both panels.
    mask = ((points[:, 0] > point_range[0]) & (points[:, 0] < point_range[2]) &
            (points[:, 1] > point_range[1]) & (points[:, 1] < point_range[3]))
    if z_range is not None:
        mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    points = points[mask]

    if z_range is None:
        # Keep front view stable even with outlier points.
        z_min_plot, z_max_plot = -3.0, 3.0
    else:
        z_min_plot, z_max_plot = z_range

    fig, (ax_bev, ax_front) = plt.subplots(1, 2, figsize=(18, 9), dpi=150)

    # Class -> colour mapping
    class_colors = {
        'Car': 'lime', 'Vehicle': 'lime',
        'Pedestrian': 'cyan',
        'Cyclist': 'yellow',
    }
    default_color = 'magenta'

    # BEV panel (x=forward, y=left)
    ax_bev.scatter(points[:, 1], points[:, 0], s=0.1, c='white', alpha=0.5)

    # Front panel (x horizontal, z vertical)
    ax_front.scatter(points[:, 0], points[:, 2], s=0.1, c='white', alpha=0.5)

    if gt_boxes is not None and len(gt_boxes) > 0:
        for i in range(len(gt_boxes)):
            x, y, z, dx, dy, dz, heading = gt_boxes[i][:7]
            name = gt_names[i] if i < len(gt_names) else '?'
            color = class_colors.get(name, default_color)

            # BEV rotated footprint
            corners = get_box_corners_2d(x, y, dx, dy, heading)
            polygon = plt.Polygon(corners[:, [1, 0]], fill=False,
                                  edgecolor=color, linewidth=1.5)
            ax_bev.add_patch(polygon)
            ax_bev.text(y, x, f'{name}\n(GT)',
                        color=color, fontsize=5, ha='center', va='bottom')

            # Front view approximation: project oriented XY box to X extent.
            # This keeps the panel simple and robust for quick debugging.
            half_x_extent = 0.5 * (abs(np.cos(heading)) * dx + abs(np.sin(heading)) * dy)
            x0 = x - half_x_extent
            z0 = z - dz / 2.0
            rect = plt.Rectangle((x0, z0), 2.0 * half_x_extent, dz,
                                 fill=False, edgecolor=color, linewidth=1.5)
            ax_front.add_patch(rect)
            ax_front.text(x, z + dz / 2.0, f'{name}\n(GT)',
                          color=color, fontsize=5, ha='center', va='bottom')

    ax_bev.set_xlim(point_range[1], point_range[3])
    ax_bev.set_ylim(point_range[0], point_range[2])
    ax_bev.set_facecolor('black')
    ax_bev.set_aspect('equal')
    ax_bev.set_xlabel('Y (m)')
    ax_bev.set_ylabel('X (m)')
    ax_bev.set_title("BEV (X-Y)  —  Ground Truth")

    ax_front.set_xlim(point_range[0], point_range[2])
    ax_front.set_ylim(z_min_plot, z_max_plot)
    ax_front.set_facecolor('black')
    ax_front.set_aspect('auto')
    ax_front.set_xlabel('X (m)')
    ax_front.set_ylabel('Z (m)')
    ax_front.set_title('Front View (X-Z)  —  Ground Truth')

    fig.tight_layout()
    fig.savefig(str(save_path), facecolor='black')
    plt.close(fig)


def sanitize_name(name):
    """Keep filenames safe and compact."""
    return re.sub(r'[^A-Za-z0-9_-]+', '-', str(name))


def extract_points_in_box(points, box, xy_margin=0.4, z_margin=0.2):
    """
    Select points that fall inside a rotated 3D box (with a small margin).
    Box format: [x, y, z, dx, dy, dz, heading].
    """
    x, y, z, dx, dy, dz, heading = box[:7]
    shifted = points[:, :3] - np.array([x, y, z], dtype=np.float32)

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    # Rotate world -> local box frame (inverse rotation around Z).
    local_x = cos_h * shifted[:, 0] + sin_h * shifted[:, 1]
    local_y = -sin_h * shifted[:, 0] + cos_h * shifted[:, 1]
    local_z = shifted[:, 2]

    mask = (
        (np.abs(local_x) <= dx / 2.0 + xy_margin) &
        (np.abs(local_y) <= dy / 2.0 + xy_margin) &
        (np.abs(local_z) <= dz / 2.0 + z_margin)
    )

    return np.stack([local_x[mask], local_y[mask], local_z[mask]], axis=1)


def draw_instance_views(local_points, box, class_name, save_path):
    """
    Draw object-centered views in local box coordinates:
      - front: X-Z
      - side : Y-Z
      - top  : X-Y
    """
    _, _, _, dx, dy, dz, _ = box[:7]

    # Stable axis ranges around the GT box extent.
    pad_xy = 0.6
    pad_z = 0.4

    x_lim = (-dx / 2.0 - pad_xy, dx / 2.0 + pad_xy)
    y_lim = (-dy / 2.0 - pad_xy, dy / 2.0 + pad_xy)
    z_lim = (-dz / 2.0 - pad_z, dz / 2.0 + pad_z)

    fig, (ax_front, ax_side, ax_top) = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    for ax in (ax_front, ax_side, ax_top):
        ax.set_facecolor('black')

    # Front view (X-Z)
    if local_points.shape[0] > 0:
        ax_front.scatter(local_points[:, 0], local_points[:, 2], s=0.2, c='white', alpha=0.8)
    front_rect = plt.Rectangle((x_lim[0] + pad_xy, z_lim[0] + pad_z), dx, dz,
                               fill=False, edgecolor='lime', linewidth=1.5)
    ax_front.add_patch(front_rect)
    ax_front.set_xlim(*x_lim)
    ax_front.set_ylim(*z_lim)
    ax_front.set_xlabel('Local X (m)')
    ax_front.set_ylabel('Local Z (m)')
    ax_front.set_title(f'Front (X-Z) - {class_name}')

    # Side view (Y-Z)
    if local_points.shape[0] > 0:
        ax_side.scatter(local_points[:, 1], local_points[:, 2], s=0.2, c='white', alpha=0.8)
    side_rect = plt.Rectangle((y_lim[0] + pad_xy, z_lim[0] + pad_z), dy, dz,
                              fill=False, edgecolor='cyan', linewidth=1.5)
    ax_side.add_patch(side_rect)
    ax_side.set_xlim(*y_lim)
    ax_side.set_ylim(*z_lim)
    ax_side.set_xlabel('Local Y (m)')
    ax_side.set_ylabel('Local Z (m)')
    ax_side.set_title(f'Side (Y-Z) - {class_name}')

    # Top view (X-Y)
    if local_points.shape[0] > 0:
        ax_top.scatter(local_points[:, 0], local_points[:, 1], s=0.2, c='white', alpha=0.8)
    top_rect = plt.Rectangle((x_lim[0] + pad_xy, y_lim[0] + pad_xy), dx, dy,
                             fill=False, edgecolor='yellow', linewidth=1.5)
    ax_top.add_patch(top_rect)
    ax_top.set_xlim(*x_lim)
    ax_top.set_ylim(*y_lim)
    ax_top.set_aspect('equal')
    ax_top.set_xlabel('Local X (m)')
    ax_top.set_ylabel('Local Y (m)')
    ax_top.set_title(f'Top (X-Y) - {class_name}')

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


# ── Worker function (runs in a subprocess) ───────────────────────────────────

def _process_one(pc_file, label_dir, output_dir, ext, z_range):
    """Load one point cloud + label, render frame image and per-instance views."""
    stem = Path(pc_file).stem
    frame_id = extract_frame_id(stem)

    label_file = Path(label_dir) / f'{frame_id}.txt' if frame_id else None
    if label_file is None or not label_file.exists():
        return None  # skip silently

    gt_boxes, gt_names = parse_label_file(label_file)

    if ext == '.bin':
        points = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
    elif ext == '.npy':
        points = np.load(pc_file)
    else:
        raise ValueError(f'Unsupported extension: {ext}')

    # Frame-level image (existing output).
    frame_save_path = Path(output_dir) / f'{stem}.png'
    frame_names = [f'{name} (GT)' for name in gt_names]
    RENDERER.draw_frame(
        points=points,
        boxes=gt_boxes,
        names=frame_names,
        save_path=frame_save_path,
        z_range=z_range,
        bev_title='BEV (X-Y)  -  Ground Truth',
        front_title='Front View (X-Z)  -  Ground Truth'
    )

    # Per-object images.
    instances_dir = Path(output_dir) / 'instances'
    instances_dir.mkdir(parents=True, exist_ok=True)
    instance_count = 0
    frame_key = frame_id if frame_id is not None else stem

    RENDERER.render_instances(
        points=points,
        boxes=gt_boxes,
        names=list(gt_names),
        frame_key=frame_key,
        instances_dir=instances_dir
    )
    instance_count = len(gt_boxes)

    return (
        f'[{stem}]  frame -> {frame_save_path.name}, '
        f'instances -> {instance_count}'
    )


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
    parser.add_argument('--z_min', type=float, default=None,
                        help='Min height (Z) of points to plot')
    parser.add_argument('--z_max', type=float, default=None,
                        help='Max height (Z) of points to plot')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    label_path = Path(args.label_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z_range = None
    if args.z_min is not None or args.z_max is not None:
        z_range = (args.z_min if args.z_min is not None else -float('inf'),
                   args.z_max if args.z_max is not None else  float('inf'))

    # Gather point cloud files
    pc_files = sorted(glob.glob(str(data_path / f'*{args.ext}')))
    print(f'Found {len(pc_files)} point cloud files in {data_path}')

    num_workers = args.workers if args.workers else min(cpu_count(), len(pc_files), 16)
    worker_fn = partial(_process_one,
                        label_dir=str(label_path),
                        output_dir=str(output_dir),
                        ext=args.ext,
                        z_range=z_range)

    done = 0
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(worker_fn, pc_files):
            if result is not None:
                done += 1
                print(result)

    print(f'\nDone. {done} GT images saved to {output_dir}  ({num_workers} workers)')


if __name__ == '__main__':
    main()
