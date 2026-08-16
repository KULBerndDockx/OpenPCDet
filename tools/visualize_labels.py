"""
Visualize ground-truth labels on point clouds as BEV images,
so they can be compared side-by-side with demo.py model predictions.

Usage:
    python3 visualize_labels.py \
        --data_path /OpenPCDet/erod/points \
        --label_path /OpenPCDet/erod/labels \
        --ext .npy

NuScenes usage (no per-frame .txt labels; uses OpenPCDet info .pkl):
    python3 visualize_labels.py \
        --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini \
        --nuscenes_info /OpenPCDet/datasets/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl \
        --output_dir /OpenPCDet/output/gt_images_nuscenes



python3 visualize_labels.py --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --nuscenes_info /OpenPCDet/datasets/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl --ext .npy --single_image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/D/S_D_N/default/eval/epoch_7862/val/default/result.pkl   --z_min -1.0 --z_max 3.0


NuScenes usage (no per-frame .txt labels; uses OpenPCDet info .pkl):
    python3 visualize_labels.py --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini --nuscenes_info /OpenPCDet/datasets/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl --output_dir /OpenPCDet/output/gt_images_nuscenes_000
"""

import argparse
import glob
from operator import index
import pickle
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

KITTI_SINGLE_IMAGE_ID = '000422'
NUSCENES_SINGLE_IMAGE_STEM = 'n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151616997246.pcd'
#KITTI_SINGLE_IMAGE_ID = 'n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151609547766.pcd'


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
                   point_range=(-50, -50, 50, 50), z_range=(-3.0, 1.0)):
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
        z_min_plot, z_max_plot = -3.0, 1.0
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

def parse_custom_label_file(label_path):
    """
    Parse LiDAR-native labels in the current EROD-style format:
      x y z dx dy dz heading ClassName
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

            first_is_number = True
            try:
                float(parts[0])
            except ValueError:
                first_is_number = False

            if first_is_number and len(parts) == 8:
                x, y, z, dx, dy, dz, heading = [float(v) for v in parts[:7]]
                name = parts[7]
            else:            
                print(f'  [WARN] Skipping unrecognised label line: {line.strip()}')
                continue

            gt_boxes.append([x, y, z, dx, dy, dz, heading])
            gt_names.append(name)

    if gt_boxes:
        return np.array(gt_boxes, dtype=np.float32), gt_names
    return np.zeros((0, 7), dtype=np.float32), []


def _load_kitti_calibration(calib_path):
    """Load KITTI calibration matrices needed for camera-to-LiDAR conversion."""
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    r0 = np.array(lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
    v2c = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
    return r0, v2c


def _rect_to_lidar(pts_rect, r0, v2c):
    pts_rect_hom = np.hstack((pts_rect, np.ones((pts_rect.shape[0], 1), dtype=np.float32)))
    r0_ext = np.hstack((r0, np.zeros((3, 1), dtype=np.float32)))
    r0_ext = np.vstack((r0_ext, np.zeros((1, 4), dtype=np.float32)))
    r0_ext[3, 3] = 1.0

    v2c_ext = np.vstack((v2c, np.zeros((1, 4), dtype=np.float32)))
    v2c_ext[3, 3] = 1.0

    pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(r0_ext, v2c_ext).T))
    return pts_lidar[:, :3]


def parse_kitti_label_file(label_path, calib_path):
    """
    Parse raw KITTI `label_2` files and convert boxes to LiDAR coordinates.
    Returns:
        gt_boxes  : np.ndarray  (N, 7) — x y z dx dy dz heading
        gt_names  : list[str]
    """
    r0, v2c = _load_kitti_calibration(calib_path)

    gt_boxes = []
    gt_names = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            name = parts[0]
            if name == 'DontCare' or len(parts) < 15:
                continue

            h = float(parts[8])
            w = float(parts[9])
            l = float(parts[10])
            x = float(parts[11])
            y = float(parts[12])
            z = float(parts[13])
            heading = float(parts[14])

            loc_lidar = _rect_to_lidar(np.array([[x, y, z]], dtype=np.float32), r0, v2c)[0]
            loc_lidar[2] += h / 2.0
            box_heading = -(np.pi / 2.0 + heading)

            gt_boxes.append([loc_lidar[0], loc_lidar[1], loc_lidar[2], l, w, h, box_heading])
            gt_names.append(name)

    if gt_boxes:
        return np.array(gt_boxes, dtype=np.float32), gt_names
    return np.zeros((0, 7), dtype=np.float32), []


def parse_label_file(label_path):
    """Backward-compatible alias for the current EROD-style label format."""
    return parse_custom_label_file(label_path)


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

    first_line = None
    with open(label_file, 'r') as f:
        for line in f:
            if line.strip():
                first_line = line.strip().split()
                break

    if first_line is None:
        return None

    first_is_number = True
    try:
        float(first_line[0])
    except ValueError:
        first_is_number = False

    if first_is_number:
        gt_boxes, gt_names = parse_custom_label_file(label_file)
    else:
        calib_file = Path(label_dir).parent / 'calib' / f'{frame_id}.txt'
        if not calib_file.exists():
            print(f'  [WARN] Missing KITTI calib file: {calib_file}')
            return None
        gt_boxes, gt_names = parse_kitti_label_file(label_file, calib_file)

    if ext == '.bin':
        raw = np.fromfile(pc_file, dtype=np.float32)
        if raw.size % 5 == 0:
            points = raw.reshape(-1, 5)
        elif raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
        else:
            raise ValueError(f"Unexpected point format: {pc_file}")
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

    """RENDERER.render_instances(
        points=points,
        boxes=gt_boxes,
        names=list(gt_names),
        frame_key=frame_key,
        instances_dir=instances_dir
    )"""
    instance_count = len(gt_boxes)

    return (
        f'[{stem}]  frame -> {frame_save_path.name}, '
        f'instances -> {instance_count}'
    )


def _process_one_nuscenes(sample, output_dir, z_range):
    """Load one NuScenes point cloud and render GT from an info dict."""
    pc_file = sample['pc_file']
    stem = sample['stem']
    frame_key = sample['frame_key']
    gt_boxes = sample['gt_boxes']
    gt_names = sample['gt_names']

    # NuScenes lidar .bin is typically (N, 5) float32.
    raw = np.fromfile(str(pc_file), dtype=np.float32)
    if raw.size % 5 == 0:
        points = raw.reshape(-1, 5)
    elif raw.size % 4 == 0:
        points = raw.reshape(-1, 4)
    else:
        raise ValueError(f"Unexpected point format: {pc_file}")

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

    instances_dir = Path(output_dir) / 'instances'
    instances_dir.mkdir(parents=True, exist_ok=True)

    RENDERER.render_instances(
        points=points,
        boxes=gt_boxes,
        names=list(gt_names),
        frame_key=frame_key,
        instances_dir=instances_dir
    )

    return (
        f'[{stem}]  frame -> {frame_save_path.name}, '
        f'instances -> {len(gt_boxes)}'
    )


def _load_nuscenes_samples(nuscenes_root: Path, info_path: Path):
    """Return a lightweight list of {pc_file, gt_boxes, gt_names, ...}.

    Expects `info_path` to be an OpenPCDet-generated `nuscenes_infos_*.pkl`.
    Each entry contains:
      - lidar_path (relative)
      - token
      - gt_boxes (N, 9) or (N, 7+) with x,y,z,dx,dy,dz,yaw in the first 7
      - gt_names
    """
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    samples = []
    for info in infos:
        rel_lidar = info.get('lidar_path', None)
        if rel_lidar is None:
            continue

        pc_file = nuscenes_root / rel_lidar
        if not pc_file.exists():
            # If the caller passed `data_path` one level higher, try resolving
            # relative paths against that.
            alt = nuscenes_root.parent / rel_lidar
            if alt.exists():
                pc_file = alt
            else:
                continue

        gt_boxes = info.get('gt_boxes', None)
        if gt_boxes is None:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
        else:
            gt_boxes = np.asarray(gt_boxes, dtype=np.float32)

        gt_names = info.get('gt_names', None)
        if gt_names is None:
            gt_names = []
        else:
            gt_names = [str(x) for x in list(gt_names)]

        token = info.get('token', None)
        stem = Path(rel_lidar).stem
        frame_key = token if token is not None else stem

        samples.append({
            'pc_file': str(pc_file),
            'stem': stem,
            'frame_key': frame_key,
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
        })

    return samples


def _load_detection_results(result_path: Path):
    """Load an OpenPCDet `result.pkl` file into a frame_id -> detection dict map."""
    with open(result_path, 'rb') as f:
        det_annos = pickle.load(f)

    if not isinstance(det_annos, list):
        raise ValueError(f'Unexpected result.pkl content: {type(det_annos)}')

    det_by_frame = {}
    for det in det_annos:
        frame_id = det.get('frame_id', None)
        if frame_id is None:
            continue
        det_by_frame[str(frame_id)] = det

    return det_by_frame


def _parse_prediction_anno(det_anno):
    """Convert one detection record to box and label arrays for rendering."""
    boxes = np.asarray(det_anno.get('boxes_lidar', np.zeros((0, 7))), dtype=np.float32)
    names = np.asarray(det_anno.get('name', []), dtype=object).astype(str)
    scores = np.asarray(det_anno.get('score', []), dtype=np.float32)

    if boxes.ndim != 2:
        boxes = np.zeros((0, 7), dtype=np.float32)

    display_names = []
    for idx, name in enumerate(names[:len(boxes)]):
        if idx < len(scores):
            display_names.append(f'{name} {scores[idx]:.2f}')
        else:
            display_names.append(str(name))

    return boxes, display_names


def _process_one_compare(pc_file, label_dir, output_dir, ext, z_range, det_by_frame):
    """Load one point cloud, ground truth, and detections, then render a comparison."""
    stem = Path(pc_file).stem
    frame_id = extract_frame_id(stem)

    label_file = Path(label_dir) / f'{frame_id}.txt' if frame_id else None
    if label_file is None or not label_file.exists():
        return None

    first_line = None
    with open(label_file, 'r') as f:
        for line in f:
            if line.strip():
                first_line = line.strip().split()
                break

    if first_line is None:
        return None

    first_is_number = True
    try:
        float(first_line[0])
    except ValueError:
        first_is_number = False

    if first_is_number:
        gt_boxes, gt_names = parse_custom_label_file(label_file)
    else:
        calib_file = Path(label_dir).parent / 'calib' / f'{frame_id}.txt'
        if not calib_file.exists():
            print(f'  [WARN] Missing KITTI calib file: {calib_file}')
            return None
        gt_boxes, gt_names = parse_kitti_label_file(label_file, calib_file)

    if ext == '.bin':
        raw = np.fromfile(pc_file, dtype=np.float32)
        if raw.size % 5 == 0:
            points = raw.reshape(-1, 5)
        elif raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
        else:
            raise ValueError(f"Unexpected point format: {pc_file}")
    elif ext == '.npy':
        points = np.load(pc_file)
    else:
        raise ValueError(f'Unsupported extension: {ext}')

    det_anno = det_by_frame.get(str(frame_id), None)
    if det_anno is None:
        return None

    pred_boxes, pred_names = _parse_prediction_anno(det_anno)

    compare_dir = Path(output_dir) / 'compare'
    compare_dir.mkdir(parents=True, exist_ok=True)
    compare_save_path = compare_dir / f'{stem}.png'

    RENDERER.draw_compare_frame(
        points=points,
        gt_boxes=gt_boxes,
        gt_names=[f'{name} (GT)' for name in gt_names],
        pred_boxes=pred_boxes,
        pred_names=pred_names,
        save_path=compare_save_path,
        z_range=z_range,
        gt_title='Ground Truth',
        pred_title='Detections'
    )

    return f'[{stem}]  compare -> {compare_save_path.name}'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualise ground-truth labels as BEV images')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Point cloud directory (txt-label mode) OR NuScenes root (v1.0-*/ folder)')
    parser.add_argument('--label_path', type=str, default=None,
                        help='Directory containing label .txt files (txt-label mode only)')
    parser.add_argument('--nuscenes_info', type=str, default=None,
                        help='Path to nuscenes_infos_*.pkl (enables NuScenes mode; ignores --label_path/--ext globbing)')
    parser.add_argument('--ext', type=str, default=None,
                        help='Point cloud file extension for txt-label mode (.npy or .bin). Default: .npy')
    parser.add_argument('--output_dir', type=str,
                        default='/OpenPCDet/output/gt_images',
                        help='Where to save the BEV images')
    parser.add_argument('--result_pkl', type=str, default=None,
                        help='Optional OpenPCDet result.pkl with detections for GT-vs-detection comparison')
    parser.add_argument('--single_image', action='store_true',
                        help='Only render the hardcoded KITTI or NuScenes frame defined in the script')
    parser.add_argument('--z_min', type=float, default=None,
                        help='Min height (Z) of points to plot')
    parser.add_argument('--z_max', type=float, default=None,
                        help='Max height (Z) of points to plot')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    label_path = Path(args.label_path) if args.label_path is not None else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z_range = None
    if args.z_min is not None or args.z_max is not None:
        z_range = (args.z_min if args.z_min is not None else -float('inf'),
                   args.z_max if args.z_max is not None else  float('inf'))

    if args.nuscenes_info:
        info_path = Path(args.nuscenes_info)
        if not info_path.exists():
            raise FileNotFoundError(f'NuScenes info file not found: {info_path}')

        samples = _load_nuscenes_samples(nuscenes_root=data_path, info_path=info_path)
        print(f'Loaded {len(samples)} NuScenes samples from {info_path}')

        if args.single_image:
            samples = [sample for sample in samples if sample['stem'] == NUSCENES_SINGLE_IMAGE_STEM]
            if not samples:
                raise FileNotFoundError(
                    f'No NuScenes sample found for NUSCENES_SINGLE_IMAGE_STEM={NUSCENES_SINGLE_IMAGE_STEM}'
                )
            print(f'Single-image mode enabled, rendering only {samples[0]["stem"]}.png')

        num_workers = args.workers if args.workers else min(cpu_count(), len(samples), 16)
        worker_fn = partial(_process_one_nuscenes,
                            output_dir=str(output_dir),
                            z_range=z_range)

        done = 0
        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(worker_fn, samples):
                if result is not None:
                    done += 1
                    print(result)

        print(f'\nDone. {done} GT images saved to {output_dir}  ({num_workers} workers)')
        return

    det_by_frame = None
    if args.result_pkl is not None:
        result_path = Path(args.result_pkl)
        if not result_path.exists():
            raise FileNotFoundError(f'Detection result file not found: {result_path}')
        det_by_frame = _load_detection_results(result_path)
        print(f'Loaded {len(det_by_frame)} detection frames from {result_path}')

    # Default: txt-label mode.
    if label_path is None:
        raise ValueError('--label_path is required unless --nuscenes_info is provided')
    ext = args.ext if args.ext is not None else '.npy'

    pc_files = sorted(glob.glob(str(data_path / f'*{ext}')))
    print(f'Found {len(pc_files)} point cloud files in {data_path}')

    if args.single_image:
        pc_files = [pc_file for pc_file in pc_files if Path(pc_file).stem == KITTI_SINGLE_IMAGE_ID]
        if not pc_files:
            raise FileNotFoundError(
                f'No point cloud found for KITTI_SINGLE_IMAGE_ID={KITTI_SINGLE_IMAGE_ID} in {data_path}'
            )
        print(f'Single-image mode enabled, rendering only {Path(pc_files[0]).name}')

    num_workers = args.workers if args.workers else min(cpu_count(), len(pc_files), 16)
    if det_by_frame is None:
        worker_fn = partial(_process_one,
                            label_dir=str(label_path),
                            output_dir=str(output_dir),
                            ext=ext,
                            z_range=z_range)
    else:
        worker_fn = partial(_process_one_compare,
                            label_dir=str(label_path),
                            output_dir=str(output_dir),
                            ext=ext,
                            z_range=z_range,
                            det_by_frame=det_by_frame)

    done = 0
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(worker_fn, pc_files):
            if result is not None:
                done += 1
                print(result)

    if det_by_frame is None:
        print(f'\nDone. {done} GT images saved to {output_dir}  ({num_workers} workers)')
    else:
        print(f'\nDone. {done} comparison images saved to {output_dir}  ({num_workers} workers)')


if __name__ == '__main__':
    main()
