#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from pcdet.utils import box_utils


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _percentile(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p))


def _as_str_array(x) -> np.ndarray:
    # Handles numpy arrays of dtype <U, object arrays, python lists, etc.
    return np.asarray(x).astype(str)


def _load_pkl(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Diagnose systematic geometry mismatches between detections (dt) and ground-truth (gt) boxes'
    )
    parser.add_argument('--result_pkl', required=True, help='Path to eval result.pkl (list of per-frame det annos)')
    parser.add_argument('--infos_pkl', required=True, help='Path to custom_infos_val.pkl (list of infos with annos)')
    parser.add_argument('--class_name', default='Car', help='Class name to analyze (default: Car)')
    parser.add_argument('--min_bev_iou', type=float, default=0.10, help='Min aligned BEV IoU for matching (default: 0.10)')
    args = parser.parse_args()

    result_pkl = Path(args.result_pkl)
    infos_pkl = Path(args.infos_pkl)

    det = _load_pkl(result_pkl)
    infos = _load_pkl(infos_pkl)

    if not isinstance(det, list) or len(det) == 0:
        raise ValueError(f'Unexpected result_pkl content: type={type(det)} len={len(det) if isinstance(det, list) else "?"}')
    if not isinstance(infos, list) or len(infos) == 0:
        raise ValueError(f'Unexpected infos_pkl content: type={type(infos)} len={len(infos) if isinstance(infos, list) else "?"}')

    gt_by_id = {str(info['point_cloud']['lidar_idx']): info for info in infos}

    angles = []
    z_diffs = []
    size_diffs = []
    xy_dists = []
    swap_better = []

    annos_applied_example = None
    for info in infos:
        pc = info.get('point_cloud', {})
        if isinstance(pc, dict) and pc.get('annos_applied') is not None:
            annos_applied_example = pc.get('annos_applied')
            break

    for d in det:
        fid = str(d.get('frame_id'))
        gt_info = gt_by_id.get(fid)
        if gt_info is None:
            continue
        gt_ann = gt_info.get('annos', None)
        if gt_ann is None:
            continue

        gt_names = _as_str_array(gt_ann.get('name', []))
        gt_boxes = np.asarray(gt_ann.get('gt_boxes_lidar', np.zeros((0, 7))), dtype=np.float32)

        dt_names = _as_str_array(d.get('name', []))
        dt_boxes = np.asarray(d.get('boxes_lidar', np.zeros((0, 7))), dtype=np.float32)

        if gt_boxes.ndim != 2 or gt_boxes.shape[1] < 7 or dt_boxes.ndim != 2 or dt_boxes.shape[1] < 7:
            continue

        gt_mask = gt_names == args.class_name
        dt_mask = dt_names == args.class_name

        gt_c = gt_boxes[gt_mask]
        dt_c = dt_boxes[dt_mask]
        if gt_c.shape[0] == 0 or dt_c.shape[0] == 0:
            continue

        iou = box_utils.boxes3d_nearest_bev_iou(torch.from_numpy(gt_c), torch.from_numpy(dt_c))
        best_iou, best_j = iou.max(dim=1)
        keep = best_iou.numpy() > args.min_bev_iou
        if not np.any(keep):
            continue

        gj = np.where(keep)[0]
        dj = best_j.numpy()[keep]

        g = gt_c[gj]
        p = dt_c[dj]

        ang = _wrap_to_pi(p[:, 6] - g[:, 6])
        angles.append(ang)
        z_diffs.append(p[:, 2] - g[:, 2])
        size_diffs.append(p[:, 3:6] - g[:, 3:6])
        xy_dists.append(np.linalg.norm(p[:, 0:2] - g[:, 0:2], axis=1))

        d_swap = np.abs(p[:, 3] - g[:, 4]) + np.abs(p[:, 4] - g[:, 3])
        d_noswap = np.abs(p[:, 3] - g[:, 3]) + np.abs(p[:, 4] - g[:, 4])
        swap_better.append((d_swap < d_noswap).astype(np.float32))

    if not angles:
        print(f'No matched pairs found for class={args.class_name} with min_bev_iou>{args.min_bev_iou}.')
        return

    ang = np.concatenate(angles)
    zd = np.concatenate(z_diffs)
    sd = np.concatenate(size_diffs)
    xy = np.concatenate(xy_dists)
    sb = np.concatenate(swap_better)

    print('Inputs')
    print(f'  result_pkl: {result_pkl}')
    print(f'  infos_pkl:  {infos_pkl}')
    print(f'  class_name: {args.class_name}')
    print(f'  min_bev_iou: {args.min_bev_iou}')
    if annos_applied_example is not None:
        print('  annos_applied example (from infos):')
        print(f'    {annos_applied_example}')

    print('\nMatched pairs')
    print(f'  N: {ang.shape[0]}')

    print('\nYaw / heading')
    print(f'  yaw_diff median (deg): {float(np.median(ang) * 180.0 / np.pi):.2f}')
    print(
        f'  |yaw_diff| median/p90 (deg): '
        f'{float(np.median(np.abs(ang)) * 180.0 / np.pi):.2f} / {float(_percentile(np.abs(ang), 90) * 180.0 / np.pi):.2f}'
    )
    for target_deg in [0.0, 90.0, -90.0, 180.0, -180.0]:
        target = np.deg2rad(target_deg)
        close = np.abs(_wrap_to_pi(ang - target)) < np.deg2rad(15)
        print(f'  within 15deg of {target_deg:>6.1f}: {int(close.sum())}/{ang.shape[0]}')

    print('\nCenter deltas')
    print(f'  XY dist median/p90 (m): {float(np.median(xy)):.3f} / {float(_percentile(xy, 90)):.3f}')
    print(f'  Z diff median p10/p90 (m): {float(np.median(zd)):.3f}  {float(_percentile(zd, 10)):.3f}/{float(_percentile(zd, 90)):.3f}')

    print('\nSize deltas (dt - gt)')
    print(f'  dx,dy,dz median (m): {[float(np.median(sd[:, i])) for i in range(3)]}')
    print(f'  dx p10/p90: {float(_percentile(sd[:, 0], 10)):.3f}/{float(_percentile(sd[:, 0], 90)):.3f}')
    print(f'  dy p10/p90: {float(_percentile(sd[:, 1], 10)):.3f}/{float(_percentile(sd[:, 1], 90)):.3f}')
    print(f'  dz p10/p90: {float(_percentile(sd[:, 2], 10)):.3f}/{float(_percentile(sd[:, 2], 90)):.3f}')

    print('\nSwap check')
    print(f'  fraction where dt dims closer if swapped (dx<->dy): {float(sb.mean()):.3f}')


if __name__ == '__main__':
    main()
