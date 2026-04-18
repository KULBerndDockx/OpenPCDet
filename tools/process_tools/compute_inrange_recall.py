#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def _parse_pc_range(vals: List[float]) -> np.ndarray:
    if len(vals) != 6:
        raise ValueError(f"POINT_CLOUD_RANGE must have 6 floats, got {len(vals)}")
    return np.asarray(vals, dtype=np.float32)


def _inside_range_by_center(gt_boxes: np.ndarray, pc_range: np.ndarray) -> np.ndarray:
    """gt_boxes: (N, 7) in lidar coords."""
    if gt_boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    centers = gt_boxes[:, 0:3]
    inside = (
        (centers[:, 0] >= pc_range[0])
        & (centers[:, 0] <= pc_range[3])
        & (centers[:, 1] >= pc_range[1])
        & (centers[:, 1] <= pc_range[4])
        & (centers[:, 2] >= pc_range[2])
        & (centers[:, 2] <= pc_range[5])
    )
    return inside


def _load_infos(infos_pkl: Path) -> Dict[str, dict]:
    infos = pickle.load(infos_pkl.open("rb"))
    by_id: Dict[str, dict] = {}
    for info in infos:
        sid = info.get("point_cloud", {}).get("lidar_idx", None)
        if sid is not None:
            by_id[str(sid)] = info
    return by_id


def _load_results(result_pkl: Path) -> Dict[str, dict]:
    results = pickle.load(result_pkl.open("rb"))
    by_frame: Dict[str, dict] = {}
    for r in results:
        fid = str(r.get("frame_id"))
        by_frame[fid] = r
    return by_frame


def _boxes_iou3d_gpu(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Returns IoU matrix (num_pred, num_gt) on GPU."""
    from pcdet.ops.iou3d_nms import iou3d_nms_utils

    return iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes)


def _recall_from_max_iou(max_iou: torch.Tensor, thresh_list: List[float]) -> List[int]:
    return [int((max_iou >= thr).sum().item()) for thr in thresh_list]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute recall using only GT boxes whose centers are inside POINT_CLOUD_RANGE. "
            "Uses OpenPCDet 3D IoU (CUDA) via boxes_iou3d_gpu."),
    )
    parser.add_argument(
        "--infos_pkl",
        type=Path,
        default=Path("/OpenPCDet/erod/custom_infos_val.pkl"),
    )
    parser.add_argument(
        "--split_txt",
        type=Path,
        default=Path("/OpenPCDet/erod/ImageSets/val_small.txt"),
    )
    parser.add_argument(
        "--result_pkl",
        type=Path,
        default=Path(
            "/OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/pointpillar_erod/default/eval/epoch_7728/val_small/default/result.pkl"
        ),
    )
    parser.add_argument(
        "--pc_range",
        type=float,
        nargs=6,
        default=[0, -39.68, -3, 69.12, 39.68, 1],
        help="x_min y_min z_min x_max y_max z_max",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="IoU thresholds to report recall at",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device (must be CUDA for boxes_iou3d_gpu)",
    )
    args = parser.parse_args()

    pc_range = _parse_pc_range(args.pc_range)
    thresh_list = list(args.thresh)

    split_ids = [x.strip() for x in args.split_txt.read_text().splitlines() if x.strip()]
    infos_by_id = _load_infos(args.infos_pkl)
    results_by_frame = _load_results(args.result_pkl)

    missing_infos = [sid for sid in split_ids if sid not in infos_by_id]
    missing_results = [sid for sid in split_ids if sid not in results_by_frame]
    if missing_infos:
        print(f"WARNING: {len(missing_infos)} split ids missing from infos: {missing_infos[:10]}")
    if missing_results:
        print(f"WARNING: {len(missing_results)} split ids missing from results: {missing_results[:10]}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This script currently requires CUDA because it uses boxes_iou3d_gpu")

    # Totals
    total_gt_all = 0
    total_gt_in = 0
    matched_in_allpred = [0 for _ in thresh_list]

    class_names = ["Car", "Pedestrian", "Cyclist"]
    per_cls_total_in = {c: 0 for c in class_names}
    per_cls_matched_in = {c: [0 for _ in thresh_list] for c in class_names}

    # Iterate frames
    for sid in split_ids:
        info = infos_by_id.get(sid, None)
        if info is None:
            continue

        ann = info.get("annos", None)
        if not ann:
            continue

        gt_names = ann.get("name", None)
        gt_boxes = np.asarray(ann.get("gt_boxes_lidar", []), dtype=np.float32)
        if gt_boxes.size == 0:
            continue

        total_gt_all += gt_boxes.shape[0]
        in_mask = _inside_range_by_center(gt_boxes, pc_range)
        if not in_mask.any():
            continue

        gt_boxes_in = gt_boxes[in_mask]
        gt_names_in = np.asarray(gt_names, dtype=str)[in_mask] if gt_names is not None else np.asarray(["UNKNOWN"] * gt_boxes_in.shape[0])

        total_gt_in += gt_boxes_in.shape[0]

        res = results_by_frame.get(sid, None)
        if res is None:
            # No predictions at all
            continue

        pred_boxes = np.asarray(res.get("boxes_lidar", []), dtype=np.float32)
        pred_names = np.asarray(res.get("name", []), dtype=str)

        # Overall in-range recall using all predictions (matches OpenPCDet recall style more closely)
        if pred_boxes.size == 0:
            max_iou_all = torch.zeros((gt_boxes_in.shape[0],), device=device, dtype=torch.float32)
        else:
            pred_t = torch.from_numpy(pred_boxes).to(device=device)
            gt_t = torch.from_numpy(gt_boxes_in).to(device=device)
            iou = _boxes_iou3d_gpu(pred_t, gt_t)  # (num_pred, num_gt_in)
            max_iou_all = iou.max(dim=0).values

        matched_counts = _recall_from_max_iou(max_iou_all, thresh_list)
        matched_in_allpred = [a + b for a, b in zip(matched_in_allpred, matched_counts)]

        # Per-class in-range recall using predictions filtered to the same class
        for cls in class_names:
            cls_gt_mask = gt_names_in == cls
            if not np.any(cls_gt_mask):
                continue

            per_cls_total_in[cls] += int(cls_gt_mask.sum())

            cls_gt_boxes = gt_boxes_in[cls_gt_mask]

            if pred_boxes.size == 0:
                cls_max_iou = torch.zeros((cls_gt_boxes.shape[0],), device=device, dtype=torch.float32)
            else:
                cls_pred_mask = pred_names == cls
                if not np.any(cls_pred_mask):
                    cls_max_iou = torch.zeros((cls_gt_boxes.shape[0],), device=device, dtype=torch.float32)
                else:
                    cls_pred_t = torch.from_numpy(pred_boxes[cls_pred_mask]).to(device=device)
                    cls_gt_t = torch.from_numpy(cls_gt_boxes).to(device=device)
                    cls_iou = _boxes_iou3d_gpu(cls_pred_t, cls_gt_t)
                    cls_max_iou = cls_iou.max(dim=0).values

            cls_matched_counts = _recall_from_max_iou(cls_max_iou, thresh_list)
            per_cls_matched_in[cls] = [a + b for a, b in zip(per_cls_matched_in[cls], cls_matched_counts)]

    print("=== In-range Recall (GT centers inside POINT_CLOUD_RANGE) ===")
    print(f"Frames in split: {len(split_ids)}")
    print(f"GT total (all): {total_gt_all}")
    print(f"GT total (in-range): {total_gt_in} ({(100.0 * total_gt_in / max(total_gt_all,1)):.1f}%)")

    for thr, m in zip(thresh_list, matched_in_allpred):
        rec = m / max(total_gt_in, 1)
        print(f"ALL-PREDS recall@{thr:.2f}: {rec:.4f} ({m}/{total_gt_in})")

    print("\n--- Per-class (preds filtered by class) ---")
    for cls in class_names:
        denom = per_cls_total_in[cls]
        if denom == 0:
            print(f"{cls}: no in-range GT")
            continue
        parts = []
        for thr, m in zip(thresh_list, per_cls_matched_in[cls]):
            parts.append(f"rec@{thr:.2f}={m/max(denom,1):.4f} ({m}/{denom})")
        print(f"{cls}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
