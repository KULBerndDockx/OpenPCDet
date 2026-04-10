import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ProcessPoolExecutor

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils.multiview_renderer import MultiViewRenderer


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
            # Normalize intensity to [0, 1] range (KITTI convention)
            if points.shape[1] >= 4 and points[:, 3].max() > 1.0:
                points[:, 3] = points[:, 3] / 255.0
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def draw_bev_image(points, pred_boxes, pred_scores, pred_labels, class_names, save_path,
                   score_thresh=0.3, point_range=(-50, -50, 50, 50), show_realtime=False,
                   fig_ax=None):
    """
    Draw bird's-eye view of point cloud. If `show_realtime` is False, saves image to
    `save_path` (like before). If `show_realtime` is True, updates or creates a
    matplotlib figure for realtime display using `fig_ax` (tuple of fig, ax).
    """

    # Filter points within range
    mask = (points[:, 0] > point_range[0]) & (points[:, 0] < point_range[2]) & \
           (points[:, 1] > point_range[1]) & (points[:, 1] < point_range[3])
    points = points[mask]

    # Prepare figure/axis (reuse if provided)
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        created_fig = True
    else:
        fig, ax = fig_ax
        ax.clear()
        created_fig = False

    # Plot points (BEV: x=forward, y=left)
    ax.scatter(points[:, 1], points[:, 0], s=0.1, c='white', alpha=0.5)

    # Colors for different classes
    colors = ['lime', 'cyan', 'yellow', 'red', 'magenta', 'orange']

    # Draw boxes
    if pred_boxes is not None:
        for i in range(len(pred_boxes)):
            if pred_scores[i] < score_thresh:
                continue
            box = pred_boxes[i]
            x, y, z, dx, dy, dz, heading = box
            label = int(pred_labels[i])
            color = colors[(label - 1) % len(colors)]

            # Create rotated rectangle (BEV: swap x/y for plotting)
            corners = get_box_corners_2d(x, y, dx, dy, heading)
            polygon = plt.Polygon(corners[:, [1, 0]], fill=False, edgecolor=color, linewidth=1.5)
            ax.add_patch(polygon)

            # Add label text
            label_name = class_names[label - 1] if label <= len(class_names) else str(label)
            ax.text(y, x, f'{label_name}\n{pred_scores[i]:.2f}',
                    color=color, fontsize=5, ha='center', va='bottom')

    ax.set_xlim(point_range[1], point_range[3])
    ax.set_ylim(point_range[0], point_range[2])
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    ax.set_title('Bird\'s Eye View')
    fig.tight_layout()

    if show_realtime:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        return (fig, ax)
    else:
        if save_path is not None:
            fig.savefig(save_path, facecolor='black')
        if created_fig:
            plt.close(fig)
        return None


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


def _render_prediction_outputs(points, boxes, scores, labels, class_names, sample_name, output_dir, score_thresh):
    """Render frame and per-instance prediction images in a worker process."""
    renderer = MultiViewRenderer()

    keep_mask = scores >= score_thresh
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]

    pred_names = []
    pred_class_names = []
    for i in range(len(boxes)):
        label = int(labels[i])
        cls_name = class_names[label - 1] if 1 <= label <= len(class_names) else str(label)
        pred_class_names.append(cls_name)
        pred_names.append(f'{cls_name} {scores[i]:.2f}')

    frame_save_path = Path(output_dir) / f'{sample_name}.png'
    renderer.draw_frame(
        points=points,
        boxes=boxes,
        names=pred_names,
        save_path=frame_save_path,
        bev_title='BEV (X-Y)  -  Predictions',
        front_title='Front View (X-Z)  -  Predictions'
    )

    instances_dir = Path(output_dir) / 'instances_pred'
    renderer.render_instances(
        points=points,
        boxes=boxes,
        names=pred_class_names,
        frame_key=sample_name,
        instances_dir=instances_dir
    )

    return sample_name


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--realtime', action='store_true', default=False,
                        help='show BEV images in realtime instead of saving to png')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of parallel workers for data loading and image rendering')
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='override MODEL.POST_PROCESSING.SCORE_THRESH at runtime')
    parser.add_argument('--nms_thresh', type=float, default=None,
                        help='override MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH at runtime')
    parser.add_argument('--nms_post_maxsize', type=int, default=None,
                        help='override MODEL.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE at runtime')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    if args.score_thresh is not None:
        cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_thresh
    if args.nms_thresh is not None:
        cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = args.nms_thresh
    if args.nms_post_maxsize is not None:
        cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE = args.nms_post_maxsize

    render_score_thresh = float(cfg.MODEL.POST_PROCESSING.SCORE_THRESH)

    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    output_dir = Path('/OpenPCDet/output/demo_images')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use DataLoader for parallel point cloud loading
    dataloader = torch.utils.data.DataLoader(
        demo_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=demo_dataset.collate_batch,
        pin_memory=True
    )

    # Setup realtime display if requested
    fig_ax = None
    if args.realtime:
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        fig_ax = (fig, ax)

    render_pool = None if args.realtime else ProcessPoolExecutor(max_workers=args.workers)
    futures = []

    with torch.no_grad():
        for idx, data_dict in enumerate(dataloader):
            logger.info(f'Processing sample index: \t{idx + 1}')
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            points = data_dict['points'][:, 1:].cpu().numpy()
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            sample_name = Path(demo_dataset.sample_file_list[idx]).stem
            save_path = output_dir / f'{sample_name}.png'

            if args.realtime:
                fig_ax = draw_bev_image(points, pred_boxes, pred_scores, pred_labels,
                                       cfg.CLASS_NAMES, None, score_thresh=render_score_thresh,
                                       show_realtime=True, fig_ax=fig_ax)
                logger.info(f'  Displaying BEV image realtime -> {sample_name}')
            else:
                # Render images in parallel background processes
                fut = render_pool.submit(
                    _render_prediction_outputs,
                    points,
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    list(cfg.CLASS_NAMES),
                    sample_name,
                    str(output_dir),
                    render_score_thresh
                )
                futures.append((sample_name, fut))

    # Wait for all renders to finish
    if render_pool is not None:
        for sample_name, fut in futures:
            fut.result()  # raises if the worker hit an error
            logger.info(f'  Saved frame + instances -> {sample_name}')
        render_pool.shutdown(wait=True)

    if args.realtime:
        plt.ioff()
        plt.show()

    logger.info(f'Demo done. Images saved to {output_dir}')


if __name__ == '__main__':
    main()
