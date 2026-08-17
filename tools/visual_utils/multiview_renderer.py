from pathlib import Path
import re
import pickle

import numpy as np
import matplotlib.pyplot as plt


class MultiViewRenderer:
    def __init__(self):
        self.class_colors = {
            'Car': 'green',
            'Vehicle': 'green',
            'Pedestrian': 'blue',
            'Cyclist': 'goldenrod',
            'car': 'green',
            'pedestrian': 'blue',
            'bicycle': 'goldenrod',
        }
        self.default_color = 'magenta'

    def _draw_frame_axes(self, ax_bev, ax_front, points, boxes, names, point_range, z_range,
                         bev_title, front_title):
        mask = ((points[:, 0] > point_range[0]) & (points[:, 0] < point_range[2]) &
                (points[:, 1] > point_range[1]) & (points[:, 1] < point_range[3]))
        if z_range is not None:
            mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        points = points[mask]

        if z_range is None:
            z_min_plot, z_max_plot = -30.0, 30.0
        else:
            z_min_plot, z_max_plot = z_range

        ax_bev.scatter(points[:, 1], points[:, 0], s=0.1, c='black', alpha=1.0)
        #ax_front.scatter(points[:, 0], points[:, 2], s=0.1, c='white', alpha=0.5)

        for i, box in enumerate(boxes):
            x, y, z, dx, dy, dz, heading = box[:7]
            name = names[i] if i < len(names) else 'Unknown'
            class_name = str(name).split()[0]
            color = self.class_colors.get(class_name, self.default_color)

            corners = self.get_box_corners_2d(x, y, dx, dy, heading)
            polygon = plt.Polygon(corners[:, [1, 0]], fill=False, edgecolor=color, linewidth=1.5)
            #polygon = plt.Polygon(corners, fill=False, edgecolor=color, linewidth=1.5)

            ax_bev.add_patch(polygon)
            ax_bev.text(y, x, name, color=color, fontsize=5, ha='center', va='bottom')

            half_x_extent = 0.5 * (abs(np.cos(heading)) * dx + abs(np.sin(heading)) * dy)
            x0 = x - half_x_extent
            z0 = z - dz / 2.0
            rect = plt.Rectangle((x0, z0), 2.0 * half_x_extent, dz, fill=False, edgecolor=color, linewidth=1.5)
            #ax_front.add_patch(rect)
            #ax_front.text(x, z + dz / 2.0, name, color=color, fontsize=5, ha='center', va='bottom')

        ax_bev.set_xlim(point_range[1], point_range[3])
        ax_bev.set_ylim(point_range[0], point_range[2])
        ax_bev.set_facecolor('white')
        ax_bev.set_aspect('equal')
        ax_bev.set_xlabel('Y (m)')
        ax_bev.set_ylabel('X (m)')
        ax_bev.set_title(bev_title)
        ax_bev.set_title("")
        # Keep the border (spines)
        for spine in ax_bev.spines.values():
            spine.set_visible(True)

        # Remove ticks and numbers
        ax_bev.set_xticks([])
        ax_bev.set_yticks([])

        # Remove axis labels
        ax_bev.set_xlabel("")
        ax_bev.set_ylabel("")

        # Remove title
        ax_bev.set_title("")
        #ax_front.set_xlim(point_range[0], point_range[2])
        #ax_front.set_ylim(z_min_plot, z_max_plot)
        #ax_front.set_facecolor('black')
        #ax_front.set_aspect('auto')
        #ax_front.set_xlabel('X (m)')
        #ax_front.set_ylabel('Z (m)')
        #ax_front.set_title(front_title)

    @staticmethod
    def get_box_corners_2d(cx, cy, dx, dy, heading):
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        half_dx = dx / 2
        half_dy = dy / 2

        corners = np.array([
            [-half_dx, -half_dy],
            [half_dx, -half_dy],
            [half_dx, half_dy],
            [-half_dx, half_dy],
        ])

        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        corners = corners @ rot.T
        corners[:, 0] += cx
        corners[:, 1] += cy
        return corners

    def draw_frame(self, points, boxes, names, save_path, point_range=(-30, -50, 50, 50),
                   z_range=(-3,3), bev_title='BEV (X-Y)', front_title='Front View (X-Z)'):
        fig, ax_bev = plt.subplots(1, 1, figsize=(9, 9), dpi=300)
        ax_front= None
        print(save_path)
        if "0422" in str(save_path):
            fig, ax_bev = plt.subplots(1, 1, figsize=(8,7), dpi=300)
            print("KITTI")
            point_range=( 0, -40, 70, 40)
        elif "n008" in str(save_path):
            fig, ax_bev = plt.subplots(1, 1, figsize=(13, 8), dpi=300)
            print("nuScenes")
            point_range=( -30, -41.2, 41.2, 51.2)
            point_range=( -40, -62, 36, 70)
            z_range=(-3,1)


            """
            samples = _load_nuscenes_samples(nuscenes_root="/OpenPCDet/datasets/nuscenes/v1.0-mini", info_path="/OpenPCDet/datasets/nuscenes/v1.0-mini/nuscenes_infos_1sweeps_val.pkl")
            for sample in samples:
                print(sample["stem"]) 
            samplesf = [sample for sample in samples if "1533151616947490" in sample['stem']]
            print(samplesf)
            #pc_file =samples['pc_file']
            pc_file = samplesf[0]['pc_file']
            raw = np.fromfile(str(pc_file), dtype=np.float32)
            if raw.size % 5 == 0:
                points = raw.reshape(-1, 5)
            elif raw.size % 4 == 0:
                points = raw.reshape(-1, 4)
            """
            pc_file = "/OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151616997246.pcd.bin"

            points = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 5)
            print(points.shape)


        elif "0326" in str(save_path):
            fig, ax_bev = plt.subplots(1, 1, figsize=(7.4, 6.7), dpi=300)
            print("eRod")
            z_range=(-3,3)
            point_range=( -32, -38, 35, 36)

        self._draw_frame_axes(ax_bev, ax_front, points, boxes, names, point_range, z_range, bev_title, front_title)

        fig.tight_layout()
        fig.savefig(str(save_path), facecolor='white')
        plt.close(fig)

    def draw_compare_frame(self, points, gt_boxes, gt_names, pred_boxes, pred_names, save_path,
                           point_range=(-50, -50, 50, 50), z_range=(-3,3),
                           gt_title='Ground Truth', pred_title='Detections'):
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)
        self._draw_frame_axes(
            axes[0, 0], axes[1, 0], points, gt_boxes, gt_names, point_range, z_range,
            bev_title=f'BEV (X-Y) — {gt_title}', front_title=f'Front View (X-Z) — {gt_title}'
        )
        self._draw_frame_axes(
            axes[0, 1], axes[1, 1], points, pred_boxes, pred_names, point_range, z_range,
            bev_title=f'BEV (X-Y) — {pred_title}', front_title=f'Front View (X-Z) — {pred_title}'
        )
        fig.tight_layout()
        fig.savefig(str(save_path), facecolor='black')
        plt.close(fig)

    @staticmethod
    def _sanitize_name(name):
        return re.sub(r'[^A-Za-z0-9_-]+', '-', str(name))

    @staticmethod
    def extract_points_in_box(points, box, xy_margin=0.4, z_margin=0.2):
        x, y, z, dx, dy, dz, heading = box[:7]
        shifted = points[:, :3] - np.array([x, y, z], dtype=np.float32)

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        local_x = cos_h * shifted[:, 0] + sin_h * shifted[:, 1]
        local_y = -sin_h * shifted[:, 0] + cos_h * shifted[:, 1]
        local_z = shifted[:, 2]

        mask = (
            (np.abs(local_x) <= dx / 2.0 + xy_margin) &
            (np.abs(local_y) <= dy / 2.0 + xy_margin) &
            (np.abs(local_z) <= dz / 2.0 + z_margin)
        )
        return np.stack([local_x[mask], local_y[mask], local_z[mask]], axis=1)

    def draw_instance_views(self, local_points, box, class_name, save_path):
        _, _, _, dx, dy, dz, _ = box[:7]

        pad_xy = 0.6
        pad_z = 0.4
        x_lim = (-dx / 2.0 - pad_xy, dx / 2.0 + pad_xy)
        y_lim = (-dy / 2.0 - pad_xy, dy / 2.0 + pad_xy)
        z_lim = (-dz / 2.0 - pad_z, dz / 2.0 + pad_z)

        fig, (ax_front, ax_side, ax_top) = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
        for ax in (ax_front, ax_side, ax_top):
            ax.set_facecolor('black')

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

    def render_instances(self, points, boxes, names, frame_key, instances_dir):
        instances_dir = Path(instances_dir)
        instances_dir.mkdir(parents=True, exist_ok=True)

        for idx, box in enumerate(boxes):
            class_name = names[idx] if idx < len(names) else 'Unknown'
            safe_class_name = self._sanitize_name(class_name)
            instance_id = f'{idx:03d}'
            local_points = self.extract_points_in_box(points, box)

            file_name = f'{frame_key}_{instance_id}_{safe_class_name}.png'
            self.draw_instance_views(local_points, box, class_name, instances_dir / file_name)



def _load_nuscenes_samples(nuscenes_root: Path, info_path: Path):
    """Return a lightweight list of {pc_file, gt_boxes, gt_names, ...}.

    Expects `info_path` to be an OpenPCDet-generated `nuscenes_infos_*.pkl`.
    Each entry contains:
      - lidar_path (relative)
      - token
      - gt_boxes (N, 9) or (N, 7+) with x,y,z,dx,dy,dz,yaw in the first 7
      - gt_names
    """
    nuscenes_root = Path(nuscenes_root)
    info_path = Path(info_path)
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