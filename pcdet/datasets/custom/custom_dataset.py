import copy
import pickle
import os
from pathlib import Path

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.sample_id_list = self._load_split_ids(self.split)

        self.custom_infos = []
        self._infos_by_lidar_idx = {}
        self._active_sample_id_list = None
        self.include_data(self.mode)
        self._build_infos_index()
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    def _load_split_ids(self, split_value):
        """Load split ids from either ImageSets/<split>.txt or an absolute .txt path."""
        if split_value is None:
            return None

        split_str = str(split_value)
        # Absolute or explicit file path
        split_path = Path(split_str)
        if split_str.endswith('.txt') and split_path.exists():
            return [x.strip() for x in split_path.read_text().splitlines() if x.strip()]

        # Named split under ImageSets
        split_file = Path(self.root_path) / 'ImageSets' / f'{split_str}.txt'
        if split_file.exists():
            return [x.strip() for x in split_file.read_text().splitlines() if x.strip()]

        return None

    def _build_infos_index(self):
        self._infos_by_lidar_idx = {}
        for info in self.custom_infos:
            lidar_idx = info.get('point_cloud', {}).get('lidar_idx', None)
            if lidar_idx is None:
                continue
            self._infos_by_lidar_idx[str(lidar_idx)] = info

        if self.sample_id_list is not None:
            missing_ids = [sid for sid in self.sample_id_list if sid not in self._infos_by_lidar_idx]
            if missing_ids:
                raise ValueError(
                    f"Split file contains {len(self.sample_id_list)} ids, but {len(missing_ids)} ids are missing from infos. "
                    f"Example missing ids: {missing_ids[:20]} (split={self.split})"
                )

            # Keep the full split ordering (do not silently drop ids).
            self._active_sample_id_list = list(self.sample_id_list)
        else:
            self._active_sample_id_list = None

    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)

        self.custom_infos = list(custom_infos)

        swap_lw = bool(self.dataset_cfg.get('SWAP_LW', False))
        heading_in_degrees_cfg = self.dataset_cfg.get('HEADING_IN_DEGREES', None)

        # Auto-detect degrees if not explicitly configured.
        heading_in_degrees_auto = False
        if heading_in_degrees_cfg is None:
            for info in self.custom_infos:
                annos = info.get('annos', None)
                if not annos or 'gt_boxes_lidar' not in annos:
                    continue
                gt = np.asarray(annos['gt_boxes_lidar'])
                if gt.size == 0 or gt.shape[1] < 7:
                    continue
                max_abs = float(np.max(np.abs(gt[:, 6])))
                if max_abs > (2 * np.pi + 1.0):
                    heading_in_degrees_auto = True
                    break

        heading_in_degrees = bool(heading_in_degrees_cfg) if heading_in_degrees_cfg is not None else heading_in_degrees_auto

        if swap_lw or heading_in_degrees:
            for info in self.custom_infos:
                annos = info.get('annos', None)
                if annos is None or 'gt_boxes_lidar' not in annos:
                    continue
                gt_boxes = np.asarray(annos['gt_boxes_lidar'])
                if gt_boxes.size == 0:
                    continue
                if swap_lw and gt_boxes.shape[1] >= 5:
                    gt_boxes[:, [3, 4]] = gt_boxes[:, [4, 3]]
                if heading_in_degrees and gt_boxes.shape[1] >= 7:
                    gt_boxes[:, 6] = np.deg2rad(gt_boxes[:, 6])
                annos['gt_boxes_lidar'] = gt_boxes

        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))

    def get_label(self, idx):
        label_file = self.root_path / 'labels' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'points' / ('%s.npy' % idx)
        assert lidar_file.exists()
        points = np.load(lidar_file)

        # Normalize intensity to KITTI-style [0, 1].
        # Common custom encodings: 0-127 or 0-255.
        if points.ndim == 2 and points.shape[1] >= 4:
            intensity_max = float(points[:, 3].max())
            if intensity_max > 1.0:
                if intensity_max <= 127.5:
                    points[:, 3] = points[:, 3] / 127.0
                elif intensity_max <= 255.0:
                    points[:, 3] = points[:, 3] / 255.0
                else:
                    points[:, 3] = points[:, 3] / max(intensity_max, 1e-6)

        return points

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        self.sample_id_list = self._load_split_ids(self.split)
        self._build_infos_index()

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            base_len = len(self._active_sample_id_list) if self._active_sample_id_list is not None else len(self.custom_infos)
            return base_len * self.total_epochs

        if self._active_sample_id_list is not None:
            return len(self._active_sample_id_list)

        return len(self.custom_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self)

        if self._active_sample_id_list is not None:
            sample_idx = self._active_sample_id_list[index]
            src_info = self._infos_by_lidar_idx.get(sample_idx, None)
            if src_info is None:
                raise KeyError(f"Missing info for sample_id={sample_idx}. Regenerate infos or fix ImageSets split.")
            info = copy.deepcopy(src_info)
            frame_id = sample_idx
        else:
            info = copy.deepcopy(self.custom_infos[index])
            sample_idx = str(info['point_cloud']['lidar_idx'])
            frame_id = sample_idx

        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': frame_id,
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if self._active_sample_id_list is not None and len(self._active_sample_id_list) > 0:
            first_info = self._infos_by_lidar_idx[self._active_sample_id_list[0]]
        else:
            first_info = self.custom_infos[0]

        if 'annos' not in first_info.keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)

        if self._active_sample_id_list is not None:
            eval_gt_annos = [
                copy.deepcopy(self._infos_by_lidar_idx[sid]['annos'])
                for sid in self._active_sample_id_list
            ]
        else:
            eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                if self.dataset_cfg.get('SWAP_LW', False) and gt_boxes_lidar.shape[1] >= 5:
                    gt_boxes_lidar[:, [3, 4]] = gt_boxes_lidar[:, [4, 3]]
                if self.dataset_cfg.get('HEADING_IN_DEGREES', False) and gt_boxes_lidar.shape[1] >= 7:
                    gt_boxes_lidar[:, 6] = np.deg2rad(gt_boxes_lidar[:, 6])
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    custom_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info train file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        data_path = Path(dataset_cfg.DATA_PATH)
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=data_path,
            save_path=data_path,
        )
