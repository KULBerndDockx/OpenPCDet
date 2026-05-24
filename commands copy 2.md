# OpenPCDet Commands Cheat Sheet

## 1. Data Preparation & Utilities

*Commands for generating infos and visualizing ground truth.*

### Info Generation

- **KITTI:** `cd /OpenPCDet && python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml && cd tools`
- **EROD:** `cd /OpenPCDet && python3 -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml && cd tools`
- **nuScenes (Mini):** `python3 -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-mini`
- **nuScenes (Trainval):** `python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/nuscenes_converted.yaml`

python3 -m pcdet.datasets.kitti.kitti_dataset --func create_kitti_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_converted.yaml --version v1.0-trainval

### Visualization

- **EROD Labels:** `python3 visualize_labels.py --data_path /OpenPCDet/erod/points --label_path /OpenPCDet/erod/labels --ext .npy --z_min -1.0 --z_max 3.0`
- **nuScenes Labels:** `cd /OpenPCDet/tools && python3 visualize_labels.py --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --label_path /OpenPCDet/erod/labels --ext .npy   --z_min -1.0 --z_max 3.0`


---

## 2. Training (`train.py`)

*Commands to train or fine-tune models.*

### SECOND

- **nuScenes (Fine-tune):** `python3 train.py --cfg_file cfgs/kitti_models/second_nuscenes.yaml --pretrained_model /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag frozen_backbone_1_9 --freeze_backbone`

- **KITTI (Focal):** `python3 train.py --cfg_file cfgs/kitti_models/second_focal.yaml` done
- **Nuscenes (Focal) (Fine-tune):** `python3 train.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal.yaml --pretrained_model /OpenPCDet/output/custom_models/second/focal/second_focal/default/ckpt/checkpoint_epoch_80.pth`


### PointPillar
- **Nuscenes (Fine-tune):** `python3 train.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes.yaml --pretrained_model /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag frozen_backbone_0 --freeze_backbone`

- **KITTI (Focal):** `python3 train.py --cfg_file cfgs/kitti_models/pointpillar_focal.yaml` 
- **Nuscenes (Focal) (Fine-tune):** `python3 train.py --cfg_file cfgs/kitti_models/pointpillar_nuscenes_focal.yaml --pretrained_model /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/pointpillar_focal/default/ckpt/checkpoint_epoch_40.pth`


---

## 3. Testing / Evaluation (`test.py`)

*Commands for model evaluation (val/test splits).*

### SECOND (DEFAULT)
custom_models/second/default/
- **KITTI (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/default/second.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_KITTI_non_retrained`
- **Nuscenes (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/default/second_nuscenes.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_nuscenes_non_retrained`
- **eRod (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/default/second_erod.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_eRod_non_retrained`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_nuscenes/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_KITTI_retrained`
- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_nuscenes.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_nuscenes/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_nuscenes_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_erod.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_erod/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_eRod_retrained`

### SECOND (FOCAL)

- **KITTI (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_focal.yaml --ckpt /OpenPCDet/output/custom_models/second/focal/second_focal/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_KITTI_focal_non_retrained_1`
- **Nuscenes (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal.yaml --ckpt /OpenPCDet/output/custom_models/second/focal/second_focal/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_nuscenes_focal_non_retrained_1`
- **eRod (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_erod_focal.yaml --ckpt /OpenPCDet/output/custom_models/second/focal/second_focal/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_eRod_focal_non_retrained_1`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_focal.yaml --ckpt output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_KITTI_focal_retrained_1`

python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_KITTI_focal_retrained_1

- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_nuscenes_focal_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_erod_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_eRod_focal_retrained`

### SECOND (TENT)

- **KITTI (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/TENT/second_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_KITTI_TENT_non_retrained`
- **Nuscenes (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/TENT/second_nuscenes_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_nuscenes_TENT_non_retrained`
- **eRod (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_erod_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_eRod_TENT_non_retrained`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/TENT/second_TENT.yaml --ckpt /OpenPCDet/output/kitti_models/second_nuscenes/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_KITTI_TENT_retrained`
- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/TENT/second_nuscenes_TENT.yaml --ckpt /OpenPCDet/output/kitti_models/second_nuscenes/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_nuscenes_TENT_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_erod_TENT.yaml --ckpt /OpenPCDet/output/kitti_models/second_nuscenes/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_eRod_TENT_retrained`

### POINTPILLAR (DEFAULT)
tools/cfgs/custom_models/pointpillar/TENT/
- **KITTI (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_KITTI_default_non_retrained`
- **Nuscenes (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_nuscenes_default_non_retrained`
- **eRod (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_erod.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_eRod_default_non_retrained`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_KITTI_default_retrained`
- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_nuscenes_default_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_erod.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_eRod_default_retrained`


### POINTPILLAR (TENT)
tools/cfgs/custom_models/pointpillar/TENT/
- **KITTI (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_KITTI_TENT_non_retrained`
- **Nuscenes (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_nuscenes_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_nuscenes_TENT_non_retrained`
- **eRod (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_erod_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_eRod_TENT_non_retrained`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_TENT.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_KITTI_TENT_retrained`
- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_nuscenes_TENT.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_nuscenes_TENT_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_erod_TENT.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_eRod_TENT_retrained`




---

## 4. Demo / Inference (`demo.py`)

*Commands for running inference on raw point cloud files and generating predictions in the form of images.*

### KITTI

- **PointPillar:** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --data_path /OpenPCDet/datasets/kitti/testing/velodyne`

### EROD

- **PointPillar:** `cd /OpenPCDet/tools && python3 demo.py --cfg_file cfgs/kitti_models/pointpillar_erod.yaml --ckpt ../pcdet/models/pth/pointpillar_7728.pth --data_path /OpenPCDet/erod/points --ext .npy`
- **SECOND:** `cd /OpenPCDet/tools && python3 demo.py --cfg_file cfgs/kitti_models/second_erod.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_erod/default/ckpt/checkpoint_epoch_80.pth --data_path /OpenPCDet/erod/points --ext .npy`
- **SECOND (FOCAL):** `cd /OpenPCDet/tools && python3 demo.py --cfg_file cfgs/kitti_models/second_erod_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_focal/default/ckpt/checkpoint_epoch_20.pth --data_path /OpenPCDet/erod/points --ext .npy`


### nuScenes

- **SECOND:** `cd /OpenPCDet/tools && python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/default/second_nuscenes.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_nuscenes_non_retrained --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin`


- **SECOND:** `cd /OpenPCDet/tools && python3 demo.py --cfg_file cfgs/kitti_models/second_nuscenes.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_nuscenes/default/ckpt_2/checkpoint_epoch_80.pth --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin`

- **SECOND (retrained):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin--extra_tag SECOND_nuscenes_focal_retrained`



- **SECOND FOCAL (retrained):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin--extra_tag SECOND_nuscenes_focal_retrained`

### SECOND (DEFAULT)
custom_models/second/default/
- **KITTI (non-retrained):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/default/second_nuscenes.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_nuscenes_non_retrained --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin`
- **Nuscenes (non-retrained):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/default/second_nuscenes.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_nuscenes_non_retrained `
- **eRod (non-retrained):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/default/second_erod.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_eRod_non_retrained --data_path /OpenPCDet/erod/points --ext .npy`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_nuscenes/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_KITTI_retrained`
- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_nuscenes.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_nuscenes/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_nuscenes_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_erod.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_erod/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_eRod_retrained`

### SECOND (FOCAL)

- **KITTI (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_focal.yaml --ckpt /OpenPCDet/output/custom_models/second/focal/second_focal/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_KITTI_focal_non_retrained_1`
- **Nuscenes (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal.yaml --ckpt /OpenPCDet/output/custom_models/second/focal/second_focal/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_nuscenes_focal_non_retrained_1`
- **eRod (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_erod_focal.yaml --ckpt /OpenPCDet/output/custom_models/second/focal/second_focal/default/ckpt/checkpoint_epoch_80.pth --extra_tag SECOND_eRod_focal_non_retrained_1`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_focal.yaml --ckpt output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_KITTI_focal_retrained_1`

python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_KITTI_focal_retrained_1

- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_nuscenes_focal_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/focal/second_erod_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/second/focal/second_nuscenes_focal/default/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_eRod_focal_retrained`

### SECOND (TENT)

- **KITTI (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/TENT/second_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_KITTI_TENT_non_retrained`
- **Nuscenes (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/TENT/second_nuscenes_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_nuscenes_TENT_non_retrained`
- **eRod (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_erod_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag SECOND_eRod_TENT_non_retrained`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/TENT/second_TENT.yaml --ckpt /OpenPCDet/output/kitti_models/second_nuscenes/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_KITTI_TENT_retrained`
- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/second/TENT/second_nuscenes_TENT.yaml --ckpt /OpenPCDet/output/kitti_models/second_nuscenes/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_nuscenes_TENT_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_erod_TENT.yaml --ckpt /OpenPCDet/output/kitti_models/second_nuscenes/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag SECOND_eRod_TENT_retrained`

### POINTPILLAR (DEFAULT)
tools/cfgs/custom_models/pointpillar/TENT/
- **KITTI (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_KITTI_default_non_retrained`
- **Nuscenes (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_nuscenes_default_non_retrained`
- **eRod (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_erod.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_eRod_default_non_retrained`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_KITTI_default_retrained`
- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_nuscenes_default_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_erod.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_eRod_default_retrained`


### POINTPILLAR (TENT)
tools/cfgs/custom_models/pointpillar/TENT/
- **KITTI (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_KITTI_TENT_non_retrained`
- **Nuscenes (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_nuscenes_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_nuscenes_TENT_non_retrained`
- **eRod (non-retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_erod_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --extra_tag POINTPILLAR_eRod_TENT_non_retrained`

- **KITTI (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_TENT.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_KITTI_TENT_retrained`
- **Nuscenes (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_nuscenes_TENT.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_nuscenes_TENT_retrained`
- **eRod (retrained):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar/TENT/pointpillar_erod_TENT.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar_nuscenes/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag POINTPILLAR_eRod_TENT_retrained`












