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
- **nuScenes:** `python3 train.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_nuscenes.yaml --work_dir /OpenPCDet/work_dirs/second_nuscenes`
- **Nuscenes (Focal):** `python3 train.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_nuscenes_focal.yaml 
- **EROD (Fine-tune):** `python3 train.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_erod.yaml --pretrained_model /OpenPCDet/pcdet/models/pth/second_7862.pth`
- **Distributed (Focal):** `python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py /OpenPCDet/tools/cfgs/kitti_models/second_focal.yaml --work_dir ./work_dirs/CONFIG



---

## 3. Testing / Evaluation (`test.py`)
*Commands for model evaluation (val/test splits).*

### KITTI
- **PointPillar:** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth`
- **PointPillar (TENT):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/pointpillar_TENT.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth`
- **SECOND:** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth`
- **SECOND (FOCAL):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_focal/default/ckpt/checkpoint_epoch_20.pth`
- **PointRCNN:** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/pointrcnn.yaml --ckpt /OpenPCDet/pcdet/models/pth/PointRCNN.pth`

### EROD
- **PointPillar:** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/pointpillar_erod.yaml --ckpt /OpenPCDet/pcdet/models/pth/pointpillar_7728.pth --set DATA_CONFIG.DATA_SPLIT.test val_small`
- **SECOND:** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_erod.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth`
- **SECOND (FOCAL):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_focal/default/ckpt/checkpoint_epoch_20.pth`

### nuScenes
- **SECOND (FOCAL):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_nuscenes_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_focal/default/ckpt/checkpoint_epoch_20.pth`   

### nuScenes (Converted)
- **SECOND (FOCAL):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/second_nuscenes_converted_focal.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_nuscenes_focal/default/ckpt/checkpoint_epoch_80.pth`


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
- **SECOND:** `cd /OpenPCDet/tools && python3 demo.py --cfg_file cfgs/kitti_models/second_nuscenes.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_nuscenes/default/ckpt/checkpoint_epoch_64.pth --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin`