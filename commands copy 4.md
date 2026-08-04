# OpenPCDet Commands Cheat Sheet

## 1. Data Preparation & Utilities



python3 nuscenes2kitti.py --nuscenes_dir <nuscenes_directory> --output_dir <output_directory>
python3 ns2kitti.py --nuscenes_dir /OpenPCDet/datasets/v1.0-trainval --output_dir /OpenPCDet/datasets/nsc


*Commands for generating infos and visualizing ground truth.*

### Info Generation

- **KITT:** `cd /OpenPCDet && python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml && cd tools`
- **EROD:** `cd /OpenPCDet && python3 -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml && cd tools`
- **nuSc (Mini):**           `python3 -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_N_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-mini`
- **nuSc (Trainval):**       `python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/nuscenes_converted.yaml`
- **Nusc:** `cd /OpenPCDet && python3 -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_N_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_converted.yaml && cd tools`

python3 -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval
python3 -m pcdet.datasets.kitti.kitti_dataset --func create_kitti_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_converted.yaml --version v1.0-trainval

### Visualization

- **EROD Labels:** `python3 visualize_labels.py --data_path /OpenPCDet/erod/points --label_path /OpenPCDet/erod/labels --ext .npy --z_min -1.0 --z_max 3.0`
- **nuSc Labels:** `cd /OpenPCDet/tools && python3 visualize_labels.py --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --label_path /OpenPCDet/erod/labels --ext .npy   --z_min -1.0 --z_max 3.0`


---

## 2. Training (`train.py`)

*Commands to train or fine-tune models.*

### SECOND

- **nuSc (Fine-tune):** `python3 train.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_N.yaml --pretrained_model /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag unfrozen_backbone_1_93 --freeze_backbone`

- **KITT (Focal):** `python3 train.py --cfg_file cfgs/models/S_F.yaml` done
- **nuSc (Focal) (Fine-tune):** `python3 train.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_N.yaml --pretrained_model /OpenPCDet/output/custom_models/S/F/S_F/default/ckpt/checkpoint_epoch_80.pth`


### PointPillar
- **nuSc (Fine-tune):** `python3 train.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_N.yaml --pretrained_model /OpenPCDet/pcdet/models/pth/pointpillars_7728.pth --extra_tag unfrozen_backbone_0 --freeze_backbone`

- **KITT (Focal):** `python3 train.py --cfg_file cfgs/models/P_F.yaml` 
- **nuSc (Focal) (Fine-tune):** `python3 train.py --cfg_file cfgs/models/P_F_N.yaml --pretrained_model /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P_F/default/ckpt/checkpoint_epoch_40.pth`


---

## 3. Testing / Evaluation (`test.py`)
### SECOND (DEFAULT)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_K.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_D_K_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_D_N_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_D_E_def`

python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_N.yaml --ckpt /OpenPCDet/output/models/S/D/S_D_K/S_K/ckpt/checkpoint_epoch_80.pth --extra_tag S_D_N_def

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_K.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S_N/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S_N/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_E.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/D/S_D_N/unfrozen_backbone_1_92/ckpt/checkpoint_epoch_40.pth`

### SECOND (FOCAL)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_K.yaml --ckpt /OpenPCDet/output/custom_models/S/F/S_F/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_K_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_N.yaml --ckpt /OpenPCDet/output/custom_models/S/F/S_F/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_N_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_E.yaml --ckpt /OpenPCDet/output/custom_models/S/F/S_F/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_E_def`

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_K.yaml --ckpt /OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_K_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_N_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_E.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_E_ftd`

### SECOND (TENT)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_K.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_T_K_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_T_N_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_T_E_def`

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_K.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_K_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_N.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_N_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_E.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_E_ftd`

### POINTPILLARS (DEFAULT)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_K.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_K_default_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_N_default_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_E_default_def`

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_K.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_K_default_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_N_default_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_E.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_E_default_ftd`

### POINTPILLARS (TENT)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_K.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_K_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_N_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_E_def`

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_K.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_T_K_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_T_N_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_E.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_T_E_ftd`


---
## 3.bis Testing / Evaluation (`test.py`)
### SECOND (DEFAULT)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_K.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_D_K_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_D_N_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_D_E_def`

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_K.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/D/S_D_Nl/unfrozen_backbone_1/ckpt --eval_all --extra_tag S_D_K_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_N.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/D/S_D_N/unfrozen_backbone_1/ckpt --eval_all --extra_tag S_D_N_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_D_E.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/D/S_D_N/unfrozen_backbone_1/ckpt --eval_all --extra_tag S_D_E_ftd`

### SECOND (FOCAL)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_K.yaml -ckpt_dir /OpenPCDet/output/cfgs/models/S/F/S_F_K/default/ckpt --eval_all --extra_tag S_F_K_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_N.yaml -ckpt_dir /OpenPCDet/output/cfgs/models/S/F/S_F_K/default/ckpt --eval_all --extra_tag S_F_N_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_E.yaml -ckpt_dir /OpenPCDet/output/cfgs/models/S/F/S_F_K/default/ckpt --eval_all --extra_tag S_F_E_def`

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_K.yaml -ckpt_dir /OpenPCDet/output/cfgs/models/S/F/S_F_N/default/ckpt --eval_all --extra_tag S_F_K_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_N.yaml -ckpt_dir /OpenPCDet/output/cfgs/models/S/F/S_F_N/default/ckpt --eval_all --extra_tag S_F_N_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_E.yaml -ckpt_dir /OpenPCDet/output/cfgs/models/S/F/S_F_N/default/ckpt --eval_all --extra_tag S_F_E_ftd`

### SECOND (TENT)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_K.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_T_K_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_T_N_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/second_7862.pth --extra_tag S_T_E_def`

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_K.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_K_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_N.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_N_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_E.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_E_ftd`

### POINTPILLARS (DEFAULT)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_K.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_K_default_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_N_default_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_E_default_def`

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_K.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_D_N/unfrozen_backbone_1/ckpt --eval_all --extra_tag P_K_default_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_N.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_D_N/unfrozen_backbone_1/ckpt --eval_all --extra_tag P_N_default_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_E.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_D_N/unfrozen_backbone_1/ckpt --eval_all --extra_tag P_E_default_ftd`

### POINTPILLARS (TENT)
- **KITT(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_K.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_K_def`
- **nuSc(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_N_def`
- **eRod(def):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_E_def`

- **KITT(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_K.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_D_N/unfrozen_backbone_1/ckpt --eval_all --extra_tag P_T_K_ftd`
- **nuSc(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_N.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_D_N/unfrozen_backbone_1/ckpt --eval_all --extra_tag P_T_N_ftd`
- **eRod(ftd):** `python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_E.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_D_N/unfrozen_backbone_1/ckpt --eval_all --extra_tag P_T_E_ftd`


---

## 3.tris Testing / Evaluation Tensorboard
### SECOND (DEFAULT)
- **KITT(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all `
- **nuSc(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`

- **KITT(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **nuSc(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`

### SECOND (FOCAL)
- **KITT(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **nuSc(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_E/default/eval/eval_all_default/default/tensorboard_val  --bind_all`

- **KITT(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **nuSc(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`

### SECOND (TENT)
- **KITT(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **nuSc(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`

- **KITT(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **nuSc(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`

### POINTPILLARS (DEFAULT)
- **KITT(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **nuSc(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`

- **KITT(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **nuSc(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`

### POINTPILLARS (TENT)
- **KITT(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **nuSc(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(def):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`

- **KITT(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **nuSc(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`
- **eRod(ftd):** `tensorboard --logdir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_K/default/eval/eval_all_default/default/tensorboard_val  --bind_all`





## 4. Demo / Inference (`demo.py`)

*Commands for running inference on raw point cloud files and generating predictions in the form of images.*

### KITTI

- **PointPillar:** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --data_path /OpenPCDet/datasets/kitti/testing/velodyne`

### EROD

- **PointPillar:** `cd /OpenPCDet/tools && python3 demo.py --cfg_file cfgs/models/P_E.yaml --ckpt ../pcdet/models/pth/P_7728.pth --data_path /OpenPCDet/erod/points --ext .npy`
- **SECOND:** `cd /OpenPCDet/tools && python3 demo.py --cfg_file cfgs/models/S_E.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S_E/default/ckpt/checkpoint_epoch_80.pth --data_path /OpenPCDet/erod/points --ext .npy`
- **SECOND (FOCAL):** `cd /OpenPCDet/tools && python3 demo.py --cfg_file cfgs/models/S_F_E.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S_F/default/ckpt/checkpoint_epoch_20.pth --data_path /OpenPCDet/erod/points --ext .npy`


### nuScenes

- **SECOND:** `cd /OpenPCDet/tools && python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/S_7862.pth --extra_tag S_N_def --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin`


- **SECOND:** `cd /OpenPCDet/tools && python3 demo.py --cfg_file cfgs/models/S_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S_N/default/ckpt_2/checkpoint_epoch_80.pth --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin`

- **SECOND(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin--extra_tag S_F_N_ftd`



- **SECOND FOCAL(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin--extra_tag S_F_N_ftd`

### SECOND (DEFAULT)
custom_models/S/default/
- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/S_7862.pth --extra_tag S_N_def --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin`
- **nuSc(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/S_7862.pth --extra_tag S_N_def `
- **eRod(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/S_7862.pth --extra_tag S_E_def --data_path /OpenPCDet/erod/points --ext .npy`

- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_Ktd --data_path /OpenPCDet/datasets/kitti/testing/velodyne --ext .bin`
- **nuSc(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_N.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_Ntd --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin`
- **eRod(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_E.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_Etd --data_path /OpenPCDet/erod/points --ext .npy`

### SECOND (FOCAL)

- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F.yaml --ckpt /OpenPCDet/output/custom_models/S/F/S_F/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_K_def_1`
- **nuSc(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_N.yaml --ckpt /OpenPCDet/output/custom_models/S/F/S_F/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_N_def_1`
- **eRod(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_E.yaml --ckpt /OpenPCDet/output/custom_models/S/F/S_F/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_E_def_1 --data_path /OpenPCDet/erod/points --ext .npy`

- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F.yaml --ckpt output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_K_ftd_1`

python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_K_ftd_1

- **nuSc(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_N_ftd`
- **eRod(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F_E.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_E_ftd`

### SECOND (TENT)

- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T.yaml --ckpt /OpenPCDet/pcdet/models/pth/S_7862.pth --extra_tag S_T_K_def`
- **nuSc(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/S_7862.pth --extra_tag S_T_N_def`
- **eRod(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/S_7862.pth --extra_tag S_T_E_def --data_path /OpenPCDet/erod/points --ext .npy`

- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_K_ftd`
- **nuSc(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_N.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_N_ftd`
- **eRod(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T_E.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_E_ftd --data_path /OpenPCDet/erod/points --ext .npy`

### POINTPILLAR (DEFAULT)
tools/cfgs/models/P/T/
- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_K_default_def`
- **nuSc(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_N_default_def`
- **eRod(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_E_default_def`

- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_K_default_ftd`
- **nuSc(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_N_default_ftd`
- **eRod(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_E.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_E_default_ftd --data_path /OpenPCDet/erod/points --ext .npy`


### POINTPILLAR (TENT)
tools/cfgs/models/P/T/
- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_K_def`
- **nuSc(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_N_def`
- **eRod(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_E.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_E_def`

- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_T_K_ftd`
- **nuSc(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_N.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_T_N_ftd`
- **eRod(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T_E.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_T_E_ftd`







## 5. vis

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F/S_KITTI_F_def_1/eval/epoch_80/val/default/result.pkl 


python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/default/eval/epoch_7862/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/default/eval/epoch_40/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F/S_KITTI_F_def_1/eval/epoch_80/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/F/S_F/S_KITTI_F_ftd_1/eval/epoch_40/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/T/S_T/S_KITTI_T_def/eval/epoch_7862/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S/T/S_T/S_KITTI_T_ftd/eval/epoch_40/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P/P_KITTI_default_def/eval/epoch_7728/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P/P_KITTI_default_ftd/eval/epoch_40/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/T/P_T/P_KITTI_T_def/eval/epoch_7728/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/kitti/training/velodyne --label_path /OpenPCDet/datasets/kitti/training/label_2 --ext .bin --single-image --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/T/P_T/P_KITTI_T_ftd/eval/epoch_40/val/default/result.pkl 

python3 visualize_labels.py --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --label_path /OpenPCDet/erod/labels --ext .npy  --ext .bin --single-image  --result_pkl /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/P_N_default_def/eval/epoch_7728/test/default/result.pkl


--z_min -1.0 --z_max 3.0`


### SECOND (DEFAULT)
custom_models/S/default/
- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/D/S_N.yaml --ckpt /OpenPCDet/pcdet/models/pth/S_7862.pth --extra_tag S_N_def --data_path /OpenPCDet/datasets/nuscenes/v1.0-mini/sweeps/LIDAR_TOP --ext .bin`


- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/S_N/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_Ktd`


### SECOND (FOCAL)

- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F.yaml --ckpt /OpenPCDet/output/custom_models/S/F/S_F/default/ckpt/checkpoint_epoch_80.pth --extra_tag S_F_K_def_1`


- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/F/S_F.yaml --ckpt output/OpenPCDet/tools/cfgs/models/S/F/S_F_N/default/ckpt/checkpoint_epoch_40.pth --extra_tag S_F_K_ftd_1`

### SECOND (TENT)

- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T.yaml --ckpt /OpenPCDet/pcdet/models/pth/S_7862.pth --extra_tag S_T_K_def`


- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/S/T/S_T.yaml --ckpt /OpenPCDet/output/models/S_N/frozen_backbone_8/ckpt/checkpoint_epoch_40.pth --extra_tag S_T_K_ftd`

### POINTPILLAR (DEFAULT)
tools/cfgs/models/P/T/
- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_K_default_def`


- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_K_default_ftd`


### POINTPILLAR (TENT)
tools/cfgs/models/P/T/
- **KITT(def):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T.yaml --ckpt /OpenPCDet/pcdet/models/pth/P_7728.pth --extra_tag P_T_K_def`


- **KITT(ftd):** `python3 demo.py --cfg_file /OpenPCDet/tools/cfgs/models/P/T/P_T.yaml --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_N/frozen_backbone_1/ckpt/checkpoint_epoch_40.pth --extra_tag P_T_K_ftd`

##/OpenPCDet/tools/cfgs/models/P/D/P_D_N.yaml
python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_N.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_D_N/unfrozen_backbone_1/ckpt --eval_all

python3 test.py --cfg_file /OpenPCDet/tools/cfgs/models/P/D/P_D_E.yaml --ckpt_dir /OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_D_N/unfrozen_backbone_1/ckpt --eval_all
/OpenPCDet/output/OpenPCDet/tools/cfgs/models/P/D/P_D_N/unfrozen_backbone_1/ckpt