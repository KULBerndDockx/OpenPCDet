from pathlib import Path
from easydict import EasyDict
import yaml
import pickle
from pcdet.datasets.custom.custom_dataset import CustomDataset
from pcdet.utils import common_utils

dataset_cfg = EasyDict(yaml.safe_load(open('/OpenPCDet/tools/cfgs/dataset_configs/custom_dataset.yaml')))
data_path = Path(dataset_cfg.DATA_PATH)
class_names = ['Car', 'Pedestrian', 'Cyclist']
num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

dataset = CustomDataset(
    dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
    training=False, logger=common_utils.create_logger()
)

test_split = 'test'
test_filename = data_path / ('custom_infos_%s.pkl' % test_split)

dataset.set_split(test_split)
custom_infos_test = dataset.get_infos(
    class_names, num_workers=4, has_label=True, num_features=num_features
)
with open(test_filename, 'wb') as f:
    pickle.dump(custom_infos_test, f)
print('Custom info test file is saved to %s' % test_filename)
