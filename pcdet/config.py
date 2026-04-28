from pathlib import Path

import yaml
from easydict import EasyDict


def _resolve_cfg_path(cfg_path, ref_cfg_file=None):
    """Resolve a config path string to an existing file.

    OpenPCDet configs historically assume the current working directory is `tools/`.
    This resolver keeps that behavior (try as-is first) but also supports running
    from the repo root by falling back to paths relative to the referencing YAML.
    """
    cfg_path = str(cfg_path)
    path = Path(cfg_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = []

    # 1) As-is relative to current working directory (legacy behavior)
    candidates.append(Path.cwd() / path)

    # 2) Relative to the referencing config file
    if ref_cfg_file is not None:
        ref_path = Path(ref_cfg_file).resolve()
        candidates.append(ref_path.parent / path)

        # 3) Relative to the nearest `tools/` ancestor (common layout: tools/cfgs/...)
        for parent in [ref_path.parent, *ref_path.parents]:
            if parent.name == 'tools':
                candidates.append(parent / path)
                break

    # 4) Relative to repo root
    repo_root = (Path(__file__).resolve().parent / '../').resolve()
    candidates.append(repo_root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = '\n'.join([str(c) for c in candidates])
    raise FileNotFoundError(
        f'Config file not found: {cfg_path}\nAttempted:\n{attempted}'
    )


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('----------- %s -----------' % (key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def merge_new_config(config, new_config, ref_cfg_file=None):
    if '_BASE_CONFIG_' in new_config:
        base_cfg_file = _resolve_cfg_path(new_config['_BASE_CONFIG_'], ref_cfg_file=ref_cfg_file)
        with open(base_cfg_file, 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)

        # Recursively merge, so base configs can themselves reference another base.
        merge_new_config(config=config, new_config=yaml_config, ref_cfg_file=base_cfg_file)

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val, ref_cfg_file=ref_cfg_file)

    return config


def cfg_from_yaml_file(cfg_file, config):
    cfg_file = _resolve_cfg_path(cfg_file)
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config, ref_cfg_file=cfg_file)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
