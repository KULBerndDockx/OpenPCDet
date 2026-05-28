from pathlib import Path
from collections import Counter

base = Path('erod')
pts_dir = base / 'points'
lbl_dir = base / 'labels'
img_sets_dir = base / 'ImageSets'


def count_objects_by_class(label_dir: Path, image_set_file: Path = None) -> Counter:
    """Count object instances per class from KITTI-style label files.

    If image_set_file is provided, only ids listed in that split file are used.
    """
    class_counts = Counter()

    if image_set_file is not None:
        ids = [ln.strip() for ln in image_set_file.read_text().splitlines() if ln.strip()]
        label_files = [label_dir / f"{sample_id}.txt" for sample_id in ids]
    else:
        label_files = sorted(label_dir.glob('*.txt'))

    for label_file in label_files:
        if not label_file.exists():
            continue

        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # Support both KITTI-style labels (class first) and custom labels (class last).
            if parts[0].isalpha():
                class_name = parts[0]
            elif parts[-1].isalpha():
                class_name = parts[-1]
            else:
                class_name = 'UNKNOWN'

            class_counts[class_name] += 1

    return class_counts

# Counts
pts = set(p.stem for p in pts_dir.glob('*') if p.suffix in ('.npy', '.bin', '.pcd'))
lbls = set(p.stem for p in lbl_dir.glob('*.txt'))

print(f"Points files: {len(pts)}")
print(f"Label files:  {len(lbls)}")

# ImageSets
for f in img_sets_dir.glob('*.txt'):
    ids = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    s_ids = set(ids)
    missing = s_ids - pts
    print(f"--- {f.name}: {len(ids)} ids, {len(missing)} missing from points")
    if ids:
        print(f"  Example ids: {ids[:5]}")

# Overlaps
names = ['train.txt', 'val.txt', 'val_small.txt']
sets = {}
for name in names:
    p = img_sets_dir / name
    if p.exists():
        sets[name] = set(ln.strip() for ln in p.read_text().splitlines() if ln.strip())

set_names = list(sets.keys())
for i, a in enumerate(set_names):
    for b in set_names[i+1:]:
        inter = sets[a] & sets[b]
        if inter:
            print(f"OVERLAP {a} vs {b}: {len(inter)} (e.g. {next(iter(inter))})")

# Class counts (all labels)
all_class_counts = count_objects_by_class(lbl_dir)
print('--- Object counts by class (all labels)')
for cls_name, count in sorted(all_class_counts.items()):
    print(f"{cls_name}: {count}")

# Class counts by split
for f in sorted(img_sets_dir.glob('*.txt')):
    split_class_counts = count_objects_by_class(lbl_dir, image_set_file=f)
    print(f"--- Object counts by class ({f.name})")
    if not split_class_counts:
        print('No objects found or no matching label files.')
        continue
    for cls_name, count in sorted(split_class_counts.items()):
        print(f"{cls_name}: {count}")

