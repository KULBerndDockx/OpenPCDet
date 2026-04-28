from pathlib import Path
import os

base = Path('erod')
pts_dir = base / 'points'
lbl_dir = base / 'labels'
img_sets_dir = base / 'ImageSets'

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

