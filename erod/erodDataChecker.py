# %%
from pathlib import Path
import json

# Set this to your label folder (current directory by default)
LABEL_DIR = Path("erod/labels")
JSON_GLOB = "**/*.json"  # recursive

def _get(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def _as_list(v, n, default=0.0):
    if isinstance(v, (list, tuple)) and len(v) >= n:
        return list(v[:n])
    return [default] * n

def to_openpcdet_line(obj):
    # Class name
    name = _get(obj, ["name", "class", "label", "type", "category"], "Unknown")

    # KITTI/OpenPCDet fields
    truncation = float(_get(obj, ["truncated", "truncation"], 0.0))
    occlusion = int(_get(obj, ["occluded", "occlusion"], 0))
    alpha = float(_get(obj, ["alpha"], -10.0))

    # bbox: [xmin, ymin, xmax, ymax]
    bbox = _get(obj, ["bbox", "box2d"], None)
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("xmin", "ymin", "xmax", "ymax")):
            bbox = [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
        elif all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
    if bbox is None:
        xmin = _get(obj, ["xmin", "x1"], 0.0)
        ymin = _get(obj, ["ymin", "y1"], 0.0)
        xmax = _get(obj, ["xmax", "x2"], 0.0)
        ymax = _get(obj, ["ymax", "y2"], 0.0)
        bbox = [xmin, ymin, xmax, ymax]
    bbox = _as_list(bbox, 4, 0.0)

    # dimensions in KITTI order: h, w, l
    dims = _get(obj, ["dimensions", "dims", "size", "box3d_size"], None)
    if isinstance(dims, dict):
        h = _get(dims, ["h", "height"], 0.0)
        w = _get(dims, ["w", "width"], 0.0)
        l = _get(dims, ["l", "length"], 0.0)
        dims = [h, w, l]
    elif dims is not None:
        dims = list(dims)
        # Try to detect l,w,h ordering and convert to h,w,l when possible
        if len(dims) >= 3:
            # If explicit keys absent, assume already h,w,l
            dims = dims[:3]
    else:
        h = float(_get(obj, ["h", "height"], 0.0))
        w = float(_get(obj, ["w", "width"], 0.0))
        l = float(_get(obj, ["l", "length"], 0.0))
        dims = [h, w, l]
    dims = _as_list(dims, 3, 0.0)

    # location: x, y, z
    loc = _get(obj, ["location", "center", "position", "translation"], None)
    if isinstance(loc, dict):
        x = _get(loc, ["x"], 0.0)
        y = _get(loc, ["y"], 0.0)
        z = _get(loc, ["z"], 0.0)
        loc = [x, y, z]
    elif loc is None:
        x = _get(obj, ["x", "cx"], 0.0)
        y = _get(obj, ["y", "cy"], 0.0)
        z = _get(obj, ["z", "cz"], 0.0)
        loc = [x, y, z]
    loc = _as_list(loc, 3, 0.0)

    rotation_y = float(_get(obj, ["rotation_y", "rot_y", "yaw", "ry"], 0.0))
    score = _get(obj, ["score", "confidence"], None)

    fields = [
        str(name),
        f"{truncation:.6f}",
        f"{occlusion:d}",
        f"{alpha:.6f}",
        f"{float(bbox[0]):.6f}",
        f"{float(bbox[1]):.6f}",
        f"{float(bbox[2]):.6f}",
        f"{float(bbox[3]):.6f}",
        f"{float(dims[0]):.6f}",
        f"{float(dims[1]):.6f}",
        f"{float(dims[2]):.6f}",
        f"{float(loc[0]):.6f}",
        f"{float(loc[1]):.6f}",
        f"{float(loc[2]):.6f}",
        f"{rotation_y:.6f}",
    ]
    if score is not None:
        fields.append(f"{float(score):.6f}")
    return " ".join(fields)

def extract_objects(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("objects", "labels", "annotations", "instances", "items"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        # Some formats store one frame with objects
        for k in ("frame", "data", "result"):
            v = data.get(k)
            if isinstance(v, dict):
                for kk in ("objects", "labels", "annotations", "instances", "items"):
                    vv = v.get(kk)
                    if isinstance(vv, list):
                        return vv
        # fallback: treat dict as single object
        return [data]
    return []

created = 0
skipped = 0
failed = 0

for json_path in LABEL_DIR.glob(JSON_GLOB):
    txt_path = json_path.with_suffix(".txt")
    if txt_path.exists():
        skipped += 1
        continue

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        objects = extract_objects(data)
        lines = [to_openpcdet_line(obj) for obj in objects if isinstance(obj, dict)]

        with open(txt_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
            else:
                f.write("")  # create empty txt if no valid objects
        created += 1
    except Exception as e:
        failed += 1
        print(f"Failed: {json_path} -> {e}")

print(f"Done. created={created}, skipped_existing={skipped}, failed={failed}")


