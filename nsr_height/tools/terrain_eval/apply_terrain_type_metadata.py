import argparse
import glob
import json
import os

import numpy as np


def _resolve_split_root(data_dir, split):
    split_root = data_dir
    explicit = os.path.join(data_dir, split)
    if os.path.isdir(explicit):
        split_root = explicit
    return split_root


def _traj_key_from_path(npz_path):
    path = npz_path.replace("\\", "/")
    parts = [p for p in path.split("/") if p]
    env = next((p for p in parts if p.startswith("env_")), None)
    traj = next((p for p in parts if p.startswith("traj_")), None)
    if env and traj:
        return f"{env}/{traj}"
    if traj:
        return traj
    return os.path.dirname(path)


def _load_labels(labels_json, label_field):
    with open(labels_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    by_traj = obj.get("labels_by_traj", {})
    labels = {}
    for k, rec in by_traj.items():
        if isinstance(rec, dict):
            labels[k] = (
                str(rec.get(label_field, "unknown")),
                float(rec.get("terrain_type_confidence", 0.0)),
            )
        else:
            labels[k] = ("unknown", 0.0)
    return labels


def _rewrite_npz(npz_path, terrain_type, confidence):
    with np.load(npz_path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    payload["terrain_type"] = np.asarray(terrain_type)
    payload["terrain_type_confidence"] = np.asarray(float(confidence), dtype=np.float32)

    tmp_path = npz_path + ".tmp"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, npz_path)


def run(args):
    split_root = _resolve_split_root(args.data_dir, args.split)
    npz_paths = sorted(glob.glob(os.path.join(split_root, "**", "*.npz"), recursive=True))
    if not npz_paths:
        raise RuntimeError(f"no npz files found under {split_root}")

    labels = _load_labels(args.labels_json, args.label_field)
    modified = 0
    skipped = 0

    print(f"[apply-meta] split_root={split_root}")
    print(f"[apply-meta] files={len(npz_paths)}")
    print(f"[apply-meta] mode={'DRY-RUN' if args.dry_run else 'APPLY'}")

    for i, npz_path in enumerate(npz_paths):
        if args.limit > 0 and i >= args.limit:
            break
        key = _traj_key_from_path(npz_path)
        terrain_type, conf = labels.get(key, ("unknown", 0.0))
        if args.skip_unknown and terrain_type == "unknown":
            skipped += 1
            continue

        if args.dry_run:
            if modified < 5:
                print(
                    f"[apply-meta] example {modified + 1}: {npz_path} "
                    f"-> terrain_type={terrain_type} conf={conf:.3f}"
                )
        else:
            _rewrite_npz(npz_path, terrain_type, conf)
        modified += 1

    print(f"[apply-meta] modified={modified} skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--labels_json", type=str, required=True)
    parser.add_argument(
        "--label_field",
        type=str,
        default="strict_terrain_type",
        help="field name inside labels_by_traj records, e.g. strict_terrain_type or strict_collect_large_terrain_type",
    )
    parser.add_argument("--skip_unknown", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
