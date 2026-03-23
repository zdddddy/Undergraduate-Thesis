#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TrajRecord:
    source_root: Path
    source_tag: str
    traj_dir: Path
    env_name: Optional[str]
    traj_name: str
    frame_count: int
    group_key: str


def _sanitize_tag(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "src"


def _find_env_and_traj_name(traj_dir: Path) -> Tuple[Optional[str], str]:
    traj_name = traj_dir.name
    env_name = None
    parent = traj_dir.parent
    if parent.name.startswith("env_"):
        env_name = parent.name
    return env_name, traj_name


def _find_traj_dirs(source_root: Path) -> List[Path]:
    # Common raw format: root/env_*/traj_*
    env_trajs = sorted(p for p in source_root.glob("env_*/traj_*") if p.is_dir())
    if env_trajs:
        return env_trajs

    # Fallback: root/traj_*
    root_trajs = sorted(p for p in source_root.glob("traj_*") if p.is_dir())
    if root_trajs:
        return root_trajs

    # Already split format: root/train|val/...
    split_trajs: List[Path] = []
    for split in ("train", "val"):
        split_root = source_root / split
        if not split_root.is_dir():
            continue
        split_trajs.extend(sorted(p for p in split_root.glob("env_*/traj_*") if p.is_dir()))
        split_trajs.extend(sorted(p for p in split_root.glob("traj_*") if p.is_dir()))
    return sorted(split_trajs)


def _u01(seed: int, key: str) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    val = int(digest[:15], 16)
    return float(val) / float(0xFFFFFFFFFFFFFFF)


def _ensure_not_all_one_side(assign: Dict[str, str], seed: int) -> Dict[str, str]:
    keys = sorted(assign.keys())
    if len(keys) <= 1:
        return assign
    vals = [assign[k] for k in keys]
    if all(v == "train" for v in vals):
        k = min(keys, key=lambda x: _u01(seed + 1009, x))
        assign[k] = "val"
    elif all(v == "val" for v in vals):
        k = max(keys, key=lambda x: _u01(seed + 2003, x))
        assign[k] = "train"
    return assign


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        os.symlink(str(src), str(dst), target_is_directory=True)
        return
    if mode == "copy":
        shutil.copytree(src, dst)
        return
    if mode == "hardlink":
        shutil.copytree(src, dst, copy_function=os.link)
        return
    raise ValueError(f"unsupported mode: {mode}")


def build_records(
    source_roots: List[Path],
    split_by: str,
    min_frames: int,
) -> List[TrajRecord]:
    records: List[TrajRecord] = []
    for root in source_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"source not found: {root}")
        source_tag = _sanitize_tag(root.name)
        traj_dirs = _find_traj_dirs(root)
        for traj_dir in traj_dirs:
            frame_count = len(list(traj_dir.glob("frame_*.npz")))
            if frame_count < min_frames:
                continue
            env_name, traj_name = _find_env_and_traj_name(traj_dir)
            if split_by == "env":
                group_key = env_name if env_name is not None else f"{source_tag}:{traj_name}"
            elif split_by == "source_env":
                group_key = f"{source_tag}:{env_name if env_name is not None else traj_name}"
            elif split_by == "traj":
                group_key = f"{source_tag}:{traj_dir.as_posix()}"
            else:
                raise ValueError(f"invalid split_by: {split_by}")
            records.append(
                TrajRecord(
                    source_root=root,
                    source_tag=source_tag,
                    traj_dir=traj_dir,
                    env_name=env_name,
                    traj_name=traj_name,
                    frame_count=frame_count,
                    group_key=group_key,
                )
            )
    if not records:
        raise RuntimeError("no valid trajectories found")
    return records


def assign_splits(records: List[TrajRecord], val_ratio: float, seed: int) -> Dict[str, str]:
    groups = sorted({r.group_key for r in records})
    n = len(groups)
    if n == 1:
        return {groups[0]: "train"}

    n_val = int(round(float(val_ratio) * float(n)))
    n_val = max(1 if val_ratio > 0.0 else 0, n_val)
    n_val = min(n - 1, n_val)

    ranked = sorted(groups, key=lambda g: _u01(seed, g))
    val_set = set(ranked[:n_val])
    split_of_group = {g: ("val" if g in val_set else "train") for g in groups}
    split_of_group = _ensure_not_all_one_side(split_of_group, seed=seed)
    return split_of_group


def build_dest_traj_path(
    rec: TrajRecord,
    out_root: Path,
    split: str,
    prefix_source: bool,
) -> Path:
    if rec.env_name is not None:
        if prefix_source:
            if rec.env_name.startswith("env_"):
                env_suffix = rec.env_name[len("env_") :]
                env_dst = f"env_{rec.source_tag}__{env_suffix}"
            else:
                env_dst = f"env_{rec.source_tag}__{rec.env_name}"
        else:
            env_dst = rec.env_name
        return out_root / split / env_dst / rec.traj_name

    # No env folder in source: still force env_*/traj_* shape for dataset loader compatibility.
    env_dst = f"env_{rec.source_tag}__misc"
    traj_dst = rec.traj_name if rec.traj_name.startswith("traj_") else f"traj_{rec.traj_name}"
    return out_root / split / env_dst / traj_dst


def summarize(records: List[TrajRecord], split_of_group: Dict[str, str]) -> Dict[str, object]:
    summary = {
        "trajectories_total": len(records),
        "frames_total": int(sum(r.frame_count for r in records)),
        "groups_total": len(split_of_group),
        "train_trajectories": 0,
        "val_trajectories": 0,
        "train_frames": 0,
        "val_frames": 0,
        "train_groups": 0,
        "val_groups": 0,
        "by_source": {},
    }
    summary["train_groups"] = int(sum(1 for s in split_of_group.values() if s == "train"))
    summary["val_groups"] = int(sum(1 for s in split_of_group.values() if s == "val"))

    by_source: Dict[str, Dict[str, int]] = {}
    for rec in records:
        split = split_of_group[rec.group_key]
        if split == "train":
            summary["train_trajectories"] += 1
            summary["train_frames"] += int(rec.frame_count)
        else:
            summary["val_trajectories"] += 1
            summary["val_frames"] += int(rec.frame_count)

        if rec.source_tag not in by_source:
            by_source[rec.source_tag] = {
                "trajectories_total": 0,
                "frames_total": 0,
                "train_trajectories": 0,
                "val_trajectories": 0,
                "train_frames": 0,
                "val_frames": 0,
            }
        src = by_source[rec.source_tag]
        src["trajectories_total"] += 1
        src["frames_total"] += int(rec.frame_count)
        if split == "train":
            src["train_trajectories"] += 1
            src["train_frames"] += int(rec.frame_count)
        else:
            src["val_trajectories"] += 1
            src["val_frames"] += int(rec.frame_count)

    summary["by_source"] = by_source
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val split by env or traj from one or more collected datasets.")
    parser.add_argument(
        "--source_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more source dataset roots, e.g. /data/dense /data/medium /data/sparse",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output dataset root containing train/ and val/")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio in [0,1].")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic hash split.")
    parser.add_argument(
        "--split_by",
        type=str,
        default="env",
        choices=["env", "source_env", "traj"],
        help="Grouping unit before split. env is recommended to avoid leakage.",
    )
    parser.add_argument(
        "--link_mode",
        type=str,
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="How to populate train/val trajectories.",
    )
    parser.add_argument("--min_frames", type=int, default=20, help="Drop trajectories shorter than this.")
    parser.add_argument(
        "--prefix_source",
        action="store_true",
        default=False,
        help="Prefix env folder with source name even when only one source dir.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Remove existing out_dir/train and out_dir/val before building.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False, help="Only print summary without writing links.")
    args = parser.parse_args()

    if not (0.0 <= args.val_ratio <= 1.0):
        raise ValueError("--val_ratio must be in [0,1]")

    source_roots = [Path(x).expanduser().resolve() for x in args.source_dirs]
    out_root = Path(args.out_dir).expanduser().resolve()

    records = build_records(
        source_roots=source_roots,
        split_by=args.split_by,
        min_frames=max(1, int(args.min_frames)),
    )
    split_of_group = assign_splits(records, val_ratio=float(args.val_ratio), seed=int(args.seed))
    summary = summarize(records, split_of_group)

    auto_prefix = len(source_roots) > 1
    prefix_source = bool(args.prefix_source or auto_prefix)

    print("[split] summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[split] split_by={args.split_by} val_ratio={args.val_ratio} seed={args.seed}")
    print(f"[split] link_mode={args.link_mode} min_frames={args.min_frames} prefix_source={prefix_source}")
    print(f"[split] out_dir={out_root}")

    if args.dry_run:
        return

    train_dir = out_root / "train"
    val_dir = out_root / "val"
    if args.clean:
        for d in (train_dir, val_dir):
            if d.exists() or d.is_symlink():
                shutil.rmtree(d)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        split = split_of_group[rec.group_key]
        dst = build_dest_traj_path(rec, out_root=out_root, split=split, prefix_source=prefix_source)
        _link_or_copy(rec.traj_dir, dst, mode=args.link_mode)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_root),
        "source_dirs": [str(x) for x in source_roots],
        "split_by": args.split_by,
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "link_mode": args.link_mode,
        "min_frames": int(args.min_frames),
        "prefix_source": bool(prefix_source),
        "summary": summary,
    }
    manifest_path = out_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[split] done. manifest={manifest_path}")


if __name__ == "__main__":
    main()
