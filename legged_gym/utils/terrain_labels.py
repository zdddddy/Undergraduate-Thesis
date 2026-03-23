import numpy as np


def build_proportions_cumsum(terrain_proportions):
    arr = np.asarray(list(terrain_proportions), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        arr = np.asarray([1.0], dtype=np.float64)
    return np.cumsum(arr).tolist()


def terrain_label_from_choice(choice, proportions_cumsum):
    n = len(proportions_cumsum)
    if n >= 1 and choice < proportions_cumsum[0]:
        return "smooth_slope"
    if n >= 2 and choice < proportions_cumsum[1]:
        return "rough_slope"
    if n >= 3 and choice < proportions_cumsum[2]:
        return "stairs_up"
    if n >= 4 and choice < proportions_cumsum[3]:
        return "stairs_down"
    if n >= 5 and choice < proportions_cumsum[4]:
        return "discrete_obstacles"
    if n >= 6 and choice < proportions_cumsum[5]:
        return "stepping_stones"
    if n >= 7 and choice < proportions_cumsum[6]:
        return "gap"
    return "pit"


def terrain_label_from_col(col_idx, num_cols, proportions_cumsum):
    choice = float(col_idx) / float(max(1, num_cols)) + 0.001
    return terrain_label_from_choice(choice, proportions_cumsum)


def terrain_family_from_label(label):
    if label in {"smooth_slope", "rough_slope"}:
        return "slope"
    if label in {"stairs_up", "stairs_down"}:
        return "stairs"
    if label == "discrete_obstacles":
        return "discrete_obstacles"
    if label == "stepping_stones":
        return "stepping_stones"
    if label == "gap":
        return "gap"
    if label == "pit":
        return "pit"
    return "unknown"


def terrain_metadata_from_indices(col_idx, row_idx, num_cols, num_rows, proportions_cumsum):
    col_idx = int(col_idx)
    row_idx = int(row_idx)
    choice = float(col_idx) / float(max(1, num_cols)) + 0.001
    terrain_type = terrain_label_from_choice(choice, proportions_cumsum)
    terrain_family = terrain_family_from_label(terrain_type)
    return {
        "terrain_col": col_idx,
        "terrain_row": row_idx,
        "terrain_choice": float(choice),
        "terrain_difficulty": float(row_idx) / float(max(1, num_rows)),
        "terrain_type": terrain_type,
        "terrain_family": terrain_family,
    }


def make_traj_label_record(meta, num_frames=0):
    terrain_type = str(meta["terrain_type"])
    terrain_family = str(meta["terrain_family"])
    return {
        "terrain_col": int(meta["terrain_col"]),
        "terrain_row": int(meta["terrain_row"]),
        "terrain_choice": float(meta["terrain_choice"]),
        "terrain_difficulty": float(meta["terrain_difficulty"]),
        "terrain_type": terrain_type,
        "terrain_type_confidence": 1.0,
        "strict_terrain_type": terrain_type,
        "collect_large_terrain_type": terrain_type,
        "collect_large_terrain_type_confidence": 1.0,
        "strict_collect_large_terrain_type": terrain_type,
        "collect_large_terrain_family": terrain_family,
        "num_frames": int(num_frames),
    }
