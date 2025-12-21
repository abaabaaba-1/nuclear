import argparse
import json
import os
import pickle
from typing import List, Tuple

import numpy as np


def load_items_from_pkl(pkl_path: str):
    """Load evaluated items from a GSCO-Lite RandomSearch baseline .pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    all_mols = data.get("all_mols", [])  # list of (Item, gen_idx)
    items = [item for item, _ in all_mols if getattr(item, "total", None) is not None]
    return items


def extract_objectives_and_cells(items) -> Tuple[np.ndarray, List[list]]:
    """Extract (f_B, f_S, I_max) and corresponding cell lists from Items.

    Returns
    -------
    objs : np.ndarray, shape (N, 3)
        Columns are [f_B, f_S, I_max]. All are to be MINIMIZED.
    cells_list : list of list
        Each element is the `cells` list from the JSON value of the item.
    """
    import json as _json

    objs = []
    cells_list = []

    for item in items:
        prop = getattr(item, "property", None)
        if not prop:
            continue
        try:
            f_B = float(prop["f_B"])
            f_S = float(prop["f_S"])
            I_max = float(prop["I_max"])
        except KeyError:
            continue

        try:
            cfg = _json.loads(item.value)
            cells = cfg.get("cells", [])
        except Exception:
            cells = []

        objs.append([f_B, f_S, I_max])
        cells_list.append(cells)

    if not objs:
        raise RuntimeError("No valid items with f_B, f_S, I_max found in pkl file.")

    return np.asarray(objs, dtype=float), cells_list


def compute_pareto_front_min(objs: np.ndarray) -> np.ndarray:
    """Compute indices of the Pareto front for a minimization problem.

    Parameters
    ----------
    objs : np.ndarray, shape (N, M)
        Objective values, all to be MINIMIZED.

    Returns
    -------
    front_indices : np.ndarray of int
        Indices of non-dominated points.
    """
    n = objs.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        # Check if there exists any j that dominates i (strictly better in at least one
        # objective, and no worse in all objectives).
        less_equal = np.all(objs <= objs[i], axis=1)
        strictly_less = np.any(objs < objs[i], axis=1)
        dominated_by_any = np.any(np.logical_and(less_equal, strictly_less))
        if dominated_by_any:
            is_dominated[i] = True

    front_indices = np.nonzero(~is_dominated)[0]
    return front_indices


def select_seed_indices(objs: np.ndarray, front_idx: np.ndarray, k: int) -> List[int]:
    """Select up to k seed indices from the population.

    Strategy:
    - Take all non-dominated points first.
    - If front size > k: down-sample along f_B (objective 0) to roughly k points.
    - If front size < k: fill remaining slots with best dominated points (by sum of objectives).
    """
    n = objs.shape[0]
    front_idx = np.asarray(front_idx, dtype=int)

    if front_idx.size == 0:
        # Fallback: just take k best by sum of objectives
        order = np.argsort(np.sum(objs, axis=1))
        return order[:k].tolist()

    if front_idx.size >= k:
        # Sort front by f_B (objective 0) and pick ~k evenly spaced points
        front_sorted = front_idx[np.argsort(objs[front_idx, 0])]
        step = front_sorted.size / float(k)
        chosen = []
        for i in range(k):
            idx = int(round(i * step))
            if idx >= front_sorted.size:
                idx = front_sorted.size - 1
            chosen.append(int(front_sorted[idx]))
        # Remove potential duplicates while preserving order
        seen = set()
        unique_chosen = []
        for idx in chosen:
            if idx not in seen:
                seen.add(idx)
                unique_chosen.append(idx)
            if len(unique_chosen) >= k:
                break
        return unique_chosen

    # front smaller than k: take all front points, then fill the rest from dominated ones
    selected = set(front_idx.tolist())
    result = list(front_idx.tolist())

    remaining = k - len(result)
    if remaining <= 0:
        return result[:k]

    all_indices = np.arange(n)
    dominated_indices = [i for i in all_indices if i not in selected]
    if dominated_indices:
        # Sort dominated by sum of objectives (smaller is better)
        dominated_sorted = sorted(dominated_indices, key=lambda idx: np.sum(objs[idx]))
        result.extend(dominated_sorted[:remaining])

    return result[:k]


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    default_pkl = os.path.join(
        root,
        "results",
        "stellarator_coil_gsco_lite",
        "baselines",
        "RandomSearch_42.pkl",
    )
    default_out = os.path.join(root, "warm_start_seeds.json")

    parser = argparse.ArgumentParser(
        description=(
            "Generate GSCO-Lite warm-start seeds from a RandomSearch baseline pkl "
            "by selecting an approximate Pareto front."
        )
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default=default_pkl,
        help="Path to RandomSearch_XX.pkl (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=default_out,
        help="Output JSON file for warm-start seeds (default: %(default)s)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of seeds to select (default: %(default)s)",
    )

    args = parser.parse_args()

    print(f"Loading items from: {args.pkl}")
    items = load_items_from_pkl(args.pkl)
    print(f"Total items with non-null score: {len(items)}")

    objs, cells_list = extract_objectives_and_cells(items)
    print(f"Objective array shape: {objs.shape}")

    print("Computing Pareto front (minimization in f_B, f_S, I_max)...")
    front_idx = compute_pareto_front_min(objs)
    print(f"Pareto front size: {front_idx.size}")

    print(f"Selecting up to {args.k} seeds from front + near-front candidates...")
    seed_indices = select_seed_indices(objs, front_idx, args.k)
    print(f"Selected {len(seed_indices)} seeds.")

    seeds = [cells_list[i] for i in seed_indices]

    with open(args.out, "w") as f:
        json.dump(seeds, f, indent=2)

    print(f"Saved {len(seeds)} warm-start seeds to: {args.out}")


if __name__ == "__main__":
    main()
