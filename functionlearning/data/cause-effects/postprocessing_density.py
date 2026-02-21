from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path(
    "functionlearning/data/cause-effects/pairs_subset_xy_with_metadata.csv"
)
DEFAULT_OUTPUT_DIR = Path("functionlearning/data/cause-effects")
DEFAULT_BASE_NAME = "pairs_subset_human_functionlearning_density24"


@dataclass
class TaskSampleResult:
    sampled_rows: List[Dict[str, object]]
    qc_row: Dict[str, object]


def _format_input_value(x: float) -> str:
    return f"[{float(x)}]"


def _build_bin_edges(
    x_min: float, x_max: float, n_bins: int, bin_width: Optional[float]
) -> np.ndarray:
    if x_max == x_min:
        return np.array([x_min, x_max], dtype=float)

    if bin_width is None:
        return np.linspace(x_min, x_max, n_bins + 1)

    n_bins_from_width = max(1, int(np.ceil((x_max - x_min) / float(bin_width))))
    edges = x_min + np.arange(n_bins_from_width + 1, dtype=float) * float(bin_width)
    if edges[-1] < x_max:
        edges = np.append(edges, x_max)
    else:
        edges[-1] = x_max
    return edges


def _assign_bins(x_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) <= 2:
        return np.zeros_like(x_values, dtype=int)
    bins = np.digitize(x_values, edges[1:-1], right=False)
    return np.clip(bins, 0, len(edges) - 2).astype(int)


def _pop_random_idx(idxs: List[int], rng: np.random.Generator) -> int:
    pos = int(rng.integers(0, len(idxs)))
    return idxs.pop(pos)


def _select_evenly_spaced_bins(nonempty_bins: Sequence[int], k: int) -> List[int]:
    bins_sorted = sorted(nonempty_bins)
    n = len(bins_sorted)
    if k >= n:
        return list(bins_sorted)

    raw_positions = np.linspace(0, n - 1, k)
    used = set()
    selected_positions: List[int] = []

    for raw in raw_positions:
        base = int(round(float(raw)))
        if base not in used:
            used.add(base)
            selected_positions.append(base)
            continue

        left = base - 1
        right = base + 1
        picked = None
        while left >= 0 or right < n:
            if left >= 0 and left not in used:
                picked = left
                break
            if right < n and right not in used:
                picked = right
                break
            left -= 1
            right += 1
        if picked is None:
            continue
        used.add(picked)
        selected_positions.append(picked)

    selected_positions = sorted(selected_positions)[:k]
    return [bins_sorted[i] for i in selected_positions]


def _allocate_hamilton(
    weights: Dict[int, int], budget: int
) -> Dict[int, int]:
    if budget <= 0 or not weights:
        return {b: 0 for b in weights}
    total_w = sum(max(0, int(w)) for w in weights.values())
    if total_w == 0:
        return {b: 0 for b in weights}

    bins = sorted(weights)
    ideals: Dict[int, float] = {}
    floors: Dict[int, int] = {}
    for b in bins:
        ideal = budget * (float(weights[b]) / float(total_w))
        ideals[b] = ideal
        floors[b] = int(np.floor(ideal))

    allocated = sum(floors.values())
    leftovers = budget - allocated
    remainders = sorted(
        bins,
        key=lambda b: (ideals[b] - floors[b], weights[b], -b),
        reverse=True,
    )
    for i in range(leftovers):
        floors[remainders[i % len(remainders)]] += 1
    return floors


def _redistribute_without_replacement(
    remaining_by_bin: Dict[int, List[int]],
    deficit: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    picked: List[Tuple[int, int]] = []
    if deficit <= 0:
        return picked

    while deficit > 0:
        active_bins = [b for b in remaining_by_bin if remaining_by_bin[b]]
        if not active_bins:
            break
        # Round-robin over bins sorted by remaining density.
        active_bins = sorted(active_bins, key=lambda b: (len(remaining_by_bin[b]), -b), reverse=True)
        for b in active_bins:
            if deficit == 0:
                break
            if not remaining_by_bin[b]:
                continue
            row_idx = _pop_random_idx(remaining_by_bin[b], rng)
            picked.append((row_idx, b))
            deficit -= 1
    return picked


def _weighted_round_robin_bins(weights: Dict[int, int], n_draws: int) -> List[int]:
    if n_draws <= 0:
        return []
    items = [(b, int(max(0, w))) for b, w in sorted(weights.items()) if w > 0]
    if not items:
        return []
    total_weight = sum(w for _, w in items)
    scores = {b: 0 for b, _ in items}
    order: List[int] = []
    for _ in range(n_draws):
        for b, w in items:
            scores[b] += w
        picked = max(items, key=lambda item: (scores[item[0]], -item[0]))[0]
        scores[picked] -= total_weight
        order.append(picked)
    return order


def _safe_get(row: pd.Series, key: str) -> object:
    return row[key] if key in row else ""


def _sample_single_task_density(
    task_df: pd.DataFrame,
    task_id: int,
    task_seed: int,
    total_points: int,
    n_bins: int,
    bin_width: Optional[float],
    shuffle_trials: bool,
) -> TaskSampleResult:
    rng = np.random.default_rng(task_seed)
    task_df = task_df.copy()
    task_df["x"] = pd.to_numeric(task_df["x"], errors="coerce")
    task_df["y"] = pd.to_numeric(task_df["y"], errors="coerce")

    rows_before = len(task_df)
    task_df = task_df.dropna(subset=["x", "y"]).reset_index(drop=True)
    dropped_invalid = rows_before - len(task_df)

    qc: Dict[str, object] = {
        "seed": int(task_seed),
        "task_id": int(task_id),
        "rows_available": int(len(task_df)),
        "unique_x_available": int(task_df["x"].nunique()),
        "rows_dropped_invalid_xy": int(dropped_invalid),
        "total_points_target": int(total_points),
    }

    if len(task_df) == 0:
        qc.update(
            {
                "n_bins_used": 0,
                "non_empty_bins": 0,
                "empty_bins": 0,
                "sampled_points": 0,
                "coverage_assigned": 0,
                "density_assigned": 0,
                "redistributed_count": 0,
                "resampled_count": 0,
                "status": "failed_no_valid_rows",
                "notes": "No valid numeric rows",
            }
        )
        return TaskSampleResult(sampled_rows=[], qc_row=qc)

    x_min = float(task_df["x"].min())
    x_max = float(task_df["x"].max())
    edges = _build_bin_edges(x_min=x_min, x_max=x_max, n_bins=n_bins, bin_width=bin_width)
    n_bins_used = int(max(1, len(edges) - 1))

    bin_ids = _assign_bins(task_df["x"].to_numpy(dtype=float), edges)
    task_df["bin_id"] = bin_ids

    bin_to_rows: Dict[int, List[int]] = {b: [] for b in range(n_bins_used)}
    for ridx, b in enumerate(bin_ids.tolist()):
        bin_to_rows[b].append(ridx)

    non_empty_bins = [b for b in range(n_bins_used) if len(bin_to_rows[b]) > 0]
    empty_bins = [b for b in range(n_bins_used) if len(bin_to_rows[b]) == 0]

    # A) Coverage allocation.
    if len(non_empty_bins) <= total_points:
        coverage_bins = sorted(non_empty_bins)
    else:
        coverage_bins = _select_evenly_spaced_bins(non_empty_bins, total_points)

    selected: List[Tuple[int, int, str]] = []
    remaining_by_bin: Dict[int, List[int]] = {b: list(rows) for b, rows in bin_to_rows.items()}

    for b in coverage_bins:
        if not remaining_by_bin[b]:
            continue
        row_idx = _pop_random_idx(remaining_by_bin[b], rng)
        selected.append((row_idx, b, "coverage"))

    coverage_assigned = len(selected)
    remaining_budget = max(0, total_points - coverage_assigned)

    # B) Density-proportional allocation (Hamilton) across coverage bins.
    density_weights = {b: len(bin_to_rows[b]) for b in coverage_bins}
    extras_target = _allocate_hamilton(density_weights, remaining_budget)
    density_assigned = 0
    for b in sorted(extras_target):
        wanted = extras_target[b]
        if wanted <= 0:
            continue
        can_take = min(wanted, len(remaining_by_bin[b]))
        for _ in range(can_take):
            row_idx = _pop_random_idx(remaining_by_bin[b], rng)
            selected.append((row_idx, b, "density"))
            density_assigned += 1

    # C1) Redistribution without replacement across bins with remaining rows.
    deficit = max(0, total_points - len(selected))
    redistributed = _redistribute_without_replacement(remaining_by_bin, deficit, rng)
    for row_idx, b in redistributed:
        selected.append((row_idx, b, "redistributed"))
    redistributed_count = len(redistributed)

    # C2) Replacement if still short.
    deficit = max(0, total_points - len(selected))
    resampled_count = 0
    if deficit > 0:
        replacement_weights = {b: len(bin_to_rows[b]) for b in non_empty_bins}
        picked_bins = _weighted_round_robin_bins(replacement_weights, deficit)
        for b in picked_bins:
            row_idx = int(rng.choice(bin_to_rows[b]))
            selected.append((row_idx, b, "resampled"))
            resampled_count += 1

    if shuffle_trials:
        rng.shuffle(selected)

    sampled_rows: List[Dict[str, object]] = []
    for trial_id, (row_idx, b, stage) in enumerate(selected):
        row = task_df.iloc[int(row_idx)]
        sampled_rows.append(
            {
                "task_id": int(task_id),
                "trial_id": int(trial_id),
                "input": _format_input_value(float(row["x"])),
                "target": float(row["y"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "x_name": _safe_get(row, "x_name"),
                "y_name": _safe_get(row, "y_name"),
                "x_description": _safe_get(row, "x_description"),
                "y_description": _safe_get(row, "y_description"),
                "pair_id": _safe_get(row, "pair_id"),
                "source_file": _safe_get(row, "source_file"),
                "bin_id": int(b),
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]) if b + 1 < len(edges) else float(edges[-1]),
                "sample_stage": stage,
                "seed": int(task_seed),
            }
        )

    status = "ok"
    notes = ""
    if len(sampled_rows) < total_points:
        status = "failed_insufficient_rows"
        notes = "Could not reach total_points"
    elif resampled_count > 0:
        status = "resampled_to_target"
        notes = "Filled deficit using replacement draws"

    qc.update(
        {
            "n_bins_used": int(n_bins_used),
            "non_empty_bins": int(len(non_empty_bins)),
            "empty_bins": int(len(empty_bins)),
            "sampled_points": int(len(sampled_rows)),
            "coverage_assigned": int(coverage_assigned),
            "density_assigned": int(density_assigned),
            "redistributed_count": int(redistributed_count),
            "resampled_count": int(resampled_count),
            "status": status,
            "notes": notes,
        }
    )
    return TaskSampleResult(sampled_rows=sampled_rows, qc_row=qc)


def build_human_subset_density(
    input_csv: Path,
    output_csv: Path,
    qc_csv: Path,
    seed: int,
    total_points: int = 24,
    n_bins: int = 8,
    bin_width: Optional[float] = None,
    shuffle_trials: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if total_points <= 0:
        raise ValueError("total_points must be > 0")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    if bin_width is not None and bin_width <= 0:
        raise ValueError("bin_width must be > 0 when provided")

    df = pd.read_csv(input_csv)
    required = {"task_id", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"input_csv must contain columns: {sorted(required)}")

    sampled_all: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []

    task_ids = sorted(
        pd.to_numeric(df["task_id"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    task_seed_rng = np.random.default_rng(seed)
    run_timestamp = datetime.now(timezone.utc).isoformat()

    for task_id in task_ids:
        task_df = df[df["task_id"] == task_id].reset_index(drop=True)
        task_seed = int(task_seed_rng.integers(0, 2**32 - 1))
        result = _sample_single_task_density(
            task_df=task_df,
            task_id=task_id,
            task_seed=task_seed,
            total_points=total_points,
            n_bins=n_bins,
            bin_width=bin_width,
            shuffle_trials=shuffle_trials,
        )
        sampled_all.extend(result.sampled_rows)
        qc = result.qc_row
        qc["run_seed"] = int(seed)
        qc["run_timestamp_utc"] = run_timestamp
        qc_rows.append(qc)

    sampled_df = pd.DataFrame(sampled_all)
    qc_df = pd.DataFrame(qc_rows)

    if len(sampled_df) > 0:
        sampled_df = sampled_df.sort_values(["task_id", "trial_id"]).reset_index(drop=True)
    if len(qc_df) > 0:
        qc_df = qc_df.sort_values(["task_id"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    qc_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    return sampled_df, qc_df


def build_many_versions_density(
    input_csv: Path,
    output_dir: Path,
    seeds: Sequence[int],
    base_name: str,
    total_points: int = 24,
    n_bins: int = 8,
    bin_width: Optional[float] = None,
    shuffle_trials: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        out_csv = output_dir / f"{base_name}_seed{seed}.csv"
        out_qc = output_dir / f"{base_name}_seed{seed}_qc.csv"
        sampled_df, qc_df = build_human_subset_density(
            input_csv=input_csv,
            output_csv=out_csv,
            qc_csv=out_qc,
            seed=seed,
            total_points=total_points,
            n_bins=n_bins,
            bin_width=bin_width,
            shuffle_trials=shuffle_trials,
        )
        print(
            f"[seed={seed}] wrote {out_csv} ({len(sampled_df)} rows), "
            f"{out_qc} ({len(qc_df)} tasks)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Density-proportional full-space postprocessing for human function-learning stimuli."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-name", type=str, default=DEFAULT_BASE_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--total-points", type=int, default=24)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--bin-width", type=float, default=None)
    parser.add_argument("--no-shuffle-trials", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    build_many_versions_density(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        seeds=seeds,
        base_name=args.base_name,
        total_points=args.total_points,
        n_bins=args.n_bins,
        bin_width=args.bin_width,
        shuffle_trials=not args.no_shuffle_trials,
    )


if __name__ == "__main__":
    main()
