from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path(
    "/Users/aj9225/Local/meta-learning-ecological-priors-from-llms/functionlearning/data/cause-effects/pairs_subset_xy_with_metadata.csv")
DEFAULT_OUTPUT_DIR = Path(
    "/Users/aj9225/Local/meta-learning-ecological-priors-from-llms/functionlearning/data/cause-effects")
DEFAULT_BASE_NAME = "pairs_subset_human_functionlearning_24pts"


@dataclass
class TaskSampleResult:
    sampled_rows: List[Dict[str, object]]
    qc_row: Dict[str, object]


def _format_input_value(x: float) -> str:
    # Keep compatibility with existing function-learning CSV format.
    return f"[{float(x)}]"


def _build_bin_edges(x_min: float, x_max: float, n_bins: int, bin_width: Optional[float]) -> np.ndarray:
    if x_max == x_min:
        return np.array([x_min, x_max], dtype=float)

    if bin_width is None:
        return np.linspace(x_min, x_max, n_bins + 1)

    n_bins_from_width = max(1, int(np.ceil((x_max - x_min) / bin_width)))
    edges = x_min + np.arange(n_bins_from_width + 1,
                              dtype=float) * float(bin_width)
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


def _redistribute_without_replacement(
    remaining_by_bin: Dict[int, List[int]],
    deficit: int,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int, int]], int]:
    picked: List[Tuple[int, int]] = []
    if deficit <= 0:
        return picked, 0

    all_bins = sorted(remaining_by_bin.keys())
    while deficit > 0:
        active_bins = [b for b in all_bins if remaining_by_bin[b]]
        if not active_bins:
            break
        for b in active_bins:
            if deficit == 0:
                break
            idx = _pop_random_idx(remaining_by_bin[b], rng)
            picked.append((idx, b))
            deficit -= 1
    return picked, deficit


def _resample_with_replacement(
    bin_to_all_rows: Dict[int, List[int]],
    deficit: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    picked: List[Tuple[int, int]] = []
    if deficit <= 0:
        return picked

    active_bins = [b for b in sorted(
        bin_to_all_rows.keys()) if len(bin_to_all_rows[b]) > 0]
    if not active_bins:
        return picked

    cursor = 0
    while deficit > 0:
        b = active_bins[cursor % len(active_bins)]
        row_idx = int(rng.choice(bin_to_all_rows[b]))
        picked.append((row_idx, b))
        deficit -= 1
        cursor += 1
    return picked


def _sample_single_task(
    task_df: pd.DataFrame,
    task_id: int,
    seed: int,
    n_bins: int,
    points_per_bin: int,
    bin_width: Optional[float],
    shuffle_trials: bool,
) -> TaskSampleResult:
    rng = np.random.default_rng(seed)
    task_df = task_df.copy()
    task_df["x"] = pd.to_numeric(task_df["x"], errors="coerce")
    task_df["y"] = pd.to_numeric(task_df["y"], errors="coerce")

    rows_before = len(task_df)
    task_df = task_df.dropna(subset=["x", "y"]).reset_index(drop=True)
    dropped_invalid = rows_before - len(task_df)

    base_qc = {
        "seed": seed,
        "task_id": task_id,
        "rows_available": int(len(task_df)),
        "unique_x_available": int(task_df["x"].nunique()),
        "points_per_bin": int(points_per_bin),
        "rows_dropped_invalid_xy": int(dropped_invalid),
    }

    if len(task_df) == 0:
        qc = {
            **base_qc,
            "n_bins_used": 0,
            "target_points": 0,
            "sampled_points": 0,
            "resampled_count": 0,
            "non_empty_bins": 0,
            "empty_bins": 0,
            "status": "failed_no_valid_rows",
            "notes": "No valid numeric x/y rows",
        }
        return TaskSampleResult(sampled_rows=[], qc_row=qc)

    x_min = float(task_df["x"].min())
    x_max = float(task_df["x"].max())
    edges = _build_bin_edges(x_min=x_min, x_max=x_max,
                             n_bins=n_bins, bin_width=bin_width)
    n_bins_used = int(max(1, len(edges) - 1))
    target_points = int(n_bins_used * points_per_bin)

    bin_ids = _assign_bins(task_df["x"].to_numpy(dtype=float), edges)
    task_df["bin_id"] = bin_ids

    bin_to_rows: Dict[int, List[int]] = {b: [] for b in range(n_bins_used)}
    for row_idx, b in enumerate(bin_ids.tolist()):
        bin_to_rows[b].append(row_idx)

    non_empty_bins = sum(1 for rows in bin_to_rows.values() if rows)
    empty_bins = n_bins_used - non_empty_bins

    selected: List[Tuple[int, int, str]] = []
    remaining_by_bin: Dict[int, List[int]] = {}
    for b in range(n_bins_used):
        row_idxs = list(bin_to_rows[b])
        if len(row_idxs) == 0:
            remaining_by_bin[b] = []
            continue
        local_idxs = list(row_idxs)
        picks = min(points_per_bin, len(local_idxs))
        for _ in range(picks):
            ridx = _pop_random_idx(local_idxs, rng)
            selected.append((ridx, b, "quota"))
        remaining_by_bin[b] = local_idxs

    deficit = max(0, target_points - len(selected))
    redistributed, deficit = _redistribute_without_replacement(
        remaining_by_bin=remaining_by_bin,
        deficit=deficit,
        rng=rng,
    )
    for ridx, b in redistributed:
        selected.append((ridx, b, "redistributed"))

    resampled_count = 0
    if deficit > 0:
        replacement = _resample_with_replacement(
            bin_to_all_rows=bin_to_rows,
            deficit=deficit,
            rng=rng,
        )
        resampled_count = len(replacement)
        for ridx, b in replacement:
            selected.append((ridx, b, "resampled"))

    if shuffle_trials:
        rng.shuffle(selected)

    sampled_rows: List[Dict[str, object]] = []
    for trial_id, (row_idx, b, stage) in enumerate(selected):
        row = task_df.iloc[int(row_idx)]
        out = {
            "task_id": int(task_id),
            "trial_id": int(trial_id),
            "input": _format_input_value(float(row["x"])),
            "target": float(row["y"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "x_name": row.get("x_name", ""),
            "y_name": row.get("y_name", ""),
            "x_description": row.get("x_description", ""),
            "y_description": row.get("y_description", ""),
            "pair_id": row.get("pair_id", ""),
            "source_file": row.get("source_file", ""),
            "bin_id": int(b),
            "bin_left": float(edges[b]),
            "bin_right": float(edges[b + 1]) if b + 1 < len(edges) else float(edges[-1]),
            "sample_stage": stage,
            "seed": int(seed),
        }
        sampled_rows.append(out)

    status = "ok"
    notes = ""
    if resampled_count > 0:
        status = "resampled_to_target"
        notes = "Filled remaining points by sampling with replacement"

    qc = {
        **base_qc,
        "n_bins_used": int(n_bins_used),
        "target_points": int(target_points),
        "sampled_points": int(len(sampled_rows)),
        "resampled_count": int(resampled_count),
        "non_empty_bins": int(non_empty_bins),
        "empty_bins": int(empty_bins),
        "status": status,
        "notes": notes,
    }
    return TaskSampleResult(sampled_rows=sampled_rows, qc_row=qc)


def build_human_subset(
    input_csv: Path,
    output_csv: Path,
    qc_csv: Path,
    seed: int,
    n_bins: int = 8,
    points_per_bin: int = 3,
    bin_width: Optional[float] = None,
    shuffle_trials: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    if points_per_bin <= 0:
        raise ValueError("points_per_bin must be > 0")
    if bin_width is not None and bin_width <= 0:
        raise ValueError("bin_width must be > 0 when provided")

    df = pd.read_csv(input_csv)
    if "task_id" not in df.columns or "x" not in df.columns or "y" not in df.columns:
        raise ValueError("input_csv must contain task_id, x, y columns")

    sampled_all: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []

    task_ids = sorted(pd.to_numeric(
        df["task_id"], errors="coerce").dropna().astype(int).unique().tolist())
    task_seed_rng = np.random.default_rng(seed)
    run_timestamp = datetime.now(timezone.utc).isoformat()

    for task_id in task_ids:
        task_df = df[df["task_id"] == task_id].reset_index(drop=True)
        task_seed = int(task_seed_rng.integers(0, 2**32 - 1))
        result = _sample_single_task(
            task_df=task_df,
            task_id=task_id,
            seed=task_seed,
            n_bins=n_bins,
            points_per_bin=points_per_bin,
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
        sampled_df = sampled_df.sort_values(
            ["task_id", "trial_id"]).reset_index(drop=True)
    if len(qc_df) > 0:
        qc_df = qc_df.sort_values(["task_id"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    qc_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)

    return sampled_df, qc_df


def build_many_versions(
    input_csv: Path,
    output_dir: Path,
    seeds: Sequence[int],
    base_name: str = DEFAULT_BASE_NAME,
    n_bins: int = 8,
    points_per_bin: int = 3,
    bin_width: Optional[float] = None,
    shuffle_trials: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        out_csv = output_dir / f"{base_name}_seed{seed}.csv"
        qc_csv = output_dir / f"{base_name}_seed{seed}_qc.csv"
        sampled_df, qc_df = build_human_subset(
            input_csv=input_csv,
            output_csv=out_csv,
            qc_csv=qc_csv,
            seed=seed,
            n_bins=n_bins,
            points_per_bin=points_per_bin,
            bin_width=bin_width,
            shuffle_trials=shuffle_trials,
        )
        print(
            f"[seed={seed}] wrote {out_csv} ({len(sampled_df)} rows), "
            f"{qc_csv} ({len(qc_df)} tasks)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process cause-effects x/y data into human function-learning stimuli "
            "using quota-bin stratified sampling."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-name", type=str, default=DEFAULT_BASE_NAME)
    parser.add_argument("--seed", type=int, default=42,
                        help="Single-run seed.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="If provided, generate one output version per seed.",
    )
    parser.add_argument("--n-bins", type=int, default=8,
                        help="Default number of equal-width bins.")
    parser.add_argument(
        "--points-per-bin",
        type=int,
        default=3,
        help="Quota points per bin. Default gives 24 points/task with 8 bins.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help="Optional numeric bin width in raw x units. Overrides fixed --n-bins per task.",
    )
    parser.add_argument(
        "--no-shuffle-trials",
        action="store_true",
        help="Disable random within-task trial order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    build_many_versions(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        seeds=seeds,
        base_name=args.base_name,
        n_bins=args.n_bins,
        points_per_bin=args.points_per_bin,
        bin_width=args.bin_width,
        shuffle_trials=not args.no_shuffle_trials,
    )


if __name__ == "__main__":
    main()
