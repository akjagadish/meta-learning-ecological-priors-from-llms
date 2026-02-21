from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


# functionlearning_density24_seed*
DEFAULT_INPUT_GLOB = "functionlearning/data/cause-effects/pairs_subset_human_functionlearning_24pts_seed*.csv"#functionlearning_density24_seed*
DEFAULT_OUTPUT_DIR = Path("functionlearning/data/cause-effects")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot postprocessed human function-learning datasets. "
            "Can plot one CSV or many seed CSVs via glob."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Single postprocessed CSV to plot.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=DEFAULT_INPUT_GLOB,
        help="Glob for multiple postprocessed CSVs. Ignored if --input-csv is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of subplot columns.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG dpi.",
    )
    parser.add_argument(
        "--color-by-stage",
        action="store_true",
        help="Color points by sample_stage (quota/redistributed/resampled).",
    )
    return parser.parse_args()


def resolve_inputs(input_csv: Path | None, input_glob: str) -> List[Path]:
    if input_csv is not None:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        return [input_csv]

    matches = sorted(Path(".").glob(input_glob))
    # Ignore QC files if matched by broad glob.
    matches = [p for p in matches if not p.name.endswith("_qc.csv")]
    if not matches:
        raise FileNotFoundError(f"No CSVs matched glob: {input_glob}")
    return matches


def plot_one_csv(csv_path: Path, output_dir: Path, cols: int, dpi: int, color_by_stage: bool) -> None:
    df = pd.read_csv(csv_path)
    needed = {"task_id", "x", "y"}
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    task_ids = sorted(pd.to_numeric(
        df["task_id"], errors="coerce").dropna().astype(int).unique().tolist())
    if len(task_ids) == 0:
        raise ValueError(f"No valid task_id rows in {csv_path}")

    rows = math.ceil(len(task_ids) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4.6, rows * 3.8),
        constrained_layout=True,
    )
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    stage_colors = {
        "quota": "#1f77b4",
        "redistributed": "#ff7f0e",
        "resampled": "#d62728",
    }

    for i, task_id in enumerate(task_ids):
        ax = axes[i]
        task_df = df[df["task_id"] == task_id].copy()
        task_df["x"] = pd.to_numeric(task_df["x"], errors="coerce")
        task_df["y"] = pd.to_numeric(task_df["y"], errors="coerce")
        task_df = task_df.dropna(subset=["x", "y"])

        x_name = str(task_df["x_name"].iloc[0]) if "x_name" in task_df.columns and len(
            task_df) > 0 else "x"
        y_name = str(task_df["y_name"].iloc[0]) if "y_name" in task_df.columns and len(
            task_df) > 0 else "y"

        if color_by_stage and "sample_stage" in task_df.columns:
            for stage, color in stage_colors.items():
                subset = task_df[task_df["sample_stage"] == stage]
                if len(subset) > 0:
                    ax.scatter(
                        subset["x"],
                        subset["y"],
                        s=20,
                        alpha=0.85,
                        linewidths=0,
                        c=color,
                        label=stage,
                    )
        else:
            ax.scatter(task_df["x"], task_df["y"],
                       s=20, alpha=0.75, linewidths=0)

        title = f"Task {task_id}"
        if "pair_id" in task_df.columns and len(task_df) > 0:
            title = f"Task {task_id} ({task_df['pair_id'].iloc[0]})"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(x_name, fontsize=8)
        ax.set_ylabel(y_name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)

    for j in range(len(task_ids), len(axes)):
        axes[j].axis("off")

    if color_by_stage and "sample_stage" in df.columns:
        # Build one global legend from first axes that has handles.
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8)

    fig.suptitle(
        f"Postprocessed Function-Learning Stimuli: {csv_path.name}", fontsize=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem + "_scatter_grid"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    args = parse_args()
    csv_paths = resolve_inputs(args.input_csv, args.input_glob)
    for csv_path in csv_paths:
        plot_one_csv(
            csv_path=csv_path,
            output_dir=args.output_dir,
            cols=args.cols,
            dpi=args.dpi,
            color_by_stage=args.color_by_stage,
        )


if __name__ == "__main__":
    main()
