from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


SELECTED_TASK_IDS = [
    1, 14, 18, 19, 23, 42, 43, 44, 45, 56, 64, 66, 67, 68,
    73, 74, 75, 76, 78, 82, 83, 84, 93, 98, 100,
]  # 47, 69, 94, 95, 96,


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def fuzzy_match(a: str, b: str) -> bool:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    return a_norm in b_norm or b_norm in a_norm


def parse_pairmeta(pairmeta_path: Path) -> Dict[int, Dict[str, str]]:
    meta: Dict[int, Dict[str, str]] = {}
    with pairmeta_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            task_id = int(parts[0])
            meta[task_id] = {
                "cause_col_start": parts[1],
                "cause_col_end": parts[2],
                "effect_col_start": parts[3],
                "effect_col_end": parts[4],
                "weight": parts[5],
            }
    return meta


def parse_readme(readme_path: Path) -> Dict[int, Dict[str, str]]:
    readme_map: Dict[int, Dict[str, str]] = {}
    with readme_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("pair"):
                continue
            match = re.match(r"^pair(\d{4})\s+(.*)$", line.strip())
            if not match:
                continue
            task_id = int(match.group(1))
            rest = match.group(2)
            cols = re.split(r"\s{2,}", rest)
            if len(cols) < 2:
                continue
            readme_map[task_id] = {
                "var1": cols[0].strip(),
                "var2": cols[1].strip(),
            }
    return readme_map


def get_file_shape(pair_path: Path) -> Tuple[int, int]:
    n_rows = 0
    min_cols = None
    max_cols = 0
    with pair_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            n_rows += 1
            n_cols = len(stripped.split())
            max_cols = max(max_cols, n_cols)
            if min_cols is None or n_cols < min_cols:
                min_cols = n_cols
    if min_cols is None:
        return 0, 0
    return n_rows, max_cols


def parse_xy_rows(pair_path: Path) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    with pair_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                continue
            x_val = float(parts[0])
            y_val = float(parts[1])
            rows.append((x_val, y_val))
    return rows


def direction_hint(meta_row: Dict[str, str] | None) -> str:
    if not meta_row:
        return "unknown"
    if (
        meta_row["cause_col_start"] == "1"
        and meta_row["cause_col_end"] == "1"
        and meta_row["effect_col_start"] == "2"
        and meta_row["effect_col_end"] == "2"
    ):
        return "var1_to_var2"
    if (
        meta_row["cause_col_start"] == "2"
        and meta_row["cause_col_end"] == "2"
        and meta_row["effect_col_start"] == "1"
        and meta_row["effect_col_end"] == "1"
    ):
        return "var2_to_var1"
    return "unknown"


def parse_xy_from_description(desc_path: Path) -> Dict[str, str]:
    if not desc_path.exists():
        return {"x": "", "y": "", "description_file_exists": "False"}

    x_patterns = [
        re.compile(
            r"^\s*(?:first\s+column\s*\(x\)|x\s*\(first\s+column\)|x\s*\([^)]*\)|x)\s*[:=\-]\s*(.+)\s*$",
            re.IGNORECASE,
        ),
    ]
    y_patterns = [
        re.compile(
            r"^\s*(?:second\s+column\s*\(y\)|y\s*\(second\s+column\)|y\s*\([^)]*\)|y)\s*[:=\-]\s*(.+)\s*$",
            re.IGNORECASE,
        ),
    ]

    x_val = ""
    y_val = ""
    with desc_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if "ground truth" in lower:
                continue

            if not x_val:
                for pattern in x_patterns:
                    match = pattern.match(line)
                    if match:
                        candidate = match.group(1).strip().strip(".")
                        if candidate and ">" not in candidate and "<" not in candidate:
                            x_val = candidate
                        break

            if not y_val:
                for pattern in y_patterns:
                    match = pattern.match(line)
                    if match:
                        candidate = match.group(1).strip().strip(".")
                        if candidate and ">" not in candidate and "<" not in candidate:
                            y_val = candidate
                        break

            if x_val and y_val:
                break

    return {"x": x_val, "y": y_val, "description_file_exists": "True"}


def compute_name_match_status(description_name: str, readme_name: str) -> str:
    if not description_name or not readme_name:
        return "unavailable"
    if normalize_text(description_name) == normalize_text(readme_name):
        return "exact"
    if fuzzy_match(description_name, readme_name):
        return "fuzzy"
    return "mismatch"


def build_pairs_subset_output(output_csv: str, report_csv: str) -> None:
    script_dir = Path(__file__).resolve().parent
    pairs_dir = script_dir / "pairs"
    pairmeta_path = pairs_dir / "pairmeta.txt"
    readme_path = pairs_dir / "README"

    pairmeta = parse_pairmeta(pairmeta_path)
    readme_map = parse_readme(readme_path)

    consolidated_rows: List[Dict[str, str]] = []
    report_rows: List[Dict[str, str]] = []
    skipped_multicolumn: List[int] = []
    missing_description_xy: List[int] = []

    for task_id in SELECTED_TASK_IDS:
        pair_id = f"pair{task_id:04d}"
        pair_file = pairs_dir / f"{pair_id}.txt"
        pair_desc_file = pairs_dir / f"{pair_id}_des.txt"

        exists_in_pairs = pair_file.exists()
        exists_in_pairmeta = task_id in pairmeta
        exists_in_readme = task_id in readme_map
        meta_row = pairmeta.get(task_id)
        readme_row = readme_map.get(task_id, {"var1": "", "var2": ""})
        readme_var1 = readme_row["var1"]
        readme_var2 = readme_row["var2"]

        desc = parse_xy_from_description(pair_desc_file)
        desc_x = desc["x"]
        desc_y = desc["y"]
        desc_exists = desc["description_file_exists"]

        x_name = desc_x or readme_var1
        y_name = desc_y or readme_var2
        x_description = desc_x or readme_var1
        y_description = desc_y or readme_var2

        if not desc_x or not desc_y:
            missing_description_xy.append(task_id)

        file_rows = 0
        file_cols = 0
        notes: List[str] = []
        if exists_in_pairs:
            file_rows, file_cols = get_file_shape(pair_file)
            if file_cols > 2:
                notes.append("excluded_multicolumn")
                skipped_multicolumn.append(task_id)

        x_status = compute_name_match_status(x_name, readme_var1)
        y_status = compute_name_match_status(y_name, readme_var2)

        report_rows.append(
            {
                "task_id": str(task_id),
                "pair_id": pair_id,
                "exists_in_pairs": str(exists_in_pairs),
                "exists_in_pairmeta": str(exists_in_pairmeta),
                "exists_in_readme": str(exists_in_readme),
                "description_file_exists": desc_exists,
                "file_rows": str(file_rows),
                "file_columns": str(file_cols),
                "readme_var1": readme_var1,
                "readme_var2": readme_var2,
                "description_x": desc_x,
                "description_y": desc_y,
                "x_description_vs_readme_status": x_status,
                "y_description_vs_readme_status": y_status,
                "direction_hint": direction_hint(meta_row),
                "notes": ";".join(notes),
            }
        )

        if not exists_in_pairs or file_cols != 2:
            continue

        xy_rows = parse_xy_rows(pair_file)
        for idx, (x_val, y_val) in enumerate(xy_rows):
            consolidated_rows.append(
                {
                    "task_id": str(task_id),
                    "pair_id": pair_id,
                    "row_id": str(idx),
                    "x": str(x_val),
                    "y": str(y_val),
                    "x_name": x_name,
                    "y_name": y_name,
                    "x_description": x_description,
                    "y_description": y_description,
                    "source_file": str(pair_file),
                    "description_file": str(pair_desc_file),
                    "readme_var1": readme_var1,
                    "readme_var2": readme_var2,
                    "pairmeta_cause_col_start": meta_row["cause_col_start"] if meta_row else "",
                    "pairmeta_cause_col_end": meta_row["cause_col_end"] if meta_row else "",
                    "pairmeta_effect_col_start": meta_row["effect_col_start"] if meta_row else "",
                    "pairmeta_effect_col_end": meta_row["effect_col_end"] if meta_row else "",
                    "pairmeta_weight": meta_row["weight"] if meta_row else "",
                }
            )

    output_path = Path(output_csv)
    report_path = Path(report_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    consolidated_columns = [
        "task_id",
        "pair_id",
        "row_id",
        "x",
        "y",
        "x_name",
        "y_name",
        "x_description",
        "y_description",
        "source_file",
        "description_file",
        "readme_var1",
        "readme_var2",
        "pairmeta_cause_col_start",
        "pairmeta_cause_col_end",
        "pairmeta_effect_col_start",
        "pairmeta_effect_col_end",
        "pairmeta_weight",
    ]
    report_columns = [
        "task_id",
        "pair_id",
        "exists_in_pairs",
        "exists_in_pairmeta",
        "exists_in_readme",
        "description_file_exists",
        "file_rows",
        "file_columns",
        "readme_var1",
        "readme_var2",
        "description_x",
        "description_y",
        "x_description_vs_readme_status",
        "y_description_vs_readme_status",
        "direction_hint",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=consolidated_columns)
        writer.writeheader()
        writer.writerows(consolidated_rows)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report_columns)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Tasks requested: {len(SELECTED_TASK_IDS)}")
    print(
        f"Tasks skipped (more than 2 columns): {len(skipped_multicolumn)} -> {sorted(set(skipped_multicolumn))}")
    print(
        f"Tasks missing x/y in descriptions (used README fallback): {sorted(set(missing_description_xy))}")
    print(f"Rows exported: {len(consolidated_rows)}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    output_csv = root / "pairs_subset_xy_with_metadata.csv"
    report_csv = root / "pairs_subset_crosscheck_report.csv"
    build_pairs_subset_output(str(output_csv), str(report_csv))
