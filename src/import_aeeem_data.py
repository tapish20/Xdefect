"""Import AEEEM/D'Ambros hybrid defect-prediction datasets.

The AEEEM archives contain class-level static source-code metrics, historical
change/process metrics, and post-release defect counts. This importer creates
normalized project CSVs that fit the static/repo/hybrid layout used by the rest
of this project.
"""

from __future__ import annotations

import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "data" / "static"
REPO_MINED_DIR = PROJECT_ROOT / "data" / "repo_mined"
HYBRID_DIR = PROJECT_ROOT / "data" / "hybrid"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "aeeem_dataset_summary.txt"

AEEEM_PROJECTS = ("eclipse", "equinox", "lucene", "mylyn", "pde")
STATIC_FILE = "single-version-ck-oo.csv"
CHANGE_FILE = "change-metrics.csv"
BUG_FILE = "bug-metrics.csv"
COMPLEXITY_CHANGE_FILE = "complexity-code-change.csv"


@dataclass(frozen=True)
class ImportedDataset:
    name: str
    static_rows: int
    repo_rows: int
    hybrid_rows: int
    matched_rows: int
    defective_rows: int
    static_columns: int
    repo_columns: int

    @property
    def match_rate(self) -> float:
        return self.matched_rows / self.static_rows if self.static_rows else 0.0


def read_semicolon_csv_from_zip(archive_path: Path, suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if not matches:
            raise FileNotFoundError(f"{suffix} not found in {archive_path}")
        with archive.open(matches[0]) as handle:
            df = pd.read_csv(handle, sep=";", engine="python")
    return clean_columns(df)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip().rstrip(":").strip() for col in cleaned.columns]
    cleaned = cleaned.drop(columns=[col for col in cleaned.columns if not col], errors="ignore")
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    for column in cleaned.columns:
        if column == "classname":
            continue
        converted = pd.to_numeric(cleaned[column], errors="coerce")
        if converted.notna().sum() == cleaned[column].notna().sum():
            cleaned[column] = converted
    return cleaned


def normalize_static(static_df: pd.DataFrame) -> pd.DataFrame:
    static = static_df.copy()
    if "classname" not in static.columns:
        raise ValueError("AEEEM static metrics must include classname.")
    if "bugs" not in static.columns:
        raise ValueError("AEEEM static metrics must include bugs defect counts.")
    static.insert(0, "dataset_family", "aeeem")
    static["defects"] = pd.to_numeric(static["bugs"], errors="coerce").fillna(0).gt(0).astype(int)
    return static


def normalize_repo_metrics(change_df: pd.DataFrame, bug_df: pd.DataFrame, entropy_df: pd.DataFrame) -> pd.DataFrame:
    for name, df in {
        "change": change_df,
        "bug": bug_df,
        "entropy": entropy_df,
    }.items():
        if "classname" not in df.columns:
            raise ValueError(f"AEEEM {name} metrics must include classname.")

    change = change_df.rename(
        columns={
            "classname": "file_module_id",
            "codeChurnUntil": "code_churn",
            "numberOfVersionsUntil": "commit_count",
            "numberOfAuthorsUntil": "num_developers",
            "numberOfFixesUntil": "bugfix_commit_count",
            "ageWithRespectTo": "days_since_last_modified",
            "linesAddedUntil": "lines_added",
            "linesRemovedUntil": "lines_deleted",
        }
    )
    repo_columns = [
        "file_module_id",
        "code_churn",
        "commit_count",
        "num_developers",
        "bugfix_commit_count",
        "days_since_last_modified",
        "lines_added",
        "lines_deleted",
        "numberOfRefactoringsUntil",
        "maxLinesAddedUntil",
        "avgLinesAddedUntil",
        "maxLinesRemovedUntil",
        "avgLinesRemovedUntil",
        "maxCodeChurnUntil",
        "avgCodeChurnUntil",
        "weightedAgeWithRespectTo",
    ]
    repo = change[[col for col in repo_columns if col in change.columns]].copy()

    bug_history = bug_df.rename(columns={"classname": "file_module_id"})
    bug_history = bug_history[
        [
            col
            for col in bug_history.columns
            if col == "file_module_id" or col.endswith("FoundUntil")
        ]
    ]
    entropy = entropy_df.rename(columns={"classname": "file_module_id"})
    repo = repo.merge(bug_history, on="file_module_id", how="left")
    repo = repo.merge(entropy, on="file_module_id", how="left")
    repo.insert(0, "dataset_family", "aeeem")
    return repo.drop_duplicates("file_module_id")


def import_archive(project: str, archive_path: Path) -> ImportedDataset:
    static_raw = read_semicolon_csv_from_zip(archive_path, STATIC_FILE)
    change_raw = read_semicolon_csv_from_zip(archive_path, CHANGE_FILE)
    bug_raw = read_semicolon_csv_from_zip(archive_path, BUG_FILE)
    entropy_raw = read_semicolon_csv_from_zip(archive_path, COMPLEXITY_CHANGE_FILE)

    static = normalize_static(static_raw)
    repo = normalize_repo_metrics(change_raw, bug_raw, entropy_raw)

    dataset_name = f"aeeem_{project}"
    static_path = STATIC_DIR / f"{dataset_name}.csv"
    repo_path = REPO_MINED_DIR / f"{dataset_name}_repo_metrics.csv"
    hybrid_path = HYBRID_DIR / f"{dataset_name}_hybrid.csv"

    static.to_csv(static_path, index=False)
    repo.to_csv(repo_path, index=False)

    hybrid = static.merge(repo, how="left", left_on="classname", right_on="file_module_id")
    metric_columns = [
        "code_churn",
        "commit_count",
        "num_developers",
        "bugfix_commit_count",
        "days_since_last_modified",
    ]
    hybrid["repo_metrics_available"] = hybrid[metric_columns].notna().any(axis=1)
    hybrid.to_csv(hybrid_path, index=False)

    return ImportedDataset(
        name=dataset_name,
        static_rows=len(static),
        repo_rows=len(repo),
        hybrid_rows=len(hybrid),
        matched_rows=int(hybrid["repo_metrics_available"].sum()),
        defective_rows=int(static["defects"].sum()),
        static_columns=len(static.columns),
        repo_columns=len(repo.columns),
    )


def write_summary(results: list[ImportedDataset], archives: dict[str, Path]) -> None:
    lines = [
        "AEEEM Hybrid Dataset Summary",
        "Source: D'Ambros, Lanza, and Robbes bug prediction dataset archives provided locally by the user.",
        "",
    ]
    for result in results:
        project = result.name.replace("aeeem_", "")
        lines.extend(
            [
                f"## {result.name}",
                f"local_archive: {archives[project]}",
                f"static_shape: {result.static_rows} rows x {result.static_columns} columns",
                f"repo_metrics_shape: {result.repo_rows} rows x {result.repo_columns} columns",
                f"hybrid_rows: {result.hybrid_rows}",
                f"defective_rows: {result.defective_rows} ({result.defective_rows / result.static_rows:.2%})",
                f"repo_match_rate: {result.match_rate:.2%}",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    default_dir = Path(r"C:\Users\agarw\Downloads")
    archives = {
        project: default_dir / f"{project}.zip"
        for project in AEEEM_PROJECTS
    }
    for arg in argv:
        path = Path(arg)
        archives[path.stem.lower()] = path

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    REPO_MINED_DIR.mkdir(parents=True, exist_ok=True)
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for project in AEEEM_PROJECTS:
        archive = archives[project]
        if not archive.exists():
            raise FileNotFoundError(archive)
        results.append(import_archive(project, archive))

    write_summary(results, archives)
    print(SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
