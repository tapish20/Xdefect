"""Mine Git repository history metrics with PyDriller.

The default mapping intentionally documents that the requested NASA MDP datasets
do not expose reliable public repository/file-path mappings. Add a repository
URL and a join-key strategy to DATASET_REPOSITORIES only when provenance is
strong enough to join mined metrics back to static rows.
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from git import Repo
from pydriller import Repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "data" / "static"
REPO_MINED_DIR = PROJECT_ROOT / "data" / "repo_mined"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUTS_DIR / "repo_mining_report.txt"

BUGFIX_PATTERN = re.compile(r"\b(fix|fixed|fixes|bug|defect|issue|fault|patch)\b", re.I)
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".java",
    ".py",
    ".js",
    ".ts",
    ".cs",
}


@dataclass(frozen=True)
class RepoMapping:
    dataset: str
    repository_url: str | None
    mapping_source_url: str
    source_evidence: str
    join_key_note: str
    labeling_date: str | None = None


DATASET_REPOSITORIES = {
    "cm1": RepoMapping(
        dataset="cm1",
        repository_url=None,
        mapping_source_url="https://zenodo.org/records/268434",
        source_evidence=(
            "OpenScience/Zenodo identify CM1 only as a NASA Metrics Data Program "
            "spacecraft-instrument defect dataset. The released ARFF rows are "
            "metric vectors/modules and do not include public file paths."
        ),
        join_key_note="No public Git repository and no module-to-file identifier were found.",
    ),
    "kc1": RepoMapping(
        dataset="kc1",
        repository_url=None,
        mapping_source_url="https://zenodo.org/records/268441",
        source_evidence=(
            "NASA MDP literature describes KC1 as storage management for ground data; "
            "the OpenScience KC1 record currently mirrors a separate class-level "
            "numeric-defect ARFF. Neither source provides a reliable public Git URL "
            "that can be joined to the requested static metrics."
        ),
        join_key_note="No reliable repository/file mapping; repository mining is disabled.",
    ),
    "pc1": RepoMapping(
        dataset="pc1",
        repository_url=None,
        mapping_source_url="https://zenodo.org/records/268456",
        source_evidence=(
            "OpenScience/Zenodo identify PC1 as a NASA Metrics Data Program dataset "
            "for flight software from an earth-orbiting satellite. The ARFF rows do "
            "not expose source file paths."
        ),
        join_key_note="No public Git repository and no module-to-file identifier were found.",
    ),
    "pc3": RepoMapping(
        dataset="pc3",
        repository_url=None,
        mapping_source_url="https://openscience.us/repo/defect/mccabehalsted/pc3.html",
        source_evidence=(
            "OpenScience/tera-PROMISE identify PC3 as a NASA Metrics Data Program "
            "Halstead/McCabe defect dataset. The official ARFF link provides module "
            "metrics but not source repository paths."
        ),
        join_key_note="No public Git repository and no module-to-file identifier were found.",
    ),
}


def clone_repo(repository_url: str, work_dir: Path) -> Path:
    destination = work_dir / "repo"
    Repo.clone_from(repository_url, destination)
    return destination


def mine_repository(repo_path: Path, labeling_date: str | None = None) -> pd.DataFrame:
    metrics = defaultdict(
        lambda: {
            "code_churn": 0,
            "commit_count": 0,
            "developers": set(),
            "bugfix_commit_count": 0,
            "last_modified": None,
        }
    )

    last_repo_commit_date: datetime | None = None
    for commit in Repository(str(repo_path)).traverse_commits():
        commit_date = commit.committer_date or commit.author_date
        if commit_date and (last_repo_commit_date is None or commit_date > last_repo_commit_date):
            last_repo_commit_date = commit_date
        is_bugfix = bool(BUGFIX_PATTERN.search(commit.msg or ""))
        author = getattr(commit.author, "email", None) or getattr(commit.author, "name", "unknown")

        for modified_file in commit.modified_files:
            path = modified_file.new_path or modified_file.old_path
            if not path or Path(path).suffix.lower() not in SOURCE_SUFFIXES:
                continue
            item = metrics[path]
            item["code_churn"] += int(modified_file.added_lines or 0) + int(
                modified_file.deleted_lines or 0
            )
            item["commit_count"] += 1
            item["developers"].add(author)
            item["bugfix_commit_count"] += int(is_bugfix)
            if commit_date and (
                item["last_modified"] is None or commit_date > item["last_modified"]
            ):
                item["last_modified"] = commit_date

    if labeling_date:
        reference_date = datetime.fromisoformat(labeling_date).replace(tzinfo=timezone.utc)
    else:
        reference_date = last_repo_commit_date

    rows = []
    for file_path, item in sorted(metrics.items()):
        last_modified = item["last_modified"]
        days_since = None
        if reference_date and last_modified:
            days_since = max((reference_date - last_modified).days, 0)
        rows.append(
            {
                "file_module_id": file_path,
                "code_churn": item["code_churn"],
                "commit_count": item["commit_count"],
                "num_developers": len(item["developers"]),
                "bugfix_commit_count": item["bugfix_commit_count"],
                "days_since_last_modified": days_since,
                "last_modified": last_modified.isoformat() if last_modified else "",
            }
        )
    return pd.DataFrame(rows)


def detect_static_join_column(static_df: pd.DataFrame) -> str | None:
    candidates = ["file", "filename", "path", "module", "module_id", "name"]
    lower_to_actual = {column.lower(): column for column in static_df.columns}
    for candidate in candidates:
        if candidate in lower_to_actual:
            return lower_to_actual[candidate]
    return None


def write_empty_metrics(dataset: str) -> None:
    path = REPO_MINED_DIR / f"{dataset}_repo_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_module_id",
                "code_churn",
                "commit_count",
                "num_developers",
                "bugfix_commit_count",
                "days_since_last_modified",
                "last_modified",
            ],
        )
        writer.writeheader()


def process_dataset(mapping: RepoMapping) -> str:
    static_path = STATIC_DIR / f"{mapping.dataset}.csv"
    static_df = pd.read_csv(static_path) if static_path.exists() else pd.DataFrame()
    static_rows = len(static_df)
    join_column = detect_static_join_column(static_df) if not static_df.empty else None

    lines = [
        f"## {mapping.dataset.upper()}",
        f"static_rows: {static_rows}",
        f"repository_url: {mapping.repository_url or 'not_available'}",
        f"mapping_source_url: {mapping.mapping_source_url}",
        f"source_evidence: {mapping.source_evidence}",
        f"join_key_note: {mapping.join_key_note}",
    ]

    if mapping.repository_url is None:
        write_empty_metrics(mapping.dataset)
        lines.extend(["mining_status: skipped_no_reliable_repository", "match_rate: 0.00%"])
        return "\n".join(lines)

    if join_column is None:
        write_empty_metrics(mapping.dataset)
        lines.extend(
            [
                "mining_status: skipped_no_static_join_column",
                "match_rate: 0.00%",
            ]
        )
        return "\n".join(lines)

    with tempfile.TemporaryDirectory(prefix=f"{mapping.dataset}_repo_") as tmp:
        tmp_path = Path(tmp)
        repo_path = clone_repo(mapping.repository_url, tmp_path)
        mined_df = mine_repository(repo_path, mapping.labeling_date)
        output_path = REPO_MINED_DIR / f"{mapping.dataset}_repo_metrics.csv"
        mined_df.to_csv(output_path, index=False)
        shutil.rmtree(repo_path, ignore_errors=True)

    static_keys = static_df[join_column].astype(str).str.replace("\\", "/", regex=False)
    mined_keys = set(mined_df["file_module_id"].astype(str))
    matched = static_keys.isin(mined_keys).sum()
    match_rate = matched / static_rows if static_rows else 0
    lines.extend(
        [
            "mining_status: mined",
            f"join_column: {join_column}",
            f"repo_metric_rows: {len(mined_df)}",
            f"matched_static_rows: {matched}",
            f"match_rate: {match_rate:.2%}",
        ]
    )
    return "\n".join(lines)


def main(dataset_names: list[str]) -> int:
    REPO_MINED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    names = [name.lower() for name in (dataset_names or DATASET_REPOSITORIES.keys())]
    sections = [
        "Repository Mining Report",
        "Generated by src/mine_repository.py",
        "",
    ]
    for name in names:
        sections.append(process_dataset(DATASET_REPOSITORIES[name]))
        sections.append("")

    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
