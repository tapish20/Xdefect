"""Inspect a compressed git log for repository-mining joinability.

This is intentionally conservative: a plain `git log` with commit messages is
not enough for file-level hybrid features. We need file paths plus line-change
statistics, such as output from `git log --numstat`.
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "git_log_inspection.txt"

COMMIT_RE = re.compile(r"^commit\s+[0-9a-f]{7,40}\b")
AUTHOR_RE = re.compile(r"^Author:\s+(.+)$")
DATE_RE = re.compile(r"^Date:\s+(.+)$")
NUMSTAT_RE = re.compile(r"^(\d+|-)\s+(\d+|-)\s+\S+")
NAME_STATUS_RE = re.compile(r"^[AMDRCT]\s+\S+")
DIFF_RE = re.compile(r"^diff --git\s+")
BUGFIX_RE = re.compile(r"\b(fix|fixed|fixes|bug|defect|issue|fault|patch)\b", re.I)


def inspect_git_log(path: Path) -> dict[str, int]:
    counts = {
        "lines": 0,
        "commits": 0,
        "authors": 0,
        "dates": 0,
        "numstat_lines": 0,
        "name_status_lines": 0,
        "diff_lines": 0,
        "bugfix_keyword_lines": 0,
    }

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            counts["lines"] += 1
            text = line.rstrip("\n")
            counts["commits"] += int(bool(COMMIT_RE.match(text)))
            counts["authors"] += int(bool(AUTHOR_RE.match(text)))
            counts["dates"] += int(bool(DATE_RE.match(text)))
            counts["numstat_lines"] += int(bool(NUMSTAT_RE.match(text)))
            counts["name_status_lines"] += int(bool(NAME_STATUS_RE.match(text)))
            counts["diff_lines"] += int(bool(DIFF_RE.match(text)))
            counts["bugfix_keyword_lines"] += int(bool(BUGFIX_RE.search(text)))
    return counts


def write_report(path: Path, counts: dict[str, int]) -> None:
    file_level_available = any(
        counts[key] > 0 for key in ("numstat_lines", "name_status_lines", "diff_lines")
    )
    lines = [
        "Git Log Inspection Report",
        f"source_file: {path}",
        "",
        f"lines: {counts['lines']}",
        f"commits: {counts['commits']}",
        f"authors: {counts['authors']}",
        f"dates: {counts['dates']}",
        f"bugfix_keyword_lines: {counts['bugfix_keyword_lines']}",
        f"numstat_lines: {counts['numstat_lines']}",
        f"name_status_lines: {counts['name_status_lines']}",
        f"diff_lines: {counts['diff_lines']}",
        "",
        f"file_level_history_available: {file_level_available}",
    ]
    if file_level_available:
        lines.append(
            "decision: This log may support repository metrics if its file paths can be "
            "joined to static metric module identifiers."
        )
    else:
        lines.append(
            "decision: This log is commit-message level only. It cannot produce per-file "
            "code_churn, commit_count, num_developers, bugfix_commit_count, or "
            "days_since_last_modified."
        )
        lines.append(
            "needed_format: git log --all --numstat --date=iso --pretty=format:'commit %H%nAuthor: %an <%ae>%nDate: %ad%n%s%n%b'"
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("Usage: python src/inspect_git_log.py path/to/git_log.txt.gz")
    path = Path(argv[0])
    counts = inspect_git_log(path)
    write_report(path, counts)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
