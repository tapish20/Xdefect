"""Automated code refactoring diff patch generator.

Generates unified Git Diff (.patch) files demonstrating code refactoring
transformations for high-risk software modules.
"""

from __future__ import annotations

import difflib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCHES_DIR = PROJECT_ROOT / "outputs" / "recommendations" / "patches"


def generate_sample_patches() -> list[tuple[str, str]]:
    sample_patches = []

    # Patch 1: Extract Method Refactoring for Complex Parser
    orig_code_1 = """class ComplexParser:
    def parse_source(self, data):
        # Monolithic function with high cyclomatic complexity (WMC=70, LOC=361)
        results = []
        for line in data.split('\\n'):
            if line.startswith('DEF'):
                tokens = line.split()
                if len(tokens) > 2 and tokens[1].isupper():
                    if 'EXPR' in tokens[2]:
                        val = int(tokens[3]) * 42
                        results.append((tokens[1], val))
            elif line.startswith('VAL'):
                pass
        return results
"""

    refactored_code_1 = """class ComplexParser:
    def parse_source(self, data):
        # Refactored via Extract Method pattern
        results = []
        for line in data.split('\\n'):
            parsed_item = self._parse_single_line(line)
            if parsed_item:
                results.append(parsed_item)
        return results

    def _parse_single_line(self, line: str):
        if not line.startswith('DEF'):
            return None
        tokens = line.split()
        if len(tokens) <= 2 or not tokens[1].isupper():
            return None
        if 'EXPR' in tokens[2]:
            val = int(tokens[3]) * 42
            return (tokens[1], val)
        return None
"""

    diff_1 = "".join(
        difflib.unified_diff(
            orig_code_1.splitlines(keepends=True),
            refactored_code_1.splitlines(keepends=True),
            fromfile="a/src/core/parser.py",
            tofile="b/src/core/parser.py",
        )
    )
    sample_patches.append(("extract_method_parser.patch", diff_1))

    # Patch 2: Dependency Inversion for Coupled Client
    orig_code_2 = """class NetworkClient:
    def __init__(self):
        # High Coupling Between Objects (CBO=18)
        self.sql_db = DatabaseConnection("localhost", 5432)
        self.s3_storage = S3StorageClient("aws-region-1")
        self.logger = FileLogger("/var/log/app.log")

    def sync_data(self, payload):
        self.logger.log("Syncing payload")
        self.sql_db.execute_query(payload)
        self.s3_storage.upload_object(payload)
"""

    refactored_code_2 = """class NetworkClient:
    def __init__(self, db_service, storage_service, logger_service):
        # Refactored via Dependency Inversion Principle (DIP)
        self.db_service = db_service
        self.storage_service = storage_service
        self.logger_service = logger_service

    def sync_data(self, payload):
        self.logger_service.log("Syncing payload")
        self.db_service.execute(payload)
        self.storage_service.store(payload)
"""

    diff_2 = "".join(
        difflib.unified_diff(
            orig_code_2.splitlines(keepends=True),
            refactored_code_2.splitlines(keepends=True),
            fromfile="a/src/network/client.py",
            tofile="b/src/network/client.py",
        )
    )
    sample_patches.append(("dependency_inversion_client.patch", diff_2))

    return sample_patches


def main() -> int:
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating automated code refactoring Git diff patches...", flush=True)
    patches = generate_sample_patches()

    for filename, diff_content in patches:
        patch_path = PATCHES_DIR / filename
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(diff_content)
        print(f"Saved Git Diff Patch to {patch_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
