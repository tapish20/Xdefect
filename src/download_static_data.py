"""Download and summarize PROMISE/NASA static defect datasets.

The script prefers official tera-PROMISE/OpenScience/Zenodo sources and writes
CSV copies into data/static. If a live source is unavailable, it records that
status in outputs/dataset_summary.txt instead of silently substituting data.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "data" / "static"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "dataset_summary.txt"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    source_url: str
    source_note: str
    filename_hint: str


DATASETS = {
    "cm1": DatasetSource(
        name="cm1",
        source_url="https://zenodo.org/api/records/268434",
        source_note="Zenodo mirror of the NASA MDP CM1 dataset; OpenScience describes CM1 as a NASA Metrics Data Program defect dataset.",
        filename_hint="cm1",
    ),
    "kc1": DatasetSource(
        name="kc1",
        source_url="https://zenodo.org/api/records/268441",
        source_note="Zenodo/OpenScience KC1 record. Note: this mirror is class-level numeric-defect ARFF, not the older 22-column NASA MDP KC1 table used in some studies.",
        filename_hint="kc1",
    ),
    "pc1": DatasetSource(
        name="pc1",
        source_url="https://zenodo.org/api/records/268456",
        source_note="Zenodo mirror of the NASA MDP PC1 dataset; OpenScience describes PC1 as a NASA Metrics Data Program defect dataset.",
        filename_hint="pc1",
    ),
    "pc3": DatasetSource(
        name="pc3",
        source_url="https://terapromise.csc.ncsu.edu/repo/defect/mccabehalsted/pc/pc3/pc3.arff",
        source_note="Official OpenScience/tera-PROMISE link for the latest PC3 ARFF file.",
        filename_hint="pc3",
    ),
}


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "xcpdp-data-setup/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def zenodo_file_url(record_api_url: str, filename_hint: str) -> str:
    record = json.loads(fetch_bytes(record_api_url).decode("utf-8"))
    files = record.get("files", [])
    if not files:
        raise RuntimeError(f"No files listed in Zenodo record: {record_api_url}")
    for file_info in files:
        key = file_info.get("key", "").lower()
        if filename_hint.lower() in key:
            links = file_info.get("links", {})
            return links.get("self") or links.get("download")
    links = files[0].get("links", {})
    return links.get("self") or links.get("download")


def relation_name(arff_text: str) -> str | None:
    match = re.search(r"(?im)^\s*@relation\s+(.+?)\s*$", arff_text)
    if not match:
        return None
    return match.group(1).strip().strip("'\"")


def parse_arff_text(arff_text: str) -> pd.DataFrame:
    attributes: list[str] = []
    data_lines: list[str] = []
    in_data = False

    for raw_line in arff_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        lower = line.lower()
        if not in_data:
            if lower.startswith("@attribute"):
                rest = line[len("@attribute") :].strip()
                if rest.startswith(("'", '"')):
                    quote = rest[0]
                    end = rest.find(quote, 1)
                    name = rest[1:end]
                else:
                    name = rest.split(None, 1)[0]
                attributes.append(name)
            elif lower.startswith("@data"):
                in_data = True
            continue
        data_lines.append(raw_line)

    if not attributes:
        raise RuntimeError("No ARFF attributes found.")
    if not data_lines:
        raise RuntimeError("No ARFF data rows found.")

    reader = csv.reader(io.StringIO("\n".join(data_lines)))
    rows = [row for row in reader if row]
    df = pd.DataFrame(rows, columns=attributes)
    df = df.replace({"?": pd.NA})
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() == df[column].notna().sum():
            df[column] = converted
    return df


def extract_arff_from_payload(payload: bytes, filename_hint: str) -> tuple[str, str]:
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = zf.namelist()
            candidates = [
                name
                for name in names
                if filename_hint.lower() in Path(name).name.lower()
                and (name.lower().endswith(".arff") or ".arff" in name.lower())
            ]
            if not candidates:
                candidates = [name for name in names if ".arff" in name.lower()]
            if not candidates:
                raise RuntimeError("Zip payload did not contain an ARFF file.")
            selected = candidates[0]
            return selected, zf.read(selected).decode("utf-8", errors="replace")
    return f"{filename_hint}.arff", payload.decode("utf-8", errors="replace")


def normalized_defect_series(df: pd.DataFrame) -> pd.Series | None:
    candidates = [
        col
        for col in df.columns
        if col.lower() in {"defects", "defect", "bug", "bugs", "class", "c"}
        or "defect" in col.lower()
    ]
    if not candidates:
        return None
    series = df[candidates[-1]]
    text = series.astype(str).str.strip().str.lower()
    return text.map(
        {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "1": 1,
            "0": 0,
        }
    ).fillna(pd.to_numeric(series, errors="coerce").gt(0).astype(int))


def summarize_dataset(name: str, df: pd.DataFrame, source: DatasetSource, raw_name: str) -> str:
    defect_series = normalized_defect_series(df)
    lines = [
        f"## {name.upper()}",
        f"source: {source.source_url}",
        f"source_note: {source.source_note}",
        f"raw_file: {raw_name}",
        f"shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"columns: {', '.join(map(str, df.columns))}",
    ]
    if defect_series is None:
        lines.append("class_balance: no defect label column detected")
    else:
        counts = defect_series.value_counts(dropna=False).sort_index()
        total = len(defect_series)
        parts = []
        for label, count in counts.items():
            parts.append(f"{label}={int(count)} ({count / total:.2%})")
        lines.append(f"class_balance: {', '.join(parts)}")
    return "\n".join(lines)


def download_one(source: DatasetSource) -> tuple[str, pd.DataFrame | None, str | None]:
    try:
        download_url = (
            zenodo_file_url(source.source_url, source.filename_hint)
            if "zenodo.org/api/records" in source.source_url
            else source.source_url
        )
        payload = fetch_bytes(download_url)
        raw_name, arff_text = extract_arff_from_payload(payload, source.filename_hint)
        df = parse_arff_text(arff_text)
        csv_path = STATIC_DIR / f"{source.name}.csv"
        df.to_csv(csv_path, index=False)
        return raw_name, df, None
    except (urllib.error.URLError, TimeoutError, RuntimeError, zipfile.BadZipFile) as exc:
        return "", None, f"{type(exc).__name__}: {exc}"


def main(dataset_names: Iterable[str] = DATASETS.keys()) -> int:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    report_sections = [
        "PROMISE/NASA Static Dataset Summary",
        "Generated by src/download_static_data.py",
        "",
    ]
    exit_code = 0
    for name in dataset_names:
        source = DATASETS[name.lower()]
        raw_name, df, error = download_one(source)
        if df is None:
            exit_code = 1
            report_sections.append(
                "\n".join(
                    [
                        f"## {name.upper()}",
                        f"source: {source.source_url}",
                        f"source_note: {source.source_note}",
                        f"status: download_or_parse_failed",
                        f"error: {error}",
                    ]
                )
            )
        else:
            report_sections.append(summarize_dataset(name, df, source, raw_name))
        report_sections.append("")

    SUMMARY_PATH.write_text("\n".join(report_sections), encoding="utf-8")
    print(SUMMARY_PATH)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or DATASETS.keys()))
