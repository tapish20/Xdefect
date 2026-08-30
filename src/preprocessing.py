"""Data cleaning and static/repository feature fusion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_LABEL_COL = "defects"
MODULE_ID_CANDIDATES = (
    "file_module_id",
    "classname",
    "class_name",
    "module_id",
    "module",
    "file",
    "filename",
    "path",
    "name",
)
REPO_METRIC_COLUMNS = (
    "code_churn",
    "commit_count",
    "num_developers",
    "bugfix_commit_count",
    "days_since_last_modified",
)


def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    return cleaned


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_actual = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def _normalize_binary_label(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "defective": 1,
            "non-defective": 0,
            "clean": 0,
            "buggy": 1,
            "1": 1,
            "0": 0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_binary = pd.Series(pd.NA, index=series.index, dtype="Int64")
    numeric_binary.loc[numeric.notna()] = numeric.loc[numeric.notna()].gt(0).astype(int)
    normalized = mapped.astype("Int64").fillna(numeric_binary)
    if normalized.isna().any():
        unknown = sorted(text[normalized.isna()].dropna().unique().tolist())
        raise ValueError(f"Could not normalize binary defect labels: {unknown}")
    return normalized.astype(int)


def _coerce_numeric_metrics(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if column in exclude:
            continue
        converted = pd.to_numeric(cleaned[column], errors="coerce")
        if converted.notna().sum() == cleaned[column].notna().sum():
            cleaned[column] = converted
    return cleaned


def _median_impute_static_metrics(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_columns = [
        column
        for column in cleaned.select_dtypes(include=[np.number]).columns
        if column not in exclude
    ]
    for column in numeric_columns:
        median = cleaned[column].median(skipna=True)
        if pd.isna(median):
            continue
        cleaned[column] = cleaned[column].fillna(median)
    return cleaned


def load_and_clean(path: str | Path, label_col: str = DEFAULT_LABEL_COL) -> pd.DataFrame:
    """Load a static metrics CSV and apply baseline cleaning.

    Cleaning includes whitespace stripping, binary label normalization, numeric
    coercion for metric columns, and median imputation for missing static
    metrics. Identifier-like columns are preserved as strings.
    """

    df = pd.read_csv(path)
    df = _strip_whitespace(df)

    actual_label_col = _find_column(df, [label_col, "defect", "bug", "bugs", "class", "c"])
    if actual_label_col is None:
        defect_like_columns = [
            column
            for column in df.columns
            if "defect" in str(column).lower() or "bug" in str(column).lower()
        ]
        actual_label_col = defect_like_columns[-1] if defect_like_columns else None
    if actual_label_col is None:
        raise ValueError(
            f"Label column '{label_col}' was not found and no common defect-label alias exists."
        )
    if actual_label_col != label_col:
        df = df.rename(columns={actual_label_col: label_col})

    module_id_col = detect_module_identifier(df)
    exclude = {label_col}
    if module_id_col:
        exclude.add(module_id_col)

    df = _coerce_numeric_metrics(df, exclude=exclude)
    df[label_col] = _normalize_binary_label(df[label_col])
    df = _median_impute_static_metrics(df, exclude=exclude)
    df.attrs["label_col"] = label_col
    df.attrs["module_id_col"] = module_id_col
    return df


def detect_module_identifier(df: pd.DataFrame) -> str | None:
    """Return the most likely module/file identifier column, if one exists."""

    return _find_column(df, MODULE_ID_CANDIDATES)


def _prepare_repo_df(repo_df: pd.DataFrame | None) -> pd.DataFrame:
    if repo_df is None or repo_df.empty:
        return pd.DataFrame(columns=("file_module_id", *REPO_METRIC_COLUMNS))
    cleaned = _strip_whitespace(repo_df)
    repo_id_col = detect_module_identifier(cleaned)
    if repo_id_col is None:
        return pd.DataFrame(columns=("file_module_id", *REPO_METRIC_COLUMNS))
    if repo_id_col != "file_module_id":
        cleaned = cleaned.rename(columns={repo_id_col: "file_module_id"})
    for column in REPO_METRIC_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned[["file_module_id", *REPO_METRIC_COLUMNS]].drop_duplicates("file_module_id")


def build_hybrid_features(static_df: pd.DataFrame, repo_df: pd.DataFrame | None) -> pd.DataFrame:
    """Join static metrics with repository-mined metrics without zero-filling.

    If repository data is unavailable or cannot be joined, repository metric
    columns are left as NaN and `repo_metrics_available` is False. Later hybrid
    experiments should exclude rows/datasets where that flag is False.
    """

    static = _strip_whitespace(static_df)
    static_id_col = detect_module_identifier(static)
    repo = _prepare_repo_df(repo_df)

    if static_id_col is None or repo.empty:
        hybrid = static.copy()
        for column in REPO_METRIC_COLUMNS:
            hybrid[column] = np.nan
        hybrid["repo_metrics_available"] = False
        hybrid["hybrid_join_status"] = (
            "missing_static_module_identifier"
            if static_id_col is None
            else "missing_repo_data"
        )
        hybrid.attrs["hybrid_match_rate"] = 0.0
        hybrid.attrs["hybrid_join_status"] = hybrid["hybrid_join_status"].iloc[0]
        return hybrid

    hybrid = static.merge(
        repo,
        how="left",
        left_on=static_id_col,
        right_on="file_module_id",
        suffixes=("", "_repo"),
    )
    matched = hybrid[list(REPO_METRIC_COLUMNS)].notna().any(axis=1)
    hybrid["repo_metrics_available"] = matched
    hybrid["hybrid_join_status"] = np.where(matched, "matched", "unmatched_repo_metrics")
    hybrid.attrs["hybrid_match_rate"] = float(matched.mean()) if len(hybrid) else 0.0
    hybrid.attrs["hybrid_join_status"] = "joined"
    return hybrid
