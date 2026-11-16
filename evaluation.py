"""Validated single-label annotation evaluation, independent of the UI."""

from io import BytesIO
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


class ValidationError(ValueError):
    """An actionable problem with an input dataset."""


def read_table(data: bytes, name: str) -> pd.DataFrame:
    """Preserve string IDs (including CSV leading zeros) and literal labels like NA."""
    suffix = Path(name).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(BytesIO(data), dtype=str, keep_default_na=False)
        elif suffix == ".xlsx":
            frame = pd.read_excel(BytesIO(data), dtype=str, keep_default_na=False)
        elif suffix == ".json":
            records = json.loads(data.decode("utf-8-sig"))
            if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
                raise ValidationError("JSON must contain an array of row objects.")
            frame = pd.DataFrame(records)
        else:
            raise ValidationError("Use a CSV, XLSX, or JSON file.")
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Could not read {name}. Check the file format and encoding.") from exc
    if frame.empty or not len(frame.columns):
        raise ValidationError(f"{name} contains no data rows.")
    return frame


def prepare(frame: pd.DataFrame, mapping: dict, title: str, ignore_case=False) -> pd.DataFrame:
    if len(set(mapping.values())) != len(mapping):
        raise ValidationError(f"{title}: select a different source column for each field.")
    if not set(mapping.values()).issubset(frame.columns):
        raise ValidationError(f"{title}: a selected column is missing.")
    result = frame[list(mapping.values())].copy()
    result.columns = list(mapping)
    for col in result:
        if result[col].map(lambda value: isinstance(value, (list, dict))).any():
            raise ValidationError(f"{title}: {col} must contain a single value per row.")
        result[col] = result[col].fillna("").astype(str).str.strip()
        if result[col].eq("").any():
            raise ValidationError(f"{title}: {col} has blank values. Fill them or remove those rows.")
    if ignore_case:
        result["label"] = result["label"].str.casefold()
    keys = ["annotator", "item_id"] if "annotator" in result else ["item_id"]
    duplicated = result.duplicated(keys, keep=False)
    if duplicated.any():
        raise ValidationError(
            f"{title}: {int(duplicated.sum())} rows have duplicate "
            f"{'annotator / item ID pairs' if 'annotator' in result else 'item IDs'}. "
            "Keep exactly one label for each."
        )
    return result


def evaluate(truth: pd.DataFrame, annotations: pd.DataFrame) -> tuple:
    """Inputs are canonical frames from prepare(). All truth items are expected per annotator.

    Unknown IDs are unscored. Missing items count against accuracy, recall and F1;
    they produce no positive prediction. Macro metrics include every truth label
    plus any additional predicted labels on known items. Undefined metrics are 0.
    """
    if truth.empty or annotations.empty:
        raise ValidationError("Both ground truth and annotations need at least one row.")
    summaries, details, label_reports = [], [], []
    gold = truth.rename(columns={"label": "expected"})
    for annotator, group in annotations.groupby("annotator", sort=True):
        merged = gold.merge(group[["item_id", "label"]].rename(columns={"label": "actual"}),
                            on="item_id", how="outer", indicator=True, validate="one_to_one")
        merged["annotator"] = annotator
        merged["status"] = "incorrect"
        merged.loc[merged["expected"].eq(merged["actual"]), "status"] = "correct"
        merged.loc[merged["_merge"].eq("left_only"), "status"] = "missing"
        merged.loc[merged["_merge"].eq("right_only"), "status"] = "unknown ID"
        known = merged[merged["_merge"].ne("right_only")]
        labels = sorted(set(known["expected"]) | set(known["actual"].dropna()))
        # Unique sentinel ensures a real uploaded label never becomes an abstention.
        missing = "__MISSING__"
        while missing in labels:
            missing += "_"
        precision, recall, f1, support = precision_recall_fscore_support(
            known["expected"], known["actual"].fillna(missing), labels=labels, zero_division=0
        )
        correct = int(known["status"].eq("correct").sum())
        submitted = int(known["actual"].notna().sum())
        summaries.append({
            "annotator": annotator, "accuracy": correct / len(truth),
            "precision": float(precision.mean()), "recall": float(recall.mean()),
            "f1": float(f1.mean()), "coverage": submitted / len(truth),
            "correct": correct, "incorrect": int(known["status"].eq("incorrect").sum()),
            "missing": len(truth) - submitted,
            "unknown_ids": int(merged["status"].eq("unknown ID").sum()),
            "expected_items": len(truth),
        })
        label_reports.extend({"annotator": annotator, "label": label, "precision": float(p),
                              "recall": float(r), "f1": float(f), "support": int(s)}
                             for label, p, r, f, s in zip(labels, precision, recall, f1, support))
        details.append(merged[["annotator", "item_id", "expected", "actual", "status"]])
    return (pd.DataFrame(summaries).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True),
            pd.concat(details, ignore_index=True), pd.DataFrame(label_reports))


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Keep uploaded text from being interpreted as spreadsheet formulas in reports."""
    safe = frame.copy()
    for col in safe.select_dtypes(include=["object", "string"]):
        safe[col] = safe[col].map(
            lambda value: "'" + value if isinstance(value, str)
            and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else value
        )
    return safe.to_csv(index=False).encode("utf-8-sig")
