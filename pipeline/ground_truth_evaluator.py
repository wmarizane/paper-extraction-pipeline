#!/usr/bin/env python3
"""Compare standardized extraction rows with PolyCrit workbook rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import openpyxl

try:
    from pipeline.standardized_csv_exporter import (
        POLYCRIT_FIELDNAMES,
        export_folder_to_csv,
    )
    from pipeline.standardizer import (
        _normalize_polymer_name,
        canonicalize_solvent_list,
    )
except ImportError:
    from standardized_csv_exporter import (  # type: ignore
        POLYCRIT_FIELDNAMES,
        export_folder_to_csv,
    )
    from standardizer import (  # type: ignore
        _normalize_polymer_name,
        canonicalize_solvent_list,
    )


REFERENCE_FIELD = "Reference"
EVALUATED_FIELDS = tuple(
    field for field in POLYCRIT_FIELDNAMES if field != REFERENCE_FIELD
)
DEFAULT_MATCH_THRESHOLD = 0.35

MATCH_FIELD_WEIGHTS = {
    "Polymer": 3.0,
    "Alternate Polymer Names": 1.0,
    "Solvents": 3.0,
    "Solvent Ratio (wt%)": 2.0,
    "Solvent Ratio (vol%)": 2.0,
    "Stationary Phase": 2.0,
    "Base Material": 1.0,
    "Phase": 1.0,
    "Pore size (Å)": 1.0,
    "Temperature (Celsius)": 1.0,
    "Flow Rate (mL/min)": 1.0,
    "Detector": 1.0,
}

NUMERIC_FIELD_TOLERANCES = {
    "Solvent Ratio (wt%)": 0.05,
    "Solvent Ratio (vol%)": 0.05,
    "Particle diameter (μm)": 0.05,
    "Pore size (Å)": 0.5,
    "Temperature (Celsius)": 0.1,
    "Flow Rate (mL/min)": 0.01,
    "Injected Polymer Concentration (g/L)": 0.01,
}


@dataclass(frozen=True)
class PolyCritRow:
    """One physical condition row and all reference memberships it declares."""

    row_id: str
    values: Mapping[str, Any]
    references: Tuple[str, ...]
    source: str = ""


@dataclass(frozen=True)
class MatchedPair:
    """One accepted condition pairing within a single reference."""

    reference: str
    ground_truth: PolyCritRow
    extracted: PolyCritRow
    score: float


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_clean(item) for item in value if _clean(item))
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if text.casefold() in {"", "none", "null", "nan"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _text_key(value: Any) -> str:
    text = _clean(value).casefold()
    text = text.replace("μ", "u").replace("å", "a")
    return re.sub(r"[^a-z0-9]+", "", text)


def _display_value(value: Any) -> str:
    return _clean(value)


def parse_references(value: Any) -> Tuple[str, ...]:
    """Return stable tokens while preserving composite numeric references."""

    if value is None:
        return ()
    if isinstance(value, bool):
        return (_text_key(value),)
    if isinstance(value, int):
        return (str(value),)
    if isinstance(value, float) and value.is_integer():
        return (str(int(value)),)

    text = _clean(value).replace(".pdf", "")
    if not text:
        return ()

    bracketed = re.match(r"^\s*\[(\d+)\]", text)
    if bracketed:
        return (bracketed.group(1),)

    parts = [
        part.strip()
        for part in re.split(r"\s*(?:,|;|/|\band\b)\s*", text, flags=re.IGNORECASE)
        if part.strip()
    ]
    numeric: List[str] = []
    all_numeric = bool(parts)
    for part in parts:
        match = re.fullmatch(r"\[?\s*(\d+)(?:\.0+)?\s*\]?", part)
        if not match:
            all_numeric = False
            break
        numeric.append(match.group(1))

    if all_numeric:
        return tuple(dict.fromkeys(numeric))

    fallback = _text_key(text)
    return (fallback,) if fallback else ()


def _row_has_content(values: Mapping[str, Any]) -> bool:
    return any(_clean(values.get(field)) for field in POLYCRIT_FIELDNAMES)


def load_polycrit_workbook(path: Path) -> List[PolyCritRow]:
    """Load every worksheet whose first row contains the PolyCrit schema."""

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows: List[PolyCritRow] = []
    eligible_sheets = 0
    try:
        for worksheet in workbook.worksheets:
            headers = [
                _clean(worksheet.cell(1, column).value)
                for column in range(1, worksheet.max_column + 1)
            ]
            index = {
                header: position
                for position, header in enumerate(headers)
                if header
            }
            if not all(field in index for field in POLYCRIT_FIELDNAMES):
                continue

            eligible_sheets += 1
            for row_number, raw_row in enumerate(
                worksheet.iter_rows(min_row=2, values_only=True),
                start=2,
            ):
                values = {
                    field: raw_row[index[field]]
                    if index[field] < len(raw_row)
                    else None
                    for field in POLYCRIT_FIELDNAMES
                }
                if not _row_has_content(values):
                    continue
                rows.append(
                    PolyCritRow(
                        row_id=f"xlsx:{worksheet.title}:{row_number}",
                        values=values,
                        references=parse_references(values.get(REFERENCE_FIELD)),
                        source=f"{path}:{worksheet.title}",
                    )
                )
    finally:
        workbook.close()

    if not eligible_sheets:
        raise ValueError(
            f"No worksheet in {path} contains all PolyCrit columns."
        )
    return rows


def load_polycrit_csv(
    path: Path,
    source_label: Optional[str] = None,
) -> List[PolyCritRow]:
    """Load a CSV produced by standardized_csv_exporter."""

    rows: List[PolyCritRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        missing = [field for field in POLYCRIT_FIELDNAMES if field not in headers]
        if missing:
            raise ValueError(
                f"CSV {path} is missing PolyCrit columns: {', '.join(missing)}"
            )

        for line_number, raw_row in enumerate(reader, start=2):
            values = {field: raw_row.get(field) for field in POLYCRIT_FIELDNAMES}
            if not _row_has_content(values):
                continue
            rows.append(
                PolyCritRow(
                    row_id=f"csv:{line_number}",
                    values=values,
                    references=parse_references(values.get(REFERENCE_FIELD)),
                    source=source_label or str(path),
                )
            )
    return rows


def load_extracted_rows(path: Path) -> List[PolyCritRow]:
    """Load an exported PolyCrit CSV or export a standardized directory."""

    if path.is_file() and path.suffix.casefold() == ".csv":
        return load_polycrit_csv(path)
    if path.is_dir():
        with tempfile.TemporaryDirectory(prefix="polycrit_eval_") as temporary:
            csv_path = Path(temporary) / "standardized_polycrit.csv"
            export_folder_to_csv(
                str(path),
                str(csv_path),
                mode="polycrit",
            )
            return load_polycrit_csv(csv_path, source_label=str(path))
    raise ValueError(
        f"Extracted source must be a PolyCrit CSV or standardized directory: {path}"
    )


def _split_tokens(value: Any) -> Tuple[str, ...]:
    text = _clean(value)
    if not text:
        return ()
    parts = re.split(r"\s*(?:,|;|/|\band\b)\s*", text, flags=re.IGNORECASE)
    keys = [_text_key(part) for part in parts if _text_key(part)]
    return tuple(dict.fromkeys(keys))


def _polymer_keys(value: Any) -> Tuple[str, ...]:
    text = _clean(value)
    if not text:
        return ()
    candidates = [text]
    candidates.extend(
        part.strip()
        for part in re.split(r",\s+|;\s*", text)
        if part.strip()
    )
    keys: List[str] = []
    for candidate in candidates:
        normalized = _normalize_polymer_name(candidate)
        key = _text_key(normalized)
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _solvent_keys(value: Any) -> Tuple[str, ...]:
    return tuple(_text_key(item) for item in canonicalize_solvent_list(_clean(value)))


NUMBER_RE = re.compile(r"(?<!\d)[+\-]?\d+(?:\.\d+)?")


def _numbers(value: Any) -> Tuple[float, ...]:
    text = _clean(value)
    if not text:
        return ()
    return tuple(float(match.group(0)) for match in NUMBER_RE.finditer(text))


def _years(value: Any) -> Tuple[str, ...]:
    years = re.findall(r"\b(?:19|20)\d{2}\b", _clean(value))
    return tuple(dict.fromkeys(years))


def _phase_key(value: Any) -> str:
    key = _text_key(value)
    if "hilic" in key or "hydrophilicinteraction" in key:
        return "hilic"
    if "reverse" in key:
        return "reverse"
    if "normal" in key:
        return "normal"
    return key


def _detector_keys(value: Any) -> Tuple[str, ...]:
    tokens = _split_tokens(value)
    normalized: List[str] = []
    for token in tokens:
        if token == "ri" or "refractiveindex" in token:
            key = "ri"
        elif token == "elsd" or "evaporativelightscattering" in token:
            key = "elsd"
        elif token == "dad" or "diodearray" in token:
            key = "dad"
        elif token == "uv" or token.startswith("uvvisible"):
            key = "uv"
        elif token == "nmr" or "nuclearmagneticresonance" in token:
            key = "nmr"
        else:
            key = token.removesuffix("detector")
        if key and key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def _jaccard(expected: Iterable[str], candidate: Iterable[str]) -> float:
    expected_set = set(expected)
    candidate_set = set(candidate)
    if not expected_set or not candidate_set:
        return 0.0
    return len(expected_set & candidate_set) / len(expected_set | candidate_set)


def _numeric_similarity(field: str, expected: Any, candidate: Any) -> float:
    expected_values = _numbers(expected)
    candidate_values = _numbers(candidate)
    if not expected_values or not candidate_values:
        return 0.0
    if len(expected_values) != len(candidate_values):
        return 0.0
    tolerance = NUMERIC_FIELD_TOLERANCES[field]
    matches = sum(
        math.isclose(left, right, rel_tol=1e-7, abs_tol=tolerance)
        for left, right in zip(expected_values, candidate_values)
    )
    return matches / len(expected_values)


def field_similarity(field: str, expected: Any, candidate: Any) -> float:
    """Return a strict, field aware value similarity from zero to one."""

    if not _clean(expected) or not _clean(candidate):
        return 0.0
    if field in NUMERIC_FIELD_TOLERANCES:
        return _numeric_similarity(field, expected, candidate)
    if field == "Author Year":
        return _jaccard(_years(expected), _years(candidate))
    if field in {"Polymer", "Alternate Polymer Names"}:
        return _jaccard(_polymer_keys(expected), _polymer_keys(candidate))
    if field == "Solvents":
        expected_solvents = _solvent_keys(expected)
        candidate_solvents = _solvent_keys(candidate)
        if expected_solvents == candidate_solvents and expected_solvents:
            return 1.0
        overlap = _jaccard(expected_solvents, candidate_solvents)
        if set(expected_solvents) == set(candidate_solvents) and overlap:
            return 0.5
        return overlap * 0.5
    if field in {"End Groups", "Detector"}:
        expected_tokens = (
            _detector_keys(expected) if field == "Detector" else _split_tokens(expected)
        )
        candidate_tokens = (
            _detector_keys(candidate) if field == "Detector" else _split_tokens(candidate)
        )
        return _jaccard(expected_tokens, candidate_tokens)
    if field == "Phase":
        return float(_phase_key(expected) == _phase_key(candidate))
    return float(_text_key(expected) == _text_key(candidate))


def _polymer_row_similarity(
    ground_truth: PolyCritRow,
    extracted: PolyCritRow,
) -> float:
    expected = set(_polymer_keys(ground_truth.values.get("Polymer")))
    expected.update(_polymer_keys(ground_truth.values.get("Alternate Polymer Names")))
    candidate = set(_polymer_keys(extracted.values.get("Polymer")))
    candidate.update(_polymer_keys(extracted.values.get("Alternate Polymer Names")))
    return _jaccard(expected, candidate)


def condition_similarity(
    ground_truth: PolyCritRow,
    extracted: PolyCritRow,
) -> float:
    """Score condition identity using populated ground truth discriminators."""

    numerator = 0.0
    denominator = 0.0
    for field, weight in MATCH_FIELD_WEIGHTS.items():
        expected = ground_truth.values.get(field)
        if not _clean(expected):
            continue
        denominator += weight
        if field in {"Polymer", "Alternate Polymer Names"}:
            similarity = _polymer_row_similarity(ground_truth, extracted)
        else:
            similarity = field_similarity(
                field,
                expected,
                extracted.values.get(field),
            )
        numerator += weight * similarity
    return numerator / denominator if denominator else 0.0


def _minimum_cost_assignment(costs: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Return an optimal rectangular assignment using the Hungarian method."""

    if not costs or not costs[0]:
        return []
    row_count = len(costs)
    column_count = len(costs[0])
    transposed = row_count > column_count
    matrix = (
        [list(row) for row in zip(*costs)]
        if transposed
        else [list(row) for row in costs]
    )
    row_count = len(matrix)
    column_count = len(matrix[0])

    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    column_match = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        column_match[0] = row
        current_column = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = column_match[current_column]
            delta = math.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    matrix[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    previous_column[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[column_match[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if column_match[current_column] == 0:
                break
        while True:
            next_column = previous_column[current_column]
            column_match[current_column] = column_match[next_column]
            current_column = next_column
            if current_column == 0:
                break

    pairs = [
        (column_match[column] - 1, column - 1)
        for column in range(1, column_count + 1)
        if column_match[column]
    ]
    if transposed:
        return [(column, row) for row, column in pairs]
    return pairs


def maximum_weight_assignment(
    weights: Sequence[Sequence[float]],
) -> List[Tuple[int, int]]:
    if not weights or not weights[0]:
        return []
    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    return _minimum_cost_assignment(costs)


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Match threshold must be between zero and one.")


def match_reference_group(
    reference: str,
    ground_truth_rows: Sequence[PolyCritRow],
    extracted_rows: Sequence[PolyCritRow],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Tuple[List[MatchedPair], List[PolyCritRow], List[PolyCritRow]]:
    """Find the best accepted one to one pairing for one reference."""

    _validate_threshold(threshold)
    if not ground_truth_rows:
        return [], [], list(extracted_rows)
    if not extracted_rows:
        return [], list(ground_truth_rows), []

    raw_scores = [
        [condition_similarity(gt_row, ex_row) for ex_row in extracted_rows]
        for gt_row in ground_truth_rows
    ]
    adjusted = [
        [max(0.0, score - threshold) for score in row]
        for row in raw_scores
    ]
    assignments = maximum_weight_assignment(adjusted)

    matched: List[MatchedPair] = []
    matched_gt = set()
    matched_ex = set()
    for gt_index, ex_index in assignments:
        score = raw_scores[gt_index][ex_index]
        if score <= threshold:
            continue
        matched.append(
            MatchedPair(
                reference=reference,
                ground_truth=ground_truth_rows[gt_index],
                extracted=extracted_rows[ex_index],
                score=score,
            )
        )
        matched_gt.add(gt_index)
        matched_ex.add(ex_index)

    unmatched_gt = [
        row for index, row in enumerate(ground_truth_rows) if index not in matched_gt
    ]
    unmatched_ex = [
        row for index, row in enumerate(extracted_rows) if index not in matched_ex
    ]
    return matched, unmatched_gt, unmatched_ex


def _field_result(field: str, expected: Any, candidate: Any) -> Dict[str, Any]:
    expected_text = _display_value(expected)
    candidate_text = _display_value(candidate)
    if not expected_text:
        status = "not_scored"
        similarity = None
    elif not candidate_text:
        status = "missing"
        similarity = 0.0
    else:
        similarity = field_similarity(field, expected, candidate)
        if math.isclose(similarity, 1.0, abs_tol=1e-12):
            status = "exact"
        elif similarity > 0:
            status = "partial"
        else:
            status = "mismatch"
    return {
        "status": status,
        "similarity": None if similarity is None else round(similarity, 4),
        "ground_truth": expected_text,
        "extracted": candidate_text,
    }


def _fields_for_pair(
    ground_truth: PolyCritRow,
    extracted: Optional[PolyCritRow],
) -> Dict[str, Dict[str, Any]]:
    return {
        field: _field_result(
            field,
            ground_truth.values.get(field),
            extracted.values.get(field) if extracted else None,
        )
        for field in EVALUATED_FIELDS
    }


def _row_payload(row: Optional[PolyCritRow]) -> Dict[str, str]:
    if row is None:
        return {}
    return {
        field: _display_value(row.values.get(field))
        for field in POLYCRIT_FIELDNAMES
    }


def _condition_result(
    reference: str,
    status: str,
    ground_truth: Optional[PolyCritRow] = None,
    extracted: Optional[PolyCritRow] = None,
    score: float = 0.0,
) -> Dict[str, Any]:
    fields = {}
    if ground_truth is not None and status in {"matched", "false_negative"}:
        fields = _fields_for_pair(ground_truth, extracted)
    return {
        "reference": reference,
        "status": status,
        "match_score": round(score, 4),
        "ground_truth_row_id": ground_truth.row_id if ground_truth else "",
        "extracted_row_id": extracted.row_id if extracted else "",
        "ground_truth_source": ground_truth.source if ground_truth else "",
        "extracted_source": extracted.source if extracted else "",
        "ground_truth_reference": _display_value(
            ground_truth.values.get(REFERENCE_FIELD)
        )
        if ground_truth
        else "",
        "extracted_reference": _display_value(
            extracted.values.get(REFERENCE_FIELD)
        )
        if extracted
        else "",
        "ground_truth_values": _row_payload(ground_truth),
        "extracted_values": _row_payload(extracted),
        "fields": fields,
    }


def _natural_reference_key(reference: str) -> Tuple[int, Any]:
    return (0, int(reference)) if reference.isdigit() else (1, reference)


def evaluate_rows(
    ground_truth_rows: Sequence[PolyCritRow],
    extracted_rows: Sequence[PolyCritRow],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate unordered condition rows through reference memberships."""

    _validate_threshold(threshold)
    gt_by_reference: Dict[str, List[PolyCritRow]] = defaultdict(list)
    ex_by_reference: Dict[str, List[PolyCritRow]] = defaultdict(list)
    no_reference_gt: List[PolyCritRow] = []
    no_reference_ex: List[PolyCritRow] = []

    for row in ground_truth_rows:
        if row.references:
            for reference in row.references:
                gt_by_reference[reference].append(row)
        else:
            no_reference_gt.append(row)
    for row in extracted_rows:
        if row.references:
            for reference in row.references:
                ex_by_reference[reference].append(row)
        else:
            no_reference_ex.append(row)

    condition_results: List[Dict[str, Any]] = []
    all_references = sorted(gt_by_reference, key=_natural_reference_key)
    for reference in all_references:
        matched, false_negatives, false_positives = match_reference_group(
            reference,
            gt_by_reference.get(reference, []),
            ex_by_reference.get(reference, []),
            threshold,
        )
        for pair in matched:
            condition_results.append(
                _condition_result(
                    reference,
                    "matched",
                    ground_truth=pair.ground_truth,
                    extracted=pair.extracted,
                    score=pair.score,
                )
            )
        for row in false_negatives:
            condition_results.append(
                _condition_result(
                    reference,
                    "false_negative",
                    ground_truth=row,
                )
            )
        for row in false_positives:
            condition_results.append(
                _condition_result(
                    reference,
                    "false_positive",
                    extracted=row,
                )
            )

    out_of_scope_references = sorted(
        set(ex_by_reference) - set(gt_by_reference),
        key=_natural_reference_key,
    )
    for reference in out_of_scope_references:
        for row in ex_by_reference[reference]:
            condition_results.append(
                _condition_result(
                    reference,
                    "out_of_scope",
                    extracted=row,
                )
            )

    for row in no_reference_gt:
        condition_results.append(
            _condition_result(
                "",
                "unmapped_ground_truth",
                ground_truth=row,
            )
        )
    for row in no_reference_ex:
        condition_results.append(
            _condition_result(
                "",
                "unmapped_extracted",
                extracted=row,
            )
        )

    status_counts = {
        "matched": 0,
        "false_negative": 0,
        "false_positive": 0,
        "out_of_scope": 0,
        "unmapped_ground_truth": 0,
        "unmapped_extracted": 0,
    }
    for result in condition_results:
        status_counts[result["status"]] += 1

    matched_count = status_counts["matched"]
    false_negative_count = status_counts["false_negative"]
    false_positive_count = status_counts["false_positive"]
    precision_denominator = matched_count + false_positive_count
    recall_denominator = matched_count + false_negative_count
    precision = matched_count / precision_denominator if precision_denominator else 0.0
    recall = matched_count / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    field_summary: Dict[str, Dict[str, Any]] = {}
    for field in EVALUATED_FIELDS:
        counts = {
            "exact": 0,
            "partial": 0,
            "mismatch": 0,
            "missing": 0,
            "not_scored": 0,
        }
        for condition in condition_results:
            field_result = condition["fields"].get(field)
            if field_result:
                counts[field_result["status"]] += 1
        expected = sum(
            counts[name] for name in ("exact", "partial", "mismatch", "missing")
        )
        present = counts["exact"] + counts["partial"] + counts["mismatch"]
        field_summary[field] = {
            **counts,
            "expected": expected,
            "coverage": round(present / expected, 4) if expected else 1.0,
            "exact_accuracy": round(counts["exact"] / expected, 4)
            if expected
            else 1.0,
            "weighted_score": round(
                (counts["exact"] + 0.5 * counts["partial"]) / expected,
                4,
            )
            if expected
            else 1.0,
        }

    gt_memberships = sum(max(1, len(row.references)) for row in ground_truth_rows)
    ex_memberships = sum(max(1, len(row.references)) for row in extracted_rows)
    return {
        "summary": {
            "ground_truth_rows": len(ground_truth_rows),
            "extracted_rows": len(extracted_rows),
            "ground_truth_reference_memberships": gt_memberships,
            "extracted_reference_memberships": ex_memberships,
            "matched_pairs": matched_count,
            "false_negatives": false_negative_count,
            "false_positives": false_positive_count,
            "out_of_scope_extracted_memberships": status_counts["out_of_scope"],
            "unmapped_ground_truth_rows": status_counts["unmapped_ground_truth"],
            "unmapped_extracted_rows": status_counts["unmapped_extracted"],
            "condition_precision": round(precision, 4),
            "condition_recall": round(recall, 4),
            "condition_f1": round(f1, 4),
            "match_threshold": threshold,
        },
        "field_summary": field_summary,
        "condition_results": condition_results,
        "notes": [
            "Rows are matched one to one within each reference membership.",
            "Composite ground truth references are evaluated once for each cited reference.",
            "False negatives and false positives are retained in condition metrics.",
            "Extracted references absent from the ground truth subset are out of scope.",
            "Blank ground truth fields are not scored.",
        ],
    }


def write_reports(report: Mapping[str, Any], output_dir: Path) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ground_truth_evaluation.json"
    condition_csv_path = output_dir / "ground_truth_condition_matches.csv"
    field_csv_path = output_dir / "ground_truth_field_scores.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    condition_fields = [
        "reference",
        "status",
        "match_score",
        "ground_truth_row_id",
        "extracted_row_id",
        "ground_truth_source",
        "extracted_source",
        "ground_truth_reference",
        "extracted_reference",
        "ground_truth_values",
        "extracted_values",
    ]
    with condition_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=condition_fields)
        writer.writeheader()
        for result in report["condition_results"]:
            output_row = {
                field: result.get(field, "") for field in condition_fields
            }
            output_row["ground_truth_values"] = json.dumps(
                result.get("ground_truth_values", {}),
                ensure_ascii=False,
            )
            output_row["extracted_values"] = json.dumps(
                result.get("extracted_values", {}),
                ensure_ascii=False,
            )
            writer.writerow(output_row)

    field_fields = [
        "reference",
        "condition_status",
        "ground_truth_row_id",
        "extracted_row_id",
        "field",
        "status",
        "similarity",
        "ground_truth",
        "extracted",
    ]
    with field_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_fields)
        writer.writeheader()
        for condition in report["condition_results"]:
            for field, result in condition["fields"].items():
                writer.writerow(
                    {
                        "reference": condition["reference"],
                        "condition_status": condition["status"],
                        "ground_truth_row_id": condition["ground_truth_row_id"],
                        "extracted_row_id": condition["extracted_row_id"],
                        "field": field,
                        **result,
                    }
                )
    return json_path, condition_csv_path, field_csv_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standardized PolyCrit rows against a workbook.",
        add_help=False,
    )
    parser.add_argument("ground_truth", help="PolyCrit ground truth XLSX path")
    parser.add_argument(
        "extracted",
        help="PolyCrit CSV path or standardized JSON directory",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "-t",
        dest="threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Minimum condition match score",
    )
    parser.add_argument("-h", action="help", help="Show this help message")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    ground_truth_path = Path(args.ground_truth)
    extracted_path = Path(args.extracted)
    output_dir = Path(args.output_dir)

    ground_truth_rows = load_polycrit_workbook(ground_truth_path)
    extracted_rows = load_extracted_rows(extracted_path)
    report = evaluate_rows(
        ground_truth_rows,
        extracted_rows,
        threshold=args.threshold,
    )
    json_path, condition_csv_path, field_csv_path = write_reports(report, output_dir)

    print(f"Saved JSON report: {json_path}")
    print(f"Saved condition report: {condition_csv_path}")
    print(f"Saved field report: {field_csv_path}")
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    main()
