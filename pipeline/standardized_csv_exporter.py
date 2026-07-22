"""Safely export corrected-standardizer LCCC JSON files to a PolyCrit CSV.

The exporter is deliberately conservative.  It does not perform a second
round of chemical standardization and it does not infer fields that the
extraction schema never requested.  Base PolyCrit columns are populated only
from explicitly approved values produced by the current standardizer.  Raw
values, statuses, candidates, reasons, and structured composition data are
retained in audit columns by default.

Expected input
--------------
``*_standardized.json`` files produced by the corrected
``pipeline/standardizer.py`` implementation.  Every document must contain
``extracted_data.conditions`` and every condition must contain the complete
status/raw-field fingerprint emitted by that implementation.  Older
standardizer outputs are rejected with an instruction to rerun the current
standardizer.

Examples
--------
Audit-rich export (recommended and default)::

    python standardized_csv_exporter.py \
        results/standardized \
        results/standardized_summary_review.csv

Exact 19-column PolyCrit projection, with audit columns intentionally omitted::

    python standardized_csv_exporter.py \
        results/standardized \
        results/standardized_summary.csv \
        --core-only

The output is written atomically.  If any input is malformed, an existing CSV
is left unchanged and the program exits nonzero.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableSequence, Optional, Sequence, Tuple


LOGGER = logging.getLogger("standardized_csv_exporter")
EXPORTER_VERSION = "1.1.0"


POLYCRIT_FIELDNAMES: Tuple[str, ...] = (
    "Reference",
    "Author Year",
    "Polymer",
    "Alternate Polymer Names",
    "End Groups",
    "Solvents",
    "Solvent Ratio (wt%)",
    "Solvent Ratio (vol%)",
    "Stationary Phase",
    "Manufacturer",
    "Base Material",
    "Base Material Modification",
    "Phase",
    "Particle diameter (μm)",
    "Pore size (Å)",
    "Temperature (Celsius)",
    "Flow Rate (mL/min)",
    "Injected Polymer Concentration (g/L)",
    "Detector",
)


AUDIT_FIELDNAMES: Tuple[str, ...] = (
    "Source JSON File",
    "Condition Index",
    "Reference (Raw)",
    "Exporter Version",
    "Standardized By",
    "Standardization Date",
    "Source Model",
    "Source Model Inputs",
    "Source Pipeline Metrics",
    "Export Flags",
    "Author Year (Raw)",
    "Publication Year (Raw)",
    "Publication Year (Standardized)",
    "Publication Year Status",
    "Paper DOI",
    "Corresponding Author Name",
    "Corresponding Email Address",
    "Physical Address",
    "Critical Condition Basis",
    "Critical Condition Confidence",
    "Model Confidences",
    "Qwen Confidence",
    "Mistral Confidence",
    "Analyte Polymer (Raw)",
    "Analyte Polymer (Standardized)",
    "Critical Component (Raw)",
    "Critical Component (Standardized)",
    "Comparison Polymer",
    "Comparison Polymer Status",
    "Comparison Polymer Candidates",
    "Comparison Polymer Reason",
    "Architecture (Raw)",
    "Architecture (Standardized)",
    "Architecture Status",
    "Architecture Candidates",
    "End Groups (Raw)",
    "Column Name (Raw)",
    "Column Name (Standardized)",
    "Column Name Status",
    "Column Name Candidates",
    "Column Name Reason",
    "Stationary Phase Chemistry (Raw)",
    "Stationary Phase Chemistry (Reported)",
    "Manufacturer (Raw)",
    "Base Material (Raw)",
    "Base Material Modification (Raw)",
    "Column Mode (Raw)",
    "Column Mode (Standardized)",
    "Column Mode Status",
    "Column Dimensions (Raw)",
    "Particle Diameter (Raw)",
    "Solvents (Raw)",
    "Solvents (Standardized)",
    "Solvent Details",
    "Solvent Qualifiers",
    "Solvent Additives",
    "Mobile Phase Component Order",
    "Solvent Status",
    "Solvent Ratio (Raw)",
    "Solvent Ratio (Standardized Field)",
    "Solvent Ratio Components",
    "Mobile Phase Composition",
    "Solvent Ratio Minimum",
    "Solvent Ratio Maximum",
    "Solvent Ratio Range Component",
    "Solvent Ratio Values Unassigned",
    "Solvent Ratio Scope",
    "Embedded Ratio Annotations",
    "Solvent Ratio Source",
    "Solvent Ratio Component Hint",
    "Solvent Ratio Status",
    "Solvent Ratio Units (Raw)",
    "Solvent Ratio Units (Standardized)",
    "Solvent Ratio Units Status",
    "Aqueous Parameters (Raw)",
    "Aqueous Parameters (Standardized)",
    "Aqueous Parameters Status",
    "Pore Size (Raw)",
    "Pore Size (Å, Standardized)",
    "Pore Size Minimum (Å)",
    "Pore Size Maximum (Å)",
    "Pore Size Uncertainty (Å)",
    "Pore Size Status",
    "Temperature (Raw)",
    "Temperature (Celsius, Standardized)",
    "Temperature Minimum (Celsius)",
    "Temperature Maximum (Celsius)",
    "Temperature Uncertainty (Celsius)",
    "Temperature Status",
    "Flow Rate (Raw)",
    "Flow Rate (mL/min, Standardized)",
    "Flow Rate Minimum (mL/min)",
    "Flow Rate Maximum (mL/min)",
    "Flow Rate Uncertainty (mL/min)",
    "Flow Rate Candidates (mL/min)",
    "Flow Rate Status",
    "Flow Rate Reason",
    "Injected Polymer Concentration (Raw)",
    "Detector (Raw)",
    "Evidence Text",
    "Notes",
)


REQUIRED_RAW_FIELDS: Tuple[str, ...] = (
    "analyte_polymer_raw",
    "critical_component_raw",
    "aqueous_parameters_raw",
    "mobile_phase_solvents_raw",
    "mobile_phase_ratio_units_raw",
    "mobile_phase_ratio_raw",
    "flow_rate_raw",
    "pore_size_raw",
    "temperature_celsius_raw",
    "column_name_raw",
    "column_mode_raw",
    "architecture_raw",
    "publication_year_raw",
)


REQUIRED_STANDARDIZED_FIELDS: Tuple[str, ...] = (
    "analyte_polymer",
    "comparison_polymer",
    "aqueous_parameters",
    "mobile_phase_solvents",
    "mobile_phase_solvent_details",
    "mobile_phase_solvent_qualifiers",
    "mobile_phase_additives",
    "mobile_phase_component_order",
    "mobile_phase_ratio_units",
    "mobile_phase_ratio",
    "mobile_phase_ratio_components",
    "mobile_phase_ratio_min",
    "mobile_phase_ratio_max",
    "mobile_phase_composition",
    "flow_rate",
    "flow_rate_ml_per_min",
    "flow_rate_min_ml_per_min",
    "flow_rate_max_ml_per_min",
    "flow_rate_uncertainty_ml_per_min",
    "pore_size",
    "pore_size_angstrom",
    "pore_size_min_angstrom",
    "pore_size_max_angstrom",
    "pore_size_uncertainty_angstrom",
    "temperature_celsius",
    "temperature_min_celsius",
    "temperature_max_celsius",
    "temperature_uncertainty_celsius",
    "column_name",
    "column_mode",
    "critical_component",
    "architecture",
    "publication_year",
)


STATUS_VOCABULARIES: Mapping[str, frozenset[str]] = {
    "comparison_polymer_standardization_status": frozenset(
        {"exact", "alias", "ambiguous", "unmapped"}
    ),
    "mobile_phase_solvent_standardization_status": frozenset(
        {"exact", "alias", "unmapped"}
    ),
    "aqueous_parameters_standardization_status": frozenset(
        {"exact", "alias", "ambiguous", "unmapped", "invalid", "reclassified_additive"}
    ),
    "mobile_phase_ratio_units_standardization_status": frozenset(
        {"exact", "alias", "ambiguous", "unmapped"}
    ),
    "mobile_phase_ratio_standardization_status": frozenset(
        {
            "parsed",
            "parsed_named",
            "parsed_unscoped",
            "parsed_assumed_percent",
            "range",
            "range_named",
            "ambiguous",
            "invalid",
            "unmapped",
        }
    ),
    "flow_rate_standardization_status": frozenset(
        {
            "parsed",
            "parsed_range",
            "parsed_with_uncertainty",
            "assumed_ml_per_min",
            "assumed_ml_per_min_range",
            "ambiguous",
            "invalid",
            "incomplete_unit",
            "unmapped",
        }
    ),
    "pore_size_standardization_status": frozenset(
        {"parsed", "parsed_range", "parsed_with_uncertainty", "ambiguous", "invalid", "unmapped"}
    ),
    "temperature_standardization_status": frozenset(
        {"parsed", "parsed_with_uncertainty", "assumed_celsius", "invalid", "unmapped"}
    ),
    "column_name_standardization_status": frozenset(
        {"exact", "alias", "ambiguous", "unmapped"}
    ),
    "column_mode_standardization_status": frozenset(
        {"exact", "alias", "unmapped"}
    ),
    "architecture_standardization_status": frozenset(
        {"exact", "alias", "ambiguous", "unmapped"}
    ),
    "publication_year_standardization_status": frozenset(
        {"parsed", "invalid", "unmapped"}
    ),
}


APPROVED_IDENTITY_STATUSES = frozenset({"exact", "alias"})
APPROVED_RATIO_STATUSES = frozenset({"parsed", "parsed_named"})
APPROVED_UNIT_STATUSES = frozenset({"exact", "alias"})
APPROVED_FLOW_STATUSES = frozenset({"parsed", "parsed_range"})
APPROVED_PORE_STATUSES = frozenset({"parsed", "parsed_range"})
APPROVED_TEMPERATURE_STATUSES = frozenset({"parsed"})
APPROVED_POLYCRIT_PHASES = frozenset({"Normal", "Reverse", "HILIC"})
RATIO_SUM_TOLERANCE = Decimal("0.02")


_EXPLICIT_PORE_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?\s*"
    r"(?:Å|angstroms?\b|A\b|nm\b|[uµμ]m\b)",
    re.IGNORECASE,
)
_PORE_NUMBER_TOKEN_RE = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)
_NON_PORE_DIMENSION_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?\s*"
    r"(?:mm|cm|met(?:er|re)s?)\b",
    re.IGNORECASE,
)
_DIMENSION_SEPARATOR_RE = re.compile(r"\b\d[^,;]*\s[x×]\s*\d", re.IGNORECASE)
_SIGNED_NUMERIC_CELL_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)


class ExportValidationError(ValueError):
    """Raised when an input violates the corrected-standardizer contract."""


@dataclass(frozen=True)
class LoadedDocument:
    """A validated standardized document and its stable source identity."""

    path: Path
    relative_path: str
    metadata: Mapping[str, Any]
    conditions: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class ExportSummary:
    """Compact result returned by :func:`export_folder_to_csv`."""

    files: int
    conditions: int
    rows: int
    flagged_rows: int
    flag_counts: Mapping[str, int]
    output_path: Path


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"", "null", "none", "nan", "n/a", "na"}
    return False


def _raw_value(condition: Mapping[str, Any], field: str) -> Any:
    """Return the authoritative raw sibling, preserving falsey raw values."""
    raw_field = f"{field}_raw"
    if raw_field in condition:
        return condition[raw_field]
    return condition.get(field)


def _text_cell(value: Any, *, context: str) -> str:
    """Render a scalar display cell without lossy list deduplication."""
    if _is_nullish(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        return _number_cell(value, context=context)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, dict)):
        return _json_cell(value, context=context)
    raise ExportValidationError(
        f"{context}: unsupported value type {type(value).__name__}"
    )


def _json_cell(value: Any, *, context: str) -> str:
    """Serialize structured audit data deterministically and losslessly."""
    if value is None:
        return ""
    if isinstance(value, str):
        # Audit cells preserve authoritative source strings exactly, including
        # leading/trailing whitespace and literal text such as "null" or "NA".
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        return _number_cell(value, context=context)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ExportValidationError(f"{context}: cannot serialize audit value: {exc}") from exc


def _as_decimal(value: Any, *, context: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ExportValidationError(f"{context}: expected a finite number, received {value!r}")
    if not isinstance(value, (int, float, Decimal, str)):
        raise ExportValidationError(
            f"{context}: expected a finite number, received {type(value).__name__}"
        )
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ExportValidationError(f"{context}: invalid numeric value {value!r}") from exc
    if not number.is_finite():
        raise ExportValidationError(f"{context}: non-finite numeric value {value!r}")
    return number


def _number_cell(value: Any, *, context: str) -> str:
    """Render a number without corrupting scientific notation."""
    number = _as_decimal(value, context=context)
    if number == 0:
        return "0"
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _number_or_blank(value: Any, *, context: str) -> str:
    if _is_nullish(value):
        return ""
    return _number_cell(value, context=context)


def _append_flag(flags: MutableSequence[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _excel_safe_cell(value: str) -> str:
    """Neutralize spreadsheet formulas without changing ordinary numbers."""
    candidate = value.lstrip(" \t\r\n")
    if not candidate or candidate[0] not in {"=", "+", "-", "@"}:
        return value
    if _SIGNED_NUMERIC_CELL_RE.fullmatch(value.strip()):
        return value
    return f"'{value}"


def _flag_if_reported(
    flags: MutableSequence[str],
    flag: str,
    raw_value: Any,
    *,
    always: bool = False,
) -> None:
    if always or not _is_nullish(raw_value):
        _append_flag(flags, flag)


def _require_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportValidationError(f"{context}: expected a non-empty string")
    return value.strip()


def _approved_identity(
    condition: Mapping[str, Any],
    value_field: str,
    status_field: str,
    raw_field: str,
    flag_prefix: str,
    flags: MutableSequence[str],
    *,
    context: str,
) -> str:
    status = condition[status_field]
    if status in APPROVED_IDENTITY_STATUSES:
        return _require_string(condition.get(value_field), context=f"{context}.{value_field}")
    _flag_if_reported(
        flags,
        f"{flag_prefix}:{status}",
        condition.get(raw_field),
        always=flag_prefix == "polymer",
    )
    return ""


def _approved_solvents(
    condition: Mapping[str, Any],
    flags: MutableSequence[str],
    *,
    context: str,
) -> Tuple[List[str], str]:
    status = condition["mobile_phase_solvent_standardization_status"]
    if status not in APPROVED_IDENTITY_STATUSES:
        _flag_if_reported(
            flags,
            f"solvents:{status}",
            condition["mobile_phase_solvents_raw"],
        )
        return [], ""

    raw_solvents = condition.get("mobile_phase_solvents")
    if not isinstance(raw_solvents, list) or not raw_solvents:
        raise ExportValidationError(
            f"{context}.mobile_phase_solvents: status {status!r} requires a non-empty list"
        )

    solvents: List[str] = []
    keys: set[str] = set()
    for index, value in enumerate(raw_solvents):
        solvent = _require_string(
            value,
            context=f"{context}.mobile_phase_solvents[{index}]",
        )
        key = solvent.casefold()
        if key in keys:
            raise ExportValidationError(
                f"{context}.mobile_phase_solvents: duplicate standardized solvent {solvent!r}"
            )
        keys.add(key)
        solvents.append(solvent)

    qualifiers = condition.get("mobile_phase_solvent_qualifiers")
    if not isinstance(qualifiers, list) or len(qualifiers) != len(solvents):
        raise ExportValidationError(
            f"{context}.mobile_phase_solvent_qualifiers: expected {len(solvents)} aligned values"
        )

    for index, qualifier in enumerate(qualifiers):
        if qualifier is None or (isinstance(qualifier, str) and not qualifier.strip()):
            continue
        _require_string(
            qualifier,
            context=f"{context}.mobile_phase_solvent_qualifiers[{index}]",
        )

    # Qualifiers remain in their dedicated audit column.  The PolyCrit core
    # Solvents field contains canonical chemical names only.
    return solvents, ", ".join(solvents)


def _approved_ratio_columns(
    condition: Mapping[str, Any],
    approved_solvents: Sequence[str],
    flags: MutableSequence[str],
    *,
    context: str,
) -> Tuple[str, str]:
    ratio_raw = condition["mobile_phase_ratio_raw"]
    ratio_status = condition["mobile_phase_ratio_standardization_status"]
    if ratio_status not in APPROVED_RATIO_STATUSES:
        _flag_if_reported(flags, f"ratio:{ratio_status}", ratio_raw)
        return "", ""

    unit_status = condition["mobile_phase_ratio_units_standardization_status"]
    if unit_status not in APPROVED_UNIT_STATUSES:
        _flag_if_reported(
            flags,
            f"ratio_units:{unit_status}",
            condition["mobile_phase_ratio_units_raw"],
            always=True,
        )
        return "", ""

    solvent_status = condition["mobile_phase_solvent_standardization_status"]
    if solvent_status not in APPROVED_IDENTITY_STATUSES or not approved_solvents:
        _append_flag(flags, f"ratio_solvents:{solvent_status}")
        return "", ""

    unit = condition.get("mobile_phase_ratio_units")
    if unit not in {"w/w", "v/v"}:
        _append_flag(flags, f"ratio_unit_not_projectable:{unit or 'missing'}")
        return "", ""

    composition = condition.get("mobile_phase_composition")
    if not isinstance(composition, list) or not composition:
        raise ExportValidationError(
            f"{context}.mobile_phase_composition: status {ratio_status!r} requires a non-empty list"
        )

    names: List[str] = []
    values: List[Decimal] = []
    for index, item in enumerate(composition):
        item_context = f"{context}.mobile_phase_composition[{index}]"
        if not isinstance(item, dict):
            raise ExportValidationError(f"{item_context}: expected an object")
        component_type = item.get("component_type")
        if component_type != "solvent":
            _append_flag(flags, "ratio_contains_non_solvent_component")
            return "", ""
        if item.get("units") != unit:
            raise ExportValidationError(
                f"{item_context}.units: expected {unit!r}, received {item.get('units')!r}"
            )
        name = item.get("solvent") or item.get("component")
        names.append(_require_string(name, context=f"{item_context}.solvent"))
        number = _as_decimal(item.get("value"), context=f"{item_context}.value")
        if number < 0 or number > 100:
            raise ExportValidationError(
                f"{item_context}.value: expected a percentage from 0 through 100"
            )
        values.append(number)

    if names != list(approved_solvents):
        _append_flag(flags, "ratio_component_alignment_mismatch")
        return "", ""

    if len({name.casefold() for name in names}) != len(names):
        raise ExportValidationError(
            f"{context}.mobile_phase_composition: duplicate solvent components"
        )

    if abs(sum(values, Decimal("0")) - Decimal("100")) > RATIO_SUM_TOLERANCE:
        _append_flag(flags, "ratio_total_not_100")
        return "", ""

    rendered = ", ".join(
        _number_cell(value, context=f"{context}.mobile_phase_composition.value")
        for value in values
    )
    if unit == "w/w":
        return rendered, ""
    return "", rendered


def _approved_measurement(
    condition: Mapping[str, Any],
    *,
    status_field: str,
    safe_statuses: frozenset[str],
    value_field: str,
    minimum_field: str,
    maximum_field: str,
    uncertainty_field: Optional[str],
    raw_field: str,
    flag_prefix: str,
    flags: MutableSequence[str],
    context: str,
    allow_multiple_values: bool = False,
) -> str:
    status = condition[status_field]
    if status not in safe_statuses:
        _flag_if_reported(flags, f"{flag_prefix}:{status}", condition[raw_field])
        return ""

    value = condition.get(value_field)
    minimum = condition.get(minimum_field)
    maximum = condition.get(maximum_field)
    uncertainty = condition.get(uncertainty_field) if uncertainty_field else None

    if isinstance(value, list):
        if not allow_multiple_values and len(value) > 1:
            _append_flag(flags, f"{flag_prefix}:multiple_values_not_projectable")
            return ""
        if not value:
            raise ExportValidationError(f"{context}.{value_field}: empty numeric list")
        return ", ".join(
            _number_cell(item, context=f"{context}.{value_field}[{index}]")
            for index, item in enumerate(value)
        )

    has_value = not _is_nullish(value)
    has_minimum = not _is_nullish(minimum)
    has_maximum = not _is_nullish(maximum)

    if has_minimum != has_maximum:
        raise ExportValidationError(
            f"{context}: {minimum_field} and {maximum_field} must be populated together"
        )
    if has_value and has_minimum:
        raise ExportValidationError(
            f"{context}: scalar and range measurement fields are both populated"
        )
    if not has_value and not has_minimum:
        raise ExportValidationError(
            f"{context}: status {status!r} has no standardized numeric value"
        )

    if has_minimum:
        low = _as_decimal(minimum, context=f"{context}.{minimum_field}")
        high = _as_decimal(maximum, context=f"{context}.{maximum_field}")
        if high < low:
            raise ExportValidationError(f"{context}: standardized range is descending")
        return (
            f"{_number_cell(low, context=f'{context}.{minimum_field}')}-"
            f"{_number_cell(high, context=f'{context}.{maximum_field}')}"
        )

    if not _is_nullish(uncertainty):
        # Uncertainty-bearing statuses are intentionally excluded from the
        # current base projection.  This guard catches inconsistent upstream
        # data if a safe scalar/range status nevertheless carries uncertainty.
        raise ExportValidationError(
            f"{context}.{uncertainty_field}: unexpected uncertainty for status {status!r}"
        )
    return _number_cell(value, context=f"{context}.{value_field}")


def _reference_from_source(source_pdf: Any, fallback_stem: str) -> Tuple[str, str, bool]:
    raw = "" if source_pdf is None else str(source_pdf)
    source = raw.strip() or fallback_stem
    basename = re.split(r"[\\/]", source)[-1].strip()
    basename = re.sub(r"\.pdf\s*$", "", basename, flags=re.IGNORECASE)
    match = re.match(r"^\s*\[(\d+)]", basename)
    if match:
        reference = match.group(1)
    elif re.fullmatch(r"\d+(?:\s*,\s*\d+)*", basename):
        reference = re.sub(r"\s+", "", basename)
    else:
        reference = ""
    return reference, raw, not bool(raw.strip())


def _has_unambiguous_explicit_pore_unit(value: Any, status: str) -> bool:
    """Reject unitless or mixed-dimension text before projecting pore values."""
    if isinstance(value, str):
        if _NON_PORE_DIMENSION_UNIT_RE.search(value) or _DIMENSION_SEPARATOR_RE.search(value):
            return False
        explicit_matches = list(_EXPLICIT_PORE_UNIT_RE.finditer(value))
        if not explicit_matches:
            return False

        explicit_spans = [match.span() for match in explicit_matches]
        first_explicit_start = explicit_spans[0][0]
        for number_match in _PORE_NUMBER_TOKEN_RE.finditer(value):
            start = number_match.start()
            if any(span_start <= start < span_end for span_start, span_end in explicit_spans):
                continue
            # Leading endpoints/list values can inherit a final unit, as in
            # "100, 300 Å" or "100-300 Å".
            if start < first_explicit_start:
                continue
            # A trailing endpoint or uncertainty may inherit the preceding
            # unit only when the standardizer explicitly classified that
            # structure as a range or uncertainty.
            if status in {"parsed_range", "parsed_with_uncertainty"}:
                continue
            return False
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(_has_unambiguous_explicit_pore_unit(item, status) for item in value)
    return False


def _reject_nonstandard_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token!r}")


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _validate_finite_json_values(value: Any, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExportValidationError(f"{context}: non-finite JSON number {value!r}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json_values(item, context=f"{context}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json_values(item, context=f"{context}.{key}")


def _validate_status(
    condition: Mapping[str, Any],
    status_field: str,
    allowed: frozenset[str],
    *,
    context: str,
) -> None:
    if status_field not in condition:
        raise ExportValidationError(
            f"{context}: missing {status_field}; rerun the corrected standardizer"
        )
    value = condition[status_field]
    if not isinstance(value, str) or value not in allowed:
        raise ExportValidationError(
            f"{context}.{status_field}: unsupported status {value!r}; "
            "rerun the corrected standardizer or update the exporter contract"
        )


def _require_null(value: Any, *, context: str) -> None:
    if value is not None:
        raise ExportValidationError(f"{context}: expected null, received {value!r}")


def _require_decimal_between(
    value: Any,
    minimum: Decimal,
    maximum: Decimal,
    *,
    context: str,
) -> Decimal:
    number = _as_decimal(value, context=context)
    if number < minimum or number > maximum:
        raise ExportValidationError(
            f"{context}: expected a value from {minimum} through {maximum}, received {value!r}"
        )
    return number


def _require_numeric_range(
    minimum_value: Any,
    maximum_value: Any,
    minimum_allowed: Decimal,
    maximum_allowed: Decimal,
    *,
    context: str,
) -> Tuple[Decimal, Decimal]:
    minimum = _require_decimal_between(
        minimum_value,
        minimum_allowed,
        maximum_allowed,
        context=f"{context}.minimum",
    )
    maximum = _require_decimal_between(
        maximum_value,
        minimum_allowed,
        maximum_allowed,
        context=f"{context}.maximum",
    )
    if maximum < minimum:
        raise ExportValidationError(f"{context}: standardized range is descending")
    return minimum, maximum


def _require_no_numeric_outputs(
    condition: Mapping[str, Any],
    fields: Sequence[str],
    *,
    context: str,
) -> None:
    for field in fields:
        _require_null(condition.get(field), context=f"{context}.{field}")


def _validate_identity_invariants(
    condition: Mapping[str, Any],
    *,
    context: str,
) -> None:
    identity_contracts = (
        (
            "comparison_polymer",
            "comparison_polymer_standardization_status",
            frozenset({"ambiguous", "unmapped"}),
        ),
        (
            "column_name",
            "column_name_standardization_status",
            frozenset({"ambiguous", "unmapped"}),
        ),
        (
            "column_mode",
            "column_mode_standardization_status",
            frozenset({"unmapped"}),
        ),
    )
    for value_field, status_field, null_statuses in identity_contracts:
        status = condition[status_field]
        value = condition.get(value_field)
        if status in APPROVED_IDENTITY_STATUSES:
            _require_string(value, context=f"{context}.{value_field}")
        elif status in null_statuses:
            _require_null(value, context=f"{context}.{value_field}")

    architecture_status = condition["architecture_standardization_status"]
    architecture = condition.get("architecture")
    if architecture_status in APPROVED_IDENTITY_STATUSES:
        # The current standardizer intentionally maps a small set of reported
        # non-architecture descriptors to null while retaining an exact
        # lookup status.  Preserve that state for review instead of failing the
        # entire document; the row-level flag collector makes it visible.
        if architecture is not None:
            _require_string(architecture, context=f"{context}.architecture")
    elif architecture_status == "unmapped":
        _require_null(architecture, context=f"{context}.architecture")
    elif architecture is not None and not isinstance(architecture, str):
        raise ExportValidationError(
            f"{context}.architecture: ambiguous value must be a string or null"
        )

    year_status = condition["publication_year_standardization_status"]
    year = condition.get("publication_year")
    if year_status == "parsed":
        current_year = datetime.now(timezone.utc).year
        if isinstance(year, bool) or not isinstance(year, int):
            raise ExportValidationError(
                f"{context}.publication_year: parsed status requires an integer"
            )
        if not 1800 <= year <= current_year + 1:
            raise ExportValidationError(
                f"{context}.publication_year: parsed year {year} is outside the accepted range"
            )
    else:
        _require_null(year, context=f"{context}.publication_year")


def _validate_solvent_invariants(
    condition: Mapping[str, Any],
    *,
    context: str,
) -> None:
    solvents = condition["mobile_phase_solvents"]
    qualifiers = condition["mobile_phase_solvent_qualifiers"]
    details = condition["mobile_phase_solvent_details"]

    if solvents is None:
        if qualifiers:
            raise ExportValidationError(
                f"{context}.mobile_phase_solvent_qualifiers: expected an empty list when solvents are null"
            )
    else:
        if len(qualifiers) != len(solvents):
            raise ExportValidationError(
                f"{context}.mobile_phase_solvent_qualifiers: expected {len(solvents)} aligned values"
            )
        seen: set[str] = set()
        for index, solvent in enumerate(solvents):
            text = _require_string(
                solvent,
                context=f"{context}.mobile_phase_solvents[{index}]",
            )
            key = text.casefold()
            if key in seen:
                raise ExportValidationError(
                    f"{context}.mobile_phase_solvents: duplicate standardized solvent {text!r}"
                )
            seen.add(key)

    for index, qualifier in enumerate(qualifiers):
        if qualifier is not None and not isinstance(qualifier, str):
            raise ExportValidationError(
                f"{context}.mobile_phase_solvent_qualifiers[{index}]: expected a string or null"
            )

    resolved_details: List[str] = []
    detail_statuses: List[str] = []
    for index, detail in enumerate(details):
        item_context = f"{context}.mobile_phase_solvent_details[{index}]"
        if not isinstance(detail, dict):
            raise ExportValidationError(f"{item_context}: expected an object")
        status = detail.get("standardization_status")
        if status not in {"exact", "alias", "unmapped"}:
            raise ExportValidationError(
                f"{item_context}.standardization_status: unsupported status {status!r}"
            )
        detail_statuses.append(status)
        if status in APPROVED_IDENTITY_STATUSES:
            resolved_details.append(
                _require_string(detail.get("solvent"), context=f"{item_context}.solvent")
            )
        else:
            _require_null(detail.get("solvent"), context=f"{item_context}.solvent")

    overall_status = condition["mobile_phase_solvent_standardization_status"]
    if overall_status in APPROVED_IDENTITY_STATUSES:
        if not isinstance(solvents, list) or not solvents or not details:
            raise ExportValidationError(
                f"{context}.mobile_phase_solvents: status {overall_status!r} requires resolved solvents"
            )
        if any(status == "unmapped" for status in detail_statuses):
            raise ExportValidationError(
                f"{context}.mobile_phase_solvent_details: approved overall status contains an unmapped detail"
            )
        ordered_unique: List[str] = []
        seen_detail: set[str] = set()
        for solvent in resolved_details:
            key = solvent.casefold()
            if key not in seen_detail:
                seen_detail.add(key)
                ordered_unique.append(solvent)
        if ordered_unique != solvents:
            raise ExportValidationError(
                f"{context}.mobile_phase_solvents: list contradicts resolved solvent details"
            )
        expected_status = "alias" if "alias" in detail_statuses else "exact"
        if overall_status != expected_status:
            raise ExportValidationError(
                f"{context}.mobile_phase_solvent_standardization_status: expected {expected_status!r}"
            )

    for field, allowed_types in (
        ("mobile_phase_additives", {"exact", "alias", "unmapped"}),
        ("mobile_phase_component_order", {"solvent", "additive"}),
    ):
        for index, item in enumerate(condition[field]):
            item_context = f"{context}.{field}[{index}]"
            if not isinstance(item, dict):
                raise ExportValidationError(f"{item_context}: expected an object")
            if field == "mobile_phase_additives":
                status = item.get("standardization_status")
                if status not in allowed_types:
                    raise ExportValidationError(
                        f"{item_context}.standardization_status: unsupported status {status!r}"
                    )
            else:
                _require_string(item.get("component"), context=f"{item_context}.component")
                if item.get("component_type") not in allowed_types:
                    raise ExportValidationError(
                        f"{item_context}.component_type: expected 'solvent' or 'additive'"
                    )


def _validate_ratio_invariants(
    condition: Mapping[str, Any],
    *,
    context: str,
) -> None:
    unit_status = condition["mobile_phase_ratio_units_standardization_status"]
    unit = condition.get("mobile_phase_ratio_units")
    if unit_status in APPROVED_UNIT_STATUSES and unit not in {"v/v", "w/w", "w/v", "v/w"}:
        raise ExportValidationError(
            f"{context}.mobile_phase_ratio_units: approved status requires a canonical unit"
        )

    components = condition.get("mobile_phase_ratio_components")
    component_values: Optional[List[Decimal]] = None
    if components is not None:
        if not isinstance(components, list) or not components:
            raise ExportValidationError(
                f"{context}.mobile_phase_ratio_components: expected a non-empty list or null"
            )
        component_values = [
            _require_decimal_between(
                value,
                Decimal("0"),
                Decimal("100"),
                context=f"{context}.mobile_phase_ratio_components[{index}]",
            )
            for index, value in enumerate(components)
        ]

    composition = condition.get("mobile_phase_composition")
    composition_values: Optional[List[Decimal]] = None
    if composition is not None:
        if not isinstance(composition, list) or not composition:
            raise ExportValidationError(
                f"{context}.mobile_phase_composition: expected a non-empty list or null"
            )
        composition_values = []
        for index, item in enumerate(composition):
            item_context = f"{context}.mobile_phase_composition[{index}]"
            if not isinstance(item, dict):
                raise ExportValidationError(f"{item_context}: expected an object")
            _require_string(item.get("component"), context=f"{item_context}.component")
            if item.get("component_type") not in {"solvent", "additive"}:
                raise ExportValidationError(
                    f"{item_context}.component_type: expected 'solvent' or 'additive'"
                )
            composition_values.append(
                _require_decimal_between(
                    item.get("value"),
                    Decimal("0"),
                    Decimal("100"),
                    context=f"{item_context}.value",
                )
            )

    if component_values is not None and composition_values is not None:
        if component_values != composition_values:
            raise ExportValidationError(
                f"{context}: mobile_phase_ratio_components contradict mobile_phase_composition"
            )

    status = condition["mobile_phase_ratio_standardization_status"]
    raw = condition["mobile_phase_ratio_raw"]
    minimum = condition.get("mobile_phase_ratio_min")
    maximum = condition.get("mobile_phase_ratio_max")
    if status in {"parsed", "parsed_named", "parsed_assumed_percent"}:
        if _is_nullish(raw):
            raise ExportValidationError(
                f"{context}.mobile_phase_ratio_raw: status {status!r} requires a reported ratio"
            )
        if component_values is None or composition_values is None:
            raise ExportValidationError(
                f"{context}: status {status!r} requires components and structured composition"
            )
        _require_null(minimum, context=f"{context}.mobile_phase_ratio_min")
        _require_null(maximum, context=f"{context}.mobile_phase_ratio_max")
    elif status == "parsed_unscoped":
        if _is_nullish(raw) or component_values is None:
            raise ExportValidationError(
                f"{context}: parsed_unscoped requires a reported ratio and components"
            )
        _require_null(composition, context=f"{context}.mobile_phase_composition")
        _require_null(minimum, context=f"{context}.mobile_phase_ratio_min")
        _require_null(maximum, context=f"{context}.mobile_phase_ratio_max")
    elif status in {"range", "range_named"}:
        if _is_nullish(raw):
            raise ExportValidationError(
                f"{context}.mobile_phase_ratio_raw: range status requires a reported ratio"
            )
        _require_numeric_range(
            minimum,
            maximum,
            Decimal("0"),
            Decimal("100"),
            context=f"{context}.mobile_phase_ratio",
        )
        _require_null(components, context=f"{context}.mobile_phase_ratio_components")
        _require_null(composition, context=f"{context}.mobile_phase_composition")
    else:
        _require_null(minimum, context=f"{context}.mobile_phase_ratio_min")
        _require_null(maximum, context=f"{context}.mobile_phase_ratio_max")


def _validate_flow_invariants(
    condition: Mapping[str, Any],
    *,
    context: str,
) -> None:
    fields = (
        "flow_rate_ml_per_min",
        "flow_rate_min_ml_per_min",
        "flow_rate_max_ml_per_min",
        "flow_rate_uncertainty_ml_per_min",
    )
    status = condition["flow_rate_standardization_status"]
    scalar, minimum, maximum, uncertainty = (condition.get(field) for field in fields)
    if status in {"parsed", "assumed_ml_per_min"}:
        number = _as_decimal(scalar, context=f"{context}.flow_rate_ml_per_min")
        if number <= 0:
            raise ExportValidationError(f"{context}.flow_rate_ml_per_min: must be positive")
        _require_no_numeric_outputs(condition, fields[1:], context=context)
    elif status in {"parsed_range", "assumed_ml_per_min_range"}:
        low, _ = _require_numeric_range(
            minimum,
            maximum,
            Decimal("0"),
            Decimal("1E+12"),
            context=f"{context}.flow_rate",
        )
        if low <= 0:
            raise ExportValidationError(f"{context}.flow_rate_min_ml_per_min: must be positive")
        _require_null(scalar, context=f"{context}.flow_rate_ml_per_min")
        _require_null(uncertainty, context=f"{context}.flow_rate_uncertainty_ml_per_min")
    elif status == "parsed_with_uncertainty":
        number = _as_decimal(scalar, context=f"{context}.flow_rate_ml_per_min")
        delta = _as_decimal(
            uncertainty,
            context=f"{context}.flow_rate_uncertainty_ml_per_min",
        )
        if number <= 0 or delta < 0:
            raise ExportValidationError(
                f"{context}: flow value must be positive and uncertainty nonnegative"
            )
        _require_null(minimum, context=f"{context}.flow_rate_min_ml_per_min")
        _require_null(maximum, context=f"{context}.flow_rate_max_ml_per_min")
    else:
        _require_no_numeric_outputs(condition, fields, context=context)


def _validate_pore_invariants(
    condition: Mapping[str, Any],
    *,
    context: str,
) -> None:
    fields = (
        "pore_size_angstrom",
        "pore_size_min_angstrom",
        "pore_size_max_angstrom",
        "pore_size_uncertainty_angstrom",
    )
    status = condition["pore_size_standardization_status"]
    value, minimum, maximum, uncertainty = (condition.get(field) for field in fields)
    if status == "parsed":
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ExportValidationError(f"{context}.pore_size_angstrom: empty list")
        for index, item in enumerate(values):
            _require_decimal_between(
                item,
                Decimal("1"),
                Decimal("10000"),
                context=f"{context}.pore_size_angstrom[{index}]",
            )
        _require_no_numeric_outputs(condition, fields[1:], context=context)
    elif status == "parsed_range":
        _require_numeric_range(
            minimum,
            maximum,
            Decimal("1"),
            Decimal("10000"),
            context=f"{context}.pore_size",
        )
        _require_null(value, context=f"{context}.pore_size_angstrom")
        _require_null(uncertainty, context=f"{context}.pore_size_uncertainty_angstrom")
    elif status == "parsed_with_uncertainty":
        _require_decimal_between(
            value,
            Decimal("1"),
            Decimal("10000"),
            context=f"{context}.pore_size_angstrom",
        )
        delta = _as_decimal(
            uncertainty,
            context=f"{context}.pore_size_uncertainty_angstrom",
        )
        if delta < 0:
            raise ExportValidationError(
                f"{context}.pore_size_uncertainty_angstrom: must be nonnegative"
            )
        _require_null(minimum, context=f"{context}.pore_size_min_angstrom")
        _require_null(maximum, context=f"{context}.pore_size_max_angstrom")
    else:
        _require_no_numeric_outputs(condition, fields, context=context)


def _validate_temperature_invariants(
    condition: Mapping[str, Any],
    *,
    context: str,
) -> None:
    fields = (
        "temperature_celsius",
        "temperature_min_celsius",
        "temperature_max_celsius",
        "temperature_uncertainty_celsius",
    )
    status = condition["temperature_standardization_status"]
    value, minimum, maximum, uncertainty = (condition.get(field) for field in fields)
    if status in {"parsed", "assumed_celsius"}:
        if value is not None:
            _require_decimal_between(
                value,
                Decimal("-150"),
                Decimal("400"),
                context=f"{context}.temperature_celsius",
            )
            _require_null(minimum, context=f"{context}.temperature_min_celsius")
            _require_null(maximum, context=f"{context}.temperature_max_celsius")
        else:
            _require_numeric_range(
                minimum,
                maximum,
                Decimal("-150"),
                Decimal("400"),
                context=f"{context}.temperature",
            )
        _require_null(uncertainty, context=f"{context}.temperature_uncertainty_celsius")
    elif status == "parsed_with_uncertainty":
        _require_decimal_between(
            value,
            Decimal("-150"),
            Decimal("400"),
            context=f"{context}.temperature_celsius",
        )
        delta = _as_decimal(
            uncertainty,
            context=f"{context}.temperature_uncertainty_celsius",
        )
        if delta < 0:
            raise ExportValidationError(
                f"{context}.temperature_uncertainty_celsius: must be nonnegative"
            )
        _require_null(minimum, context=f"{context}.temperature_min_celsius")
        _require_null(maximum, context=f"{context}.temperature_max_celsius")
    else:
        _require_no_numeric_outputs(condition, fields, context=context)


def _validate_condition_invariants(
    condition: Mapping[str, Any],
    *,
    context: str,
) -> None:
    _validate_identity_invariants(condition, context=context)
    _validate_solvent_invariants(condition, context=context)
    _validate_ratio_invariants(condition, context=context)
    _validate_flow_invariants(condition, context=context)
    _validate_pore_invariants(condition, context=context)
    _validate_temperature_invariants(condition, context=context)

    aqueous_status = condition["aqueous_parameters_standardization_status"]
    if aqueous_status in {"exact", "alias", "reclassified_additive"} and not isinstance(
        condition.get("aqueous_parameters"), dict
    ):
        raise ExportValidationError(
            f"{context}.aqueous_parameters: status {aqueous_status!r} requires an object"
        )


def _validate_condition_contract(condition: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(condition, dict):
        raise ExportValidationError(f"{context}: every condition must be an object")

    missing_raw = [field for field in REQUIRED_RAW_FIELDS if field not in condition]
    if missing_raw:
        raise ExportValidationError(
            f"{context}: missing corrected-standardizer raw fields {missing_raw}; "
            "rerun the corrected standardizer"
        )

    missing_standardized = [
        field for field in REQUIRED_STANDARDIZED_FIELDS if field not in condition
    ]
    if missing_standardized:
        raise ExportValidationError(
            f"{context}: missing corrected-standardizer fields {missing_standardized}; "
            "rerun the corrected standardizer"
        )

    for status_field, allowed in STATUS_VOCABULARIES.items():
        _validate_status(condition, status_field, allowed, context=context)

    list_fields = (
        "mobile_phase_solvent_details",
        "mobile_phase_solvent_qualifiers",
        "mobile_phase_additives",
        "mobile_phase_component_order",
    )
    for field in list_fields:
        if not isinstance(condition[field], list):
            raise ExportValidationError(f"{context}.{field}: expected a list")

    solvents = condition["mobile_phase_solvents"]
    if solvents is not None and not isinstance(solvents, list):
        raise ExportValidationError(
            f"{context}.mobile_phase_solvents: expected a list or null"
        )

    composition = condition["mobile_phase_composition"]
    if composition is not None and not isinstance(composition, list):
        raise ExportValidationError(
            f"{context}.mobile_phase_composition: expected a list or null"
        )

    aqueous = condition["aqueous_parameters"]
    if aqueous is not None and not isinstance(aqueous, dict):
        raise ExportValidationError(
            f"{context}.aqueous_parameters: expected an object or null"
        )

    _validate_condition_invariants(condition, context=context)

    return condition


def _load_document(path: Path, root: Path) -> LoadedDocument:
    relative_path = path.relative_to(root).as_posix()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(
                handle,
                parse_constant=_reject_nonstandard_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ExportValidationError(f"{relative_path}: cannot read valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ExportValidationError(f"{relative_path}: top-level JSON value must be an object")
    _validate_finite_json_values(data, context=relative_path)

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ExportValidationError(f"{relative_path}: metadata must be an object")
    _require_string(
        metadata.get("standardized_by"),
        context=f"{relative_path}.metadata.standardized_by",
    )
    standardization_date = _require_string(
        metadata.get("standardization_date"),
        context=f"{relative_path}.metadata.standardization_date",
    )
    try:
        parsed_date = datetime.fromisoformat(standardization_date.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExportValidationError(
            f"{relative_path}.metadata.standardization_date: invalid ISO-8601 timestamp"
        ) from exc
    if parsed_date.tzinfo is None:
        raise ExportValidationError(
            f"{relative_path}.metadata.standardization_date: timezone information is required"
        )
    if "source_pdf" not in metadata:
        raise ExportValidationError(f"{relative_path}.metadata: missing source_pdf")
    source_pdf = metadata.get("source_pdf")
    if source_pdf is not None and not isinstance(source_pdf, str):
        raise ExportValidationError(
            f"{relative_path}.metadata.source_pdf: expected a string or null"
        )

    extracted_data = data.get("extracted_data")
    if not isinstance(extracted_data, dict):
        raise ExportValidationError(
            f"{relative_path}: missing or invalid extracted_data object"
        )
    if "conditions" not in extracted_data:
        raise ExportValidationError(f"{relative_path}: missing extracted_data.conditions")
    conditions = extracted_data["conditions"]
    if not isinstance(conditions, list):
        raise ExportValidationError(
            f"{relative_path}: extracted_data.conditions must be a list"
        )

    validated: List[Mapping[str, Any]] = []
    for zero_based_index, condition in enumerate(conditions):
        validated.append(
            _validate_condition_contract(
                condition,
                context=f"{relative_path}.conditions[{zero_based_index}]",
            )
        )

    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ExportValidationError(f"{relative_path}: summary must be an object")
    if "total_conditions" not in summary:
        raise ExportValidationError(f"{relative_path}.summary: missing total_conditions")
    total = summary["total_conditions"]
    if isinstance(total, bool) or not isinstance(total, int):
        raise ExportValidationError(
            f"{relative_path}.summary.total_conditions: expected an integer"
        )
    if total != len(validated):
        raise ExportValidationError(
            f"{relative_path}.summary.total_conditions={total} does not match "
            f"{len(validated)} condition records"
        )

    return LoadedDocument(
        path=path,
        relative_path=relative_path,
        metadata=metadata,
        conditions=validated,
    )


def _model_confidence(condition: Mapping[str, Any], model: str) -> Any:
    confidences = condition.get("model_confidences")
    if not isinstance(confidences, dict):
        return None
    return confidences.get(model)


def _collect_review_only_flags(
    document: LoadedDocument,
    condition: Mapping[str, Any],
    flags: MutableSequence[str],
) -> None:
    architecture_raw = condition["architecture_raw"]
    architecture_status = condition["architecture_standardization_status"]
    if not _is_nullish(architecture_raw) and (
        architecture_status not in APPROVED_IDENTITY_STATUSES
        or condition.get("architecture") is None
    ):
        _append_flag(flags, f"architecture:{architecture_status}")

    publication_raw = condition["publication_year_raw"]
    publication_status = condition["publication_year_standardization_status"]
    if not _is_nullish(publication_raw) and publication_status != "parsed":
        _append_flag(flags, f"publication_year:{publication_status}")

    aqueous_raw = condition["aqueous_parameters_raw"]
    aqueous_status = condition["aqueous_parameters_standardization_status"]
    if not _is_nullish(aqueous_raw) and aqueous_status not in {
        "exact",
        "alias",
        "reclassified_additive",
    }:
        _append_flag(flags, f"aqueous_parameters:{aqueous_status}")

    confidence = condition.get("critical_condition_confidence")
    if isinstance(confidence, str) and confidence.strip().casefold() == "unclear":
        _append_flag(flags, "critical_condition_confidence:unclear")

    for field in (
        "author_year",
        "end_groups",
        "manufacturer",
        "base_material",
        "base_material_modification",
        "particle_diameter_um",
        "injected_polymer_concentration_g_l",
    ):
        if not _is_nullish(_raw_value(condition, field)):
            _append_flag(flags, f"not_projected:{field}")

    pipeline_metrics = document.metadata.get("pipeline_metrics")
    if isinstance(pipeline_metrics, dict) and pipeline_metrics.get("success") is False:
        _append_flag(flags, "source_pipeline:reported_failure")


def _build_row(
    document: LoadedDocument,
    condition: Mapping[str, Any],
    condition_index: int,
) -> Tuple[Dict[str, str], List[str]]:
    context = f"{document.relative_path}.conditions[{condition_index - 1}]"
    flags: List[str] = []

    source_pdf = document.metadata.get("source_pdf")
    fallback_stem = document.path.stem.removesuffix("_standardized")
    reference, reference_raw, used_fallback = _reference_from_source(
        source_pdf,
        fallback_stem,
    )
    if used_fallback:
        _append_flag(flags, "reference:fallback_filename")
    if not reference:
        _append_flag(flags, "reference:non_numeric_identifier")

    polymer = _approved_identity(
        condition,
        "comparison_polymer",
        "comparison_polymer_standardization_status",
        "critical_component_raw",
        "polymer",
        flags,
        context=context,
    )
    stationary_phase = _approved_identity(
        condition,
        "column_name",
        "column_name_standardization_status",
        "column_name_raw",
        "column_name",
        flags,
        context=context,
    )
    phase = _approved_identity(
        condition,
        "column_mode",
        "column_mode_standardization_status",
        "column_mode_raw",
        "phase",
        flags,
        context=context,
    )
    if phase and phase not in APPROVED_POLYCRIT_PHASES:
        _append_flag(flags, f"phase:not_polycrit_phase:{phase}")
        phase = ""

    approved_solvents, solvent_display = _approved_solvents(
        condition,
        flags,
        context=context,
    )
    weight_ratio, volume_ratio = _approved_ratio_columns(
        condition,
        approved_solvents,
        flags,
        context=context,
    )

    if (
        condition["pore_size_standardization_status"] in APPROVED_PORE_STATUSES
        and not _has_unambiguous_explicit_pore_unit(
            condition["pore_size_raw"],
            condition["pore_size_standardization_status"],
        )
    ):
        _append_flag(flags, "pore_size:unit_or_dimension_ambiguous")
        pore_size = ""
    else:
        pore_size = _approved_measurement(
            condition,
            status_field="pore_size_standardization_status",
            safe_statuses=APPROVED_PORE_STATUSES,
            value_field="pore_size_angstrom",
            minimum_field="pore_size_min_angstrom",
            maximum_field="pore_size_max_angstrom",
            uncertainty_field="pore_size_uncertainty_angstrom",
            raw_field="pore_size_raw",
            flag_prefix="pore_size",
            flags=flags,
            context=context,
            allow_multiple_values=True,
        )
    temperature = _approved_measurement(
        condition,
        status_field="temperature_standardization_status",
        safe_statuses=APPROVED_TEMPERATURE_STATUSES,
        value_field="temperature_celsius",
        minimum_field="temperature_min_celsius",
        maximum_field="temperature_max_celsius",
        uncertainty_field="temperature_uncertainty_celsius",
        raw_field="temperature_celsius_raw",
        flag_prefix="temperature",
        flags=flags,
        context=context,
    )
    flow_rate = _approved_measurement(
        condition,
        status_field="flow_rate_standardization_status",
        safe_statuses=APPROVED_FLOW_STATUSES,
        value_field="flow_rate_ml_per_min",
        minimum_field="flow_rate_min_ml_per_min",
        maximum_field="flow_rate_max_ml_per_min",
        uncertainty_field="flow_rate_uncertainty_ml_per_min",
        raw_field="flow_rate_raw",
        flag_prefix="flow_rate",
        flags=flags,
        context=context,
    )

    _collect_review_only_flags(document, condition, flags)

    # The current extraction schema does not request author-year, alternate
    # polymer names, end groups, manufacturer/support metadata, particle
    # diameter, or injection concentration.  These remain blank rather than
    # being guessed from other text fields.
    row: Dict[str, str] = {
        "Reference": reference,
        "Author Year": "",
        "Polymer": polymer,
        "Alternate Polymer Names": "",
        "End Groups": "",
        "Solvents": solvent_display,
        "Solvent Ratio (wt%)": weight_ratio,
        "Solvent Ratio (vol%)": volume_ratio,
        "Stationary Phase": stationary_phase,
        "Manufacturer": "",
        "Base Material": "",
        "Base Material Modification": "",
        "Phase": phase,
        "Particle diameter (μm)": "",
        "Pore size (Å)": pore_size,
        "Temperature (Celsius)": temperature,
        "Flow Rate (mL/min)": flow_rate,
        "Injected Polymer Concentration (g/L)": "",
        "Detector": _text_cell(condition.get("detector"), context=f"{context}.detector"),
    }

    model_confidences = condition.get("model_confidences")
    row.update(
        {
            "Source JSON File": document.relative_path,
            "Condition Index": str(condition_index),
            "Reference (Raw)": reference_raw,
            "Exporter Version": EXPORTER_VERSION,
            "Standardized By": _text_cell(
                document.metadata.get("standardized_by"),
                context=f"{context}.metadata.standardized_by",
            ),
            "Standardization Date": _text_cell(
                document.metadata.get("standardization_date"),
                context=f"{context}.metadata.standardization_date",
            ),
            "Source Model": _text_cell(
                document.metadata.get("model"),
                context=f"{context}.metadata.model",
            ),
            "Source Model Inputs": _json_cell(
                document.metadata.get("inputs"),
                context=f"{context}.metadata.inputs",
            ),
            "Source Pipeline Metrics": _json_cell(
                document.metadata.get("pipeline_metrics"),
                context=f"{context}.metadata.pipeline_metrics",
            ),
            "Export Flags": " | ".join(flags),
            "Author Year (Raw)": _json_cell(
                _raw_value(condition, "author_year"),
                context=f"{context}.author_year_raw",
            ),
            "Publication Year (Raw)": _json_cell(
                condition["publication_year_raw"],
                context=f"{context}.publication_year_raw",
            ),
            "Publication Year (Standardized)": _number_or_blank(
                condition.get("publication_year"),
                context=f"{context}.publication_year",
            ),
            "Publication Year Status": condition["publication_year_standardization_status"],
            "Paper DOI": _json_cell(
                _raw_value(condition, "paper_doi"),
                context=f"{context}.paper_doi",
            ),
            "Corresponding Author Name": _json_cell(
                _raw_value(condition, "corresponding_author_name"),
                context=f"{context}.corresponding_author_name",
            ),
            "Corresponding Email Address": _json_cell(
                _raw_value(condition, "corresponding_email_address"),
                context=f"{context}.corresponding_email_address",
            ),
            "Physical Address": _json_cell(
                _raw_value(condition, "physical_address"),
                context=f"{context}.physical_address",
            ),
            "Critical Condition Basis": _json_cell(
                condition.get("critical_condition_basis"),
                context=f"{context}.critical_condition_basis",
            ),
            "Critical Condition Confidence": _json_cell(
                condition.get("critical_condition_confidence"),
                context=f"{context}.critical_condition_confidence",
            ),
            "Model Confidences": _json_cell(
                model_confidences,
                context=f"{context}.model_confidences",
            ),
            "Qwen Confidence": _text_cell(
                _model_confidence(condition, "qwen"),
                context=f"{context}.model_confidences.qwen",
            ),
            "Mistral Confidence": _text_cell(
                _model_confidence(condition, "mistral"),
                context=f"{context}.model_confidences.mistral",
            ),
            "Analyte Polymer (Raw)": _json_cell(
                condition["analyte_polymer_raw"],
                context=f"{context}.analyte_polymer_raw",
            ),
            "Analyte Polymer (Standardized)": _json_cell(
                condition.get("analyte_polymer"),
                context=f"{context}.analyte_polymer",
            ),
            "Critical Component (Raw)": _json_cell(
                condition["critical_component_raw"],
                context=f"{context}.critical_component_raw",
            ),
            "Critical Component (Standardized)": _json_cell(
                condition.get("critical_component"),
                context=f"{context}.critical_component",
            ),
            "Comparison Polymer": _text_cell(
                condition.get("comparison_polymer"),
                context=f"{context}.comparison_polymer",
            ),
            "Comparison Polymer Status": condition[
                "comparison_polymer_standardization_status"
            ],
            "Comparison Polymer Candidates": _json_cell(
                condition.get("comparison_polymer_standardization_candidates"),
                context=f"{context}.comparison_polymer_standardization_candidates",
            ),
            "Comparison Polymer Reason": _text_cell(
                condition.get("comparison_polymer_standardization_reason"),
                context=f"{context}.comparison_polymer_standardization_reason",
            ),
            "Architecture (Raw)": _json_cell(
                condition["architecture_raw"],
                context=f"{context}.architecture_raw",
            ),
            "Architecture (Standardized)": _json_cell(
                condition.get("architecture"),
                context=f"{context}.architecture",
            ),
            "Architecture Status": condition["architecture_standardization_status"],
            "Architecture Candidates": _json_cell(
                condition.get("architecture_standardization_candidates"),
                context=f"{context}.architecture_standardization_candidates",
            ),
            "End Groups (Raw)": _json_cell(
                _raw_value(condition, "end_groups"),
                context=f"{context}.end_groups",
            ),
            "Column Name (Raw)": _json_cell(
                condition["column_name_raw"],
                context=f"{context}.column_name_raw",
            ),
            "Column Name (Standardized)": _json_cell(
                condition.get("column_name"),
                context=f"{context}.column_name",
            ),
            "Column Name Status": condition["column_name_standardization_status"],
            "Column Name Candidates": _json_cell(
                condition.get("column_name_standardization_candidates"),
                context=f"{context}.column_name_standardization_candidates",
            ),
            "Column Name Reason": _text_cell(
                condition.get("column_name_standardization_reason"),
                context=f"{context}.column_name_standardization_reason",
            ),
            "Stationary Phase Chemistry (Raw)": _json_cell(
                _raw_value(condition, "stationary_phase_chemistry"),
                context=f"{context}.stationary_phase_chemistry_raw",
            ),
            "Stationary Phase Chemistry (Reported)": _json_cell(
                condition.get("stationary_phase_chemistry"),
                context=f"{context}.stationary_phase_chemistry",
            ),
            "Manufacturer (Raw)": _json_cell(
                _raw_value(condition, "manufacturer"),
                context=f"{context}.manufacturer",
            ),
            "Base Material (Raw)": _json_cell(
                _raw_value(condition, "base_material"),
                context=f"{context}.base_material",
            ),
            "Base Material Modification (Raw)": _json_cell(
                _raw_value(condition, "base_material_modification"),
                context=f"{context}.base_material_modification",
            ),
            "Column Mode (Raw)": _json_cell(
                condition["column_mode_raw"],
                context=f"{context}.column_mode_raw",
            ),
            "Column Mode (Standardized)": _json_cell(
                condition.get("column_mode"),
                context=f"{context}.column_mode",
            ),
            "Column Mode Status": condition["column_mode_standardization_status"],
            "Column Dimensions (Raw)": _json_cell(
                _raw_value(condition, "column_dimensions"),
                context=f"{context}.column_dimensions",
            ),
            "Particle Diameter (Raw)": _json_cell(
                _raw_value(condition, "particle_diameter_um"),
                context=f"{context}.particle_diameter_um",
            ),
            "Solvents (Raw)": _json_cell(
                condition["mobile_phase_solvents_raw"],
                context=f"{context}.mobile_phase_solvents_raw",
            ),
            "Solvents (Standardized)": _json_cell(
                condition.get("mobile_phase_solvents"),
                context=f"{context}.mobile_phase_solvents",
            ),
            "Solvent Details": _json_cell(
                condition.get("mobile_phase_solvent_details"),
                context=f"{context}.mobile_phase_solvent_details",
            ),
            "Solvent Qualifiers": _json_cell(
                condition.get("mobile_phase_solvent_qualifiers"),
                context=f"{context}.mobile_phase_solvent_qualifiers",
            ),
            "Solvent Additives": _json_cell(
                condition.get("mobile_phase_additives"),
                context=f"{context}.mobile_phase_additives",
            ),
            "Mobile Phase Component Order": _json_cell(
                condition.get("mobile_phase_component_order"),
                context=f"{context}.mobile_phase_component_order",
            ),
            "Solvent Status": condition["mobile_phase_solvent_standardization_status"],
            "Solvent Ratio (Raw)": _json_cell(
                condition["mobile_phase_ratio_raw"],
                context=f"{context}.mobile_phase_ratio_raw",
            ),
            "Solvent Ratio (Standardized Field)": _json_cell(
                condition.get("mobile_phase_ratio"),
                context=f"{context}.mobile_phase_ratio",
            ),
            "Solvent Ratio Components": _json_cell(
                condition.get("mobile_phase_ratio_components"),
                context=f"{context}.mobile_phase_ratio_components",
            ),
            "Mobile Phase Composition": _json_cell(
                condition.get("mobile_phase_composition"),
                context=f"{context}.mobile_phase_composition",
            ),
            "Solvent Ratio Minimum": _number_or_blank(
                condition.get("mobile_phase_ratio_min"),
                context=f"{context}.mobile_phase_ratio_min",
            ),
            "Solvent Ratio Maximum": _number_or_blank(
                condition.get("mobile_phase_ratio_max"),
                context=f"{context}.mobile_phase_ratio_max",
            ),
            "Solvent Ratio Range Component": _json_cell(
                condition.get("mobile_phase_ratio_range_component"),
                context=f"{context}.mobile_phase_ratio_range_component",
            ),
            "Solvent Ratio Values Unassigned": _json_cell(
                condition.get("mobile_phase_ratio_values_unassigned"),
                context=f"{context}.mobile_phase_ratio_values_unassigned",
            ),
            "Solvent Ratio Scope": _json_cell(
                condition.get("mobile_phase_ratio_scope"),
                context=f"{context}.mobile_phase_ratio_scope",
            ),
            "Embedded Ratio Annotations": _json_cell(
                condition.get("mobile_phase_embedded_ratio_annotations"),
                context=f"{context}.mobile_phase_embedded_ratio_annotations",
            ),
            "Solvent Ratio Source": _json_cell(
                condition.get("mobile_phase_ratio_source"),
                context=f"{context}.mobile_phase_ratio_source",
            ),
            "Solvent Ratio Component Hint": _json_cell(
                condition.get("mobile_phase_ratio_component_hint"),
                context=f"{context}.mobile_phase_ratio_component_hint",
            ),
            "Solvent Ratio Status": condition["mobile_phase_ratio_standardization_status"],
            "Solvent Ratio Units (Raw)": _json_cell(
                condition["mobile_phase_ratio_units_raw"],
                context=f"{context}.mobile_phase_ratio_units_raw",
            ),
            "Solvent Ratio Units (Standardized)": _json_cell(
                condition.get("mobile_phase_ratio_units"),
                context=f"{context}.mobile_phase_ratio_units",
            ),
            "Solvent Ratio Units Status": condition[
                "mobile_phase_ratio_units_standardization_status"
            ],
            "Aqueous Parameters (Raw)": _json_cell(
                condition["aqueous_parameters_raw"],
                context=f"{context}.aqueous_parameters_raw",
            ),
            "Aqueous Parameters (Standardized)": _json_cell(
                condition.get("aqueous_parameters"),
                context=f"{context}.aqueous_parameters",
            ),
            "Aqueous Parameters Status": condition[
                "aqueous_parameters_standardization_status"
            ],
            "Pore Size (Raw)": _json_cell(
                condition["pore_size_raw"],
                context=f"{context}.pore_size_raw",
            ),
            "Pore Size (Å, Standardized)": _json_cell(
                condition.get("pore_size_angstrom"),
                context=f"{context}.pore_size_angstrom",
            ),
            "Pore Size Minimum (Å)": _number_or_blank(
                condition.get("pore_size_min_angstrom"),
                context=f"{context}.pore_size_min_angstrom",
            ),
            "Pore Size Maximum (Å)": _number_or_blank(
                condition.get("pore_size_max_angstrom"),
                context=f"{context}.pore_size_max_angstrom",
            ),
            "Pore Size Uncertainty (Å)": _number_or_blank(
                condition.get("pore_size_uncertainty_angstrom"),
                context=f"{context}.pore_size_uncertainty_angstrom",
            ),
            "Pore Size Status": condition["pore_size_standardization_status"],
            "Temperature (Raw)": _json_cell(
                condition["temperature_celsius_raw"],
                context=f"{context}.temperature_celsius_raw",
            ),
            "Temperature (Celsius, Standardized)": _number_or_blank(
                condition.get("temperature_celsius"),
                context=f"{context}.temperature_celsius",
            ),
            "Temperature Minimum (Celsius)": _number_or_blank(
                condition.get("temperature_min_celsius"),
                context=f"{context}.temperature_min_celsius",
            ),
            "Temperature Maximum (Celsius)": _number_or_blank(
                condition.get("temperature_max_celsius"),
                context=f"{context}.temperature_max_celsius",
            ),
            "Temperature Uncertainty (Celsius)": _number_or_blank(
                condition.get("temperature_uncertainty_celsius"),
                context=f"{context}.temperature_uncertainty_celsius",
            ),
            "Temperature Status": condition["temperature_standardization_status"],
            "Flow Rate (Raw)": _json_cell(
                condition["flow_rate_raw"],
                context=f"{context}.flow_rate_raw",
            ),
            "Flow Rate (mL/min, Standardized)": _number_or_blank(
                condition.get("flow_rate_ml_per_min"),
                context=f"{context}.flow_rate_ml_per_min",
            ),
            "Flow Rate Minimum (mL/min)": _number_or_blank(
                condition.get("flow_rate_min_ml_per_min"),
                context=f"{context}.flow_rate_min_ml_per_min",
            ),
            "Flow Rate Maximum (mL/min)": _number_or_blank(
                condition.get("flow_rate_max_ml_per_min"),
                context=f"{context}.flow_rate_max_ml_per_min",
            ),
            "Flow Rate Uncertainty (mL/min)": _number_or_blank(
                condition.get("flow_rate_uncertainty_ml_per_min"),
                context=f"{context}.flow_rate_uncertainty_ml_per_min",
            ),
            "Flow Rate Candidates (mL/min)": _json_cell(
                condition.get("flow_rate_candidates_ml_per_min"),
                context=f"{context}.flow_rate_candidates_ml_per_min",
            ),
            "Flow Rate Status": condition["flow_rate_standardization_status"],
            "Flow Rate Reason": _text_cell(
                condition.get("flow_rate_standardization_reason"),
                context=f"{context}.flow_rate_standardization_reason",
            ),
            "Injected Polymer Concentration (Raw)": _json_cell(
                _raw_value(condition, "injected_polymer_concentration_g_l"),
                context=f"{context}.injected_polymer_concentration_g_l",
            ),
            "Detector (Raw)": _json_cell(
                _raw_value(condition, "detector"),
                context=f"{context}.detector",
            ),
            "Evidence Text": _json_cell(
                condition.get("evidence_text"),
                context=f"{context}.evidence_text",
            ),
            "Notes": _json_cell(
                condition.get("notes"),
                context=f"{context}.notes",
            ),
        }
    )

    expected_fields = set(POLYCRIT_FIELDNAMES) | set(AUDIT_FIELDNAMES)
    actual_fields = set(row)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ExportValidationError(
            f"{context}: internal row schema mismatch; missing={missing}, extra={extra}"
        )

    return row, flags


def _atomic_write_csv(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_mode = stat.S_IMODE(output_path.stat().st_mode)
    else:
        current_umask = os.umask(0)
        os.umask(current_umask)
        output_mode = 0o666 & ~current_umask

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, output_mode)
        os.replace(temporary_path, output_path)

        # Durably record the directory entry on filesystems that support
        # directory fsync.  A failure here is logged because replacement has
        # already completed and cannot be safely rolled back.
        if hasattr(os, "O_DIRECTORY"):
            try:
                directory_descriptor = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError as exc:
                LOGGER.warning("Could not fsync output directory %s: %s", output_path.parent, exc)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def export_folder_to_csv(
    folder_path: str | Path,
    output_csv: str | Path,
    mode: str = "polycrit",
    *,
    polymer_reference_xlsx: Optional[str] = None,
    include_review_columns: bool = True,
    core_only: bool = False,
    fail_on_flags: bool = False,
    allow_empty: bool = False,
    excel_safe: bool = False,
) -> ExportSummary:
    """Export all current ``*_standardized.json`` files below a directory.

    ``mode`` and ``polymer_reference_xlsx`` are retained for command-line/API
    compatibility with the earlier draft.  Only the safe PolyCrit projection
    is supported, and no external alias workbook is used.
    """
    if mode != "polycrit":
        raise ExportValidationError(
            "Only mode='polycrit' is supported for corrected-standardizer output. "
            "The obsolete legacy schema is intentionally rejected."
        )
    if polymer_reference_xlsx:
        LOGGER.warning(
            "--polymer-reference-xlsx is deprecated and ignored; approved identities "
            "come only from the corrected standardizer"
        )
    if core_only and include_review_columns is True:
        include_review_columns = False

    folder = Path(folder_path).expanduser().resolve()
    requested_output = Path(output_csv).expanduser()
    if requested_output.is_symlink():
        raise ExportValidationError(
            f"Refusing to replace a symbolic-link output path: {requested_output}"
        )
    output = requested_output.resolve()
    if output.suffix.casefold() != ".csv":
        raise ExportValidationError(
            f"Output must use a .csv suffix; received: {output}"
        )
    if output.exists() and output.is_dir():
        raise ExportValidationError(f"Output path is a directory: {output}")
    if not folder.is_dir():
        raise FileNotFoundError(f"Standardized directory does not exist: {folder}")

    input_paths = sorted(
        folder.rglob("*_standardized.json"),
        key=lambda path: path.relative_to(folder).as_posix(),
    )
    if not input_paths:
        raise FileNotFoundError(f"No *_standardized.json files found in: {folder}")
    if output in {path.resolve() for path in input_paths}:
        raise ExportValidationError(
            f"Refusing to overwrite a standardized JSON input with CSV output: {output}"
        )

    documents = [_load_document(path, folder) for path in input_paths]
    expected_conditions = sum(len(document.conditions) for document in documents)
    if expected_conditions == 0 and not allow_empty:
        raise ExportValidationError(
            "The standardized files contain zero conditions.  Refusing to replace the CSV; "
            "use --allow-empty only when a header-only export is intentional."
        )

    rows: List[Dict[str, str]] = []
    flag_counts: Counter[str] = Counter()
    flagged_rows = 0
    for document in documents:
        for condition_index, condition in enumerate(document.conditions, start=1):
            row, flags = _build_row(document, condition, condition_index)
            if flags:
                flagged_rows += 1
                flag_counts.update(flags)
            rows.append(row)

    if len(rows) != expected_conditions:
        raise ExportValidationError(
            f"Internal row-count error: expected {expected_conditions}, built {len(rows)}"
        )
    if fail_on_flags and flagged_rows:
        counts = ", ".join(f"{name}={count}" for name, count in sorted(flag_counts.items()))
        raise ExportValidationError(
            f"Refusing export because {flagged_rows} rows have review flags: {counts}"
        )

    fieldnames: Tuple[str, ...]
    if include_review_columns:
        fieldnames = (*POLYCRIT_FIELDNAMES, *AUDIT_FIELDNAMES)
    else:
        fieldnames = POLYCRIT_FIELDNAMES

    if len(fieldnames) != len(set(fieldnames)):
        raise ExportValidationError("Internal schema error: duplicate CSV field names")

    projected_rows = [{field: row[field] for field in fieldnames} for row in rows]
    if excel_safe:
        projected_rows = [
            {field: _excel_safe_cell(value) for field, value in row.items()}
            for row in projected_rows
        ]
    _atomic_write_csv(output, fieldnames, projected_rows)

    return ExportSummary(
        files=len(documents),
        conditions=expected_conditions,
        rows=len(rows),
        flagged_rows=flagged_rows,
        flag_counts=dict(sorted(flag_counts.items())),
        output_path=output,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively export corrected-standardizer LCCC JSON files to a "
            "PolyCrit CSV with audit columns by default."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="results/standardized",
        help="Directory containing *_standardized.json files",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="results/standardized_summary_review.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--mode",
        default="polycrit",
        choices=("polycrit",),
        help="Compatibility option; only 'polycrit' is supported",
    )
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument(
        "--core-only",
        action="store_true",
        help="Write only the 19 PolyCrit columns (audit columns are recommended)",
    )
    review_group.add_argument(
        "--include-review-columns",
        action="store_true",
        help="Deprecated compatibility flag; audit columns are already the default",
    )
    parser.add_argument(
        "--polymer-reference-xlsx",
        default=None,
        help="Deprecated and ignored; the exporter never remaps polymer identities",
    )
    parser.add_argument(
        "--fail-on-flags",
        action="store_true",
        help="Fail without replacing the CSV if any row needs manual review",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow a deliberate header-only CSV when every standardized file has zero conditions",
    )
    parser.add_argument(
        "--excel-safe",
        action="store_true",
        help=(
            "Prefix formula-like text cells with an apostrophe before opening untrusted "
            "exports in spreadsheet applications"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {EXPORTER_VERSION}",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(levelname)s: %(message)s",
    )
    try:
        summary = export_folder_to_csv(
            arguments.folder,
            arguments.output,
            mode=arguments.mode,
            polymer_reference_xlsx=arguments.polymer_reference_xlsx,
            include_review_columns=(
                arguments.include_review_columns or not arguments.core_only
            ),
            core_only=arguments.core_only,
            fail_on_flags=arguments.fail_on_flags,
            allow_empty=arguments.allow_empty,
            excel_safe=arguments.excel_safe,
        )
    except (ExportValidationError, FileNotFoundError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print(
        f"Exported {summary.rows} conditions from {summary.files} standardized "
        f"JSON files to {summary.output_path}"
    )
    if summary.flagged_rows:
        print(f"Rows requiring review: {summary.flagged_rows}")
        for flag, count in summary.flag_counts.items():
            print(f"  {flag}: {count}")
    else:
        print("Rows requiring review: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
