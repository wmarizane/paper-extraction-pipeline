"""Conservative post-processing standardizer for consensus LCCC JSON output.

Reads ``*_consensus.json`` files whose records are stored under
``extracted_data.conditions`` and writes corresponding ``*_standardized.json``
files. Source JSON files are never modified. Every transformed field retains
its original value in a sibling ``*_raw`` field, and every potentially
ambiguous transformation receives a durable status rather than relying only
on a runtime warning.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().casefold() in {
        "",
        "null",
        "none",
        "nan",
        "n/a",
        "na",
    }


def _display_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", text).casefold()


def _alphanumeric_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = (
        text.replace("µ", "u")
        .replace("μ", "u")
        .replace("×", "x")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )
    text = text.encode("ascii", errors="ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def _capture_raw(cond: Dict[str, Any], field: str) -> Any:
    """Capture once, then always return the authoritative original value."""
    raw_field = f"{field}_raw"
    if raw_field not in cond:
        cond[raw_field] = copy.deepcopy(cond.get(field))
    return copy.deepcopy(cond.get(raw_field))


def _append_unique_text(existing: Any, value: str) -> str:
    values = [part.strip() for part in str(existing or "").split(",") if part.strip()]
    present = {part.casefold() for part in values}
    if value and value.casefold() not in present:
        values.append(value)
    return ", ".join(values)


def _finite_number(value: float) -> bool:
    return math.isfinite(value)


# =============================================================================
# POLYMER / CRITICAL-COMPONENT STANDARDIZATION
# =============================================================================


_POLYMER_CANONICAL_TO_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Poly(ethylene glycol)": (
        "PEG",
        "PEO",
        "EO",
        "mPEG",
        "PEG-MME",
        "PEG-DME",
        "MeO-PEG-DME",
        "MeO-PEG",
        "Polyethylene glycol",
        "Poly(ethylene glycol)",
        "Polyethylene oxide",
        "Poly(ethylene oxide)",
        "Polyoxyethylene",
        "PEG backbone",
        "PEO backbone",
        "ethylene oxide backbone",
        "PEG block",
        "PEO block",
        "EO block",
        "ethylene oxide block",
        "Oxyethylene unit",
        "Ethylene oxide (EO) unit",
        "Polyethylene oxide (EO) unit",
        "Polyoxyethylene chain",
        "PEG monomethyl ethers",
        "PEO-MA macromonomer",
        "PEO-MA macromonomers",
    ),
    "Poly(propylene glycol)": (
        "PPG",
        "PPO",
        "PO",
        "Polypropylene glycol",
        "Poly(propylene glycol)",
        "Polypropylene oxide",
        "Poly(propylene oxide)",
        "Polyoxypropylene",
        "PPG backbone",
        "PPO backbone",
        "PPG block",
        "PPO block",
        "PO block",
    ),
    "Poly(L-lactide)": (
        "PLLA",
        "Poly(L-lactic acid)",
        "Poly-L-lactide",
    ),
}


_POLYMER_AMBIGUOUS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    _alphanumeric_key(value): (
        ("Poly(L-lactide)",),
        "The reported polymer does not establish L stereochemistry and "
        "cannot safely be assigned to Poly(L-lactide).",
    )
    for value in (
        "PLA",
        "Poly(lactide)",
        "Polylactide",
        "Poly(lactic acid)",
    )
}


_POLYMER_DESCRIPTOR_RE = re.compile(
    r"^(?:linear|cyclic|ring|star|graft|branched|isotactic|"
    r"syndiotactic|atactic)\s+",
    re.IGNORECASE,
)


def _build_polymer_lookups() -> Tuple[Dict[str, str], Dict[str, str]]:
    exact: Dict[str, str] = {}
    targets: Dict[str, set[str]] = defaultdict(set)

    for canonical, aliases in _POLYMER_CANONICAL_TO_ALIASES.items():
        exact_key = _display_key(canonical)
        if not exact_key:
            raise ValueError(f"Empty canonical polymer name: {canonical!r}")
        previous = exact.get(exact_key)
        if previous is not None and previous != canonical:
            raise ValueError(
                f"Canonical polymer collision: {previous!r} and {canonical!r}"
            )
        exact[exact_key] = canonical
        for value in (canonical, *aliases):
            key = _alphanumeric_key(value)
            if key:
                targets[key].add(canonical)

    collisions = {
        key: tuple(sorted(values, key=str.casefold))
        for key, values in targets.items()
        if len(values) > 1
    }
    if collisions:
        raise ValueError(f"Polymer alias collisions: {collisions}")

    aliases = {key: next(iter(values)) for key, values in targets.items()}
    overlap = set(aliases) & set(_POLYMER_AMBIGUOUS)
    if overlap:
        raise ValueError(f"Polymer rule overlap: {sorted(overlap)}")
    return exact, aliases


_POLYMER_EXACT_LOOKUP, _POLYMER_ALIAS_LOOKUP = _build_polymer_lookups()


def _extract_polymer_candidates(raw: str) -> List[str]:
    """Return conservative candidates without splitting multi-polymer analytes."""
    text = str(raw).strip()
    if not text:
        return []

    candidates: List[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        item = re.sub(r"\s+", " ", candidate).strip(" ,;")
        if not item:
            return
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            candidates.append(item)

        descriptor_free = _POLYMER_DESCRIPTOR_RE.sub("", item).strip()
        descriptor_key = descriptor_free.casefold()
        if descriptor_free and descriptor_key not in seen:
            seen.add(descriptor_key)
            candidates.append(descriptor_free)

    add(text)

    # Only separate a final explanatory parenthesis preceded by whitespace.
    # This handles "Poly(ethylene oxide) (PEO)" while leaving formulas such as
    # "H(EO)x(PO)y(EO)xOH" intact.
    trailing_parenthetical = re.fullmatch(r"(.+?)\s+\(([^()]*)\)\s*", text)
    if trailing_parenthetical:
        add(trailing_parenthetical.group(1))
        add(trailing_parenthetical.group(2))

    # Recognize molecular-weight suffixes without turning a copolymer formula
    # into one of its constituent blocks.
    suffix_re = re.compile(
        r"\s+(?:(?:M[nw]?|MW)\s*[=:]?\s*)?"
        r"\d+(?:\.\d+)?\s*(?:kDa|Da|kg/?mol|g/?mol|k)?\s*$",
        re.IGNORECASE,
    )
    for candidate in tuple(candidates):
        stripped = suffix_re.sub("", candidate).strip()
        if stripped and stripped != candidate:
            add(stripped)

    return candidates


def _classify_polymer(value: Any) -> Tuple[Optional[str], str, List[str], str]:
    if not isinstance(value, str) or not value.strip():
        return None, "unmapped", [], "Critical component was not reported as text."

    ambiguous_match: Optional[Tuple[Tuple[str, ...], str]] = None
    for candidate in _extract_polymer_candidates(value):
        exact = _POLYMER_EXACT_LOOKUP.get(_display_key(candidate))
        if exact is not None:
            return exact, "exact", [], ""

        key = _alphanumeric_key(candidate)
        canonical = _POLYMER_ALIAS_LOOKUP.get(key)
        if canonical is not None:
            return canonical, "alias", [], ""

        if ambiguous_match is None:
            ambiguous_match = _POLYMER_AMBIGUOUS.get(key)

    if ambiguous_match is not None:
        candidates, reason = ambiguous_match
        return None, "ambiguous", list(candidates), reason

    return (
        None,
        "unmapped",
        [],
        "No conservative critical-component rule exists for this value.",
    )


def _normalize_polymer_fields(cond: Dict[str, Any]) -> None:
    analyte_raw = _capture_raw(cond, "analyte_polymer")
    # Do not collapse a full analyte or a comma-separated multi-analyte field.
    cond["analyte_polymer"] = copy.deepcopy(analyte_raw)

    source = _capture_raw(cond, "critical_component")
    cond.pop("comparison_polymer_standardization_candidates", None)
    cond.pop("comparison_polymer_standardization_reason", None)

    if _is_nullish(source):
        cond["critical_component"] = None
        cond["comparison_polymer"] = None
        cond["comparison_polymer_standardization_status"] = "unmapped"
        cond["comparison_polymer_standardization_reason"] = (
            "Critical component was not reported."
        )
        return

    canonical, status, candidates, reason = _classify_polymer(source)
    cond["critical_component"] = canonical if canonical is not None else source
    cond["comparison_polymer"] = canonical
    cond["comparison_polymer_standardization_status"] = status

    if candidates:
        cond["comparison_polymer_standardization_candidates"] = candidates
    if reason:
        cond["comparison_polymer_standardization_reason"] = reason


# =============================================================================
# SOLVENT AND ADDITIVE STANDARDIZATION
# =============================================================================


_SOLVENT_CANONICAL_TO_ALIASES: Dict[str, Tuple[str, ...]] = {
    "1,4-Dioxane": ("1,4-Dioxan", "Dioxane"),
    "1-Propanol": ("n-Propanol", "Propan-1-ol", "n-Propyl alcohol"),
    "2-Propanol": ("IPA", "Isopropanol", "Isopropyl alcohol", "i-PrOH"),
    "Acetone": ("Propanone", "2-Propanone"),
    "Acetonitrile": ("ACN", "CH3CN", "MeCN", "Methyl cyanide"),
    "Carbon Dioxide": ("CO2", "scCO2", "Supercritical carbon dioxide"),
    "Chloroform": ("CHCl3", "Trichloromethane"),
    "Cyclohexane": ("c-Hexane", "c-Hexan", "Cyclohexan"),
    "Cyclohexanone": (),
    "Decalin": ("Decahydronaphthalene",),
    "Deuterated Acetone": ("Acetone-d6", "d6-Acetone"),
    "Dichloromethane": ("DCM", "Methylene chloride", "CH2Cl2"),
    "Dimethoxyethane": (
        "DME",
        "Glyme",
        "Monoglyme",
        "Dimethyl glycol",
        "Ethylene glycol dimethyl ether",
        "Dimethyl cellosolve",
    ),
    "Dimethylacetamide": ("DMAc", "DMAC", "N,N-Dimethylacetamide"),
    "Dimethylformamide": (
        "DMF",
        "N,N-Dimethylformamide",
        "N,N-Dimethylformamide (DMF)",
        "Dimethyl Formamide",
    ),
    "Ethyl Acetate": ("EtOAc", "Ethyl ethanoate", "Ethylacetat"),
    "Heptane": ("n-Heptane",),
    "Hexane": ("n-Hexane",),
    "Methanol": ("MeOH", "CH3OH", "Methyl alcohol"),
    "Methyl Ethyl Ketone": (
        "MEK",
        "Butanone",
        "2-Butanone",
        "Methylethylketon",
        "Methyl ethyl ketone (MEK)",
    ),
    "Tetrachloromethane": ("CCl4", "Carbon tetrachloride"),
    "Tetrahydrofuran": ("THF", "Tetrahydrofuran (THF)", "Tetrahydrofurane"),
    "Toluene": ("Toluol", "Methylbenzene"),
    "Water": ("H2O", "Deionized water", "DI water", "Aqueous buffer", "Aqueous"),
    "Xylene": ("Xylenes", "Xylol"),
}


_ADDITIVE_CANONICAL_TO_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Acetic Acid": ("AcOH", "HOAc", "Ethanoic acid"),
    "Formic Acid": ("FA", "HCOOH", "Methanoic acid"),
    "Trifluoroacetic Acid": ("TFA",),
    "Triethylamine": ("TEA", "Et3N"),
    "Trimethylamine": ("TMA", "Me3N"),
}


_NEAR_CRITICAL_RE = re.compile(
    r"\s*\(?\s*near[\s_-]*crit(?:ical)?\s*\)?\s*",
    re.IGNORECASE,
)
_CONCENTRATION_NUMBER = (
    r"(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:\s*(?:to|[-–—−])\s*"
    r"(?:(?:\d+(?:\.\d*)?)|(?:\.\d+)))?"
)
_CONCENTRATION_UNIT = (
    r"(?:mmol\s*/?\s*L|mol\s*/?\s*L|mg\s*/?\s*mL|"
    r"mg\s*/?\s*L|g\s*/?\s*L|mol(?:ar)?|mM|M|"
    r"%\s*(?:[wv]\s*/\s*[wv])?)"
)
_CONCENTRATION_PREFIX_RE = re.compile(
    rf"^\s*(?P<concentration>{_CONCENTRATION_NUMBER}\s*"
    rf"{_CONCENTRATION_UNIT})\s+",
    re.IGNORECASE,
)


def _build_name_lookup(
    registry: Dict[str, Tuple[str, ...]],
    label: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    exact: Dict[str, str] = {}
    targets: Dict[str, set[str]] = defaultdict(set)
    for canonical, aliases in registry.items():
        exact[_display_key(canonical)] = canonical
        for value in (canonical, *aliases):
            key = _alphanumeric_key(value)
            if key:
                targets[key].add(canonical)

    collisions = {
        key: tuple(sorted(values, key=str.casefold))
        for key, values in targets.items()
        if len(values) > 1
    }
    if collisions:
        raise ValueError(f"{label} alias collisions: {collisions}")
    aliases = {key: next(iter(values)) for key, values in targets.items()}
    return exact, aliases


_SOLVENT_EXACT_LOOKUP, _SOLVENT_ALIAS_LOOKUP = _build_name_lookup(
    _SOLVENT_CANONICAL_TO_ALIASES,
    "Solvent",
)
_ADDITIVE_EXACT_LOOKUP, _ADDITIVE_ALIAS_LOOKUP = _build_name_lookup(
    _ADDITIVE_CANONICAL_TO_ALIASES,
    "Additive",
)


def _resolve_registered_name(
    value: Any,
    exact_lookup: Dict[str, str],
    alias_lookup: Dict[str, str],
) -> Tuple[Optional[str], str]:
    if not isinstance(value, str) or not value.strip():
        return None, "unmapped"

    text = re.sub(r"\s+", " ", value).strip(" ,;")
    exact = exact_lookup.get(_display_key(text))
    if exact is not None:
        return exact, "exact"

    candidate_texts = [text]
    trailing = re.fullmatch(r"(.+?)\s+\(([^()]*)\)\s*", text)
    if trailing:
        candidate_texts.extend((trailing.group(1), trailing.group(2)))

    concentration_free = _CONCENTRATION_PREFIX_RE.sub("", text).strip()
    if concentration_free and concentration_free != text:
        candidate_texts.append(concentration_free)

    seen: set[str] = set()
    for candidate in candidate_texts:
        key = _alphanumeric_key(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        canonical = alias_lookup.get(key)
        if canonical is not None:
            return canonical, "alias"
    return None, "unmapped"


def _resolve_solvent_name(value: Any) -> Tuple[Any, Optional[str], str]:
    if not isinstance(value, str):
        return value, None, "unmapped"
    text = value.strip()
    qualifier = "near crit" if _NEAR_CRITICAL_RE.search(text) else None
    base = _NEAR_CRITICAL_RE.sub(" ", text).strip(" ,;")
    canonical, status = _resolve_registered_name(
        base,
        _SOLVENT_EXACT_LOOKUP,
        _SOLVENT_ALIAS_LOOKUP,
    )
    return canonical if canonical is not None else base, qualifier, status


def _resolve_additive(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    concentration_match = _CONCENTRATION_PREFIX_RE.match(text)
    concentration = (
        concentration_match.group("concentration") if concentration_match else None
    )
    canonical, status = _resolve_registered_name(
        text,
        _ADDITIVE_EXACT_LOOKUP,
        _ADDITIVE_ALIAS_LOOKUP,
    )
    if canonical is None:
        return None
    return {
        "raw": text,
        "name": canonical,
        "concentration": concentration,
        "standardization_status": status,
    }


_INTERNAL_COMMA = "\uf000"
_INTERNAL_SLASH = "\uf001"


def _protect_internal_chemical_commas(text: str) -> str:
    protected = re.sub(
        r"(?i)\b([A-Z])\s*,\s*([A-Z])(?=\s*[-–—− ])",
        rf"\1{_INTERNAL_COMMA}\2",
        text,
    )
    protected = re.sub(
        r"(?i)\b(1)\s*,\s*(4)(?=\s*[-–—−]?\s*dioxan)",
        rf"\1{_INTERNAL_COMMA}\2",
        protected,
    )
    # A slash inside a concentration unit is not a solvent delimiter.
    # Ordinary solvent mixtures such as THF/DMF remain unprotected.
    protected = re.sub(
        r"(?i)\b(mmol|mol|mg|g)\s*/\s*(mL|L)\b",
        lambda match: f"{match.group(1)}{_INTERNAL_SLASH}{match.group(2)}",
        protected,
    )
    protected = re.sub(
        r"(?i)(?<![A-Za-z])([wv])\s*/\s*([wv])(?![A-Za-z])",
        lambda match: f"{match.group(1)}{_INTERNAL_SLASH}{match.group(2)}",
        protected,
    )
    return protected


def _restore_internal_chemical_commas(text: str) -> str:
    return text.replace(_INTERNAL_COMMA, ",").replace(_INTERNAL_SLASH, "/")


def _split_top_level_delimiters(text: str) -> List[str]:
    """Split recursively while protecting parentheses and chemical commas."""
    protected = _protect_internal_chemical_commas(text)
    parts: List[str] = []
    buffer: List[str] = []
    depth = 0
    index = 0

    def flush() -> None:
        item = _restore_internal_chemical_commas("".join(buffer)).strip()
        buffer.clear()
        if item:
            parts.append(item)

    while index < len(protected):
        character = protected[index]
        if character == "(":
            depth += 1
            buffer.append(character)
            index += 1
            continue
        if character == ")":
            depth = max(0, depth - 1)
            buffer.append(character)
            index += 1
            continue

        if depth == 0:
            word_and = re.match(r"\s+and\s+", protected[index:], re.IGNORECASE)
            if word_and:
                flush()
                index += word_and.end()
                continue
            if character in ",;/&+":
                flush()
                index += 1
                continue

        buffer.append(character)
        index += 1

    flush()
    return parts


def _split_unresolved_dash_token(token: str) -> List[str]:
    """Split dash mixtures only when every resulting component is known."""
    if not re.search(r"[-–—]", token):
        return [token]
    if re.fullmatch(r"water\s+with\s+.+", token, re.IGNORECASE):
        return [token]
    if _resolve_additive(token) is not None or _resolve_salt_additive(token) is not None:
        return [token]
    _, _, status = _resolve_solvent_name(token)
    if status != "unmapped":
        return [token]

    dash_positions = [match.start() for match in re.finditer(r"[-–—]", token)]
    memo: Dict[int, Optional[List[str]]] = {}

    def known_component(value: str) -> bool:
        if _resolve_additive(value) is not None or _resolve_salt_additive(value) is not None:
            return True
        return _resolve_solvent_name(value)[2] != "unmapped"

    def partition(start: int) -> Optional[List[str]]:
        if start in memo:
            return memo[start]
        boundaries = [position for position in dash_positions if position >= start]
        boundaries.append(len(token))
        for boundary in boundaries:
            candidate = token[start:boundary].strip()
            if not candidate or not known_component(candidate):
                continue
            if boundary == len(token):
                memo[start] = [candidate]
                return memo[start]
            remainder = partition(boundary + 1)
            if remainder:
                memo[start] = [candidate, *remainder]
                return memo[start]
        memo[start] = None
        return None

    resolved = partition(0)
    return resolved if resolved and len(resolved) > 1 else [token]


_EMBEDDED_RATIO_PATTERN = (
    r"\d+(?:\.\d+)?\s*(?::|/)\s*\d+(?:\.\d+)?"
    r"(?:\s*(?::|/)\s*\d+(?:\.\d+)?)*"
)
_EMBEDDED_UNIT_PATTERN = (
    r"(?:"
    r"(?:v|w|vol|wt|volume|weight)\s*(?:/|by)\s*"
    r"(?:v|w|vol|wt|volume|weight)(?:\s*/\s*(?:v|w|vol|wt))?"
    r"|%\s*(?:by\s+)?(?:volume|weight)"
    r"|(?:vol|volume|wt|weight)\s*%"
    r"|[vw]\s*%"
    r")"
)
_EMBEDDED_BODY_SEPARATOR = r"(?:\s*=\s*|\s+)"

_EMBEDDED_RATIO_RE = re.compile(
    rf"^(?P<body>.+?){_EMBEDDED_BODY_SEPARATOR}"
    rf"(?P<ratio>{_EMBEDDED_RATIO_PATTERN})\s*"
    rf"(?:\((?P<units>[^()]*)\))?\s*$",
    re.IGNORECASE,
)

_EMBEDDED_RATIO_PAREN_RE = re.compile(
    rf"^(?P<body>.+?)\s*(?:=\s*)?\(\s*"
    rf"(?P<ratio>{_EMBEDDED_RATIO_PATTERN})\s*"
    rf"(?:,?\s*(?P<units>{_EMBEDDED_UNIT_PATTERN}))?\s*\)\s*$",
    re.IGNORECASE,
)

_EMBEDDED_RATIO_BARE_UNITS_RE = re.compile(
    rf"^(?P<body>.+?){_EMBEDDED_BODY_SEPARATOR}"
    rf"(?P<ratio>{_EMBEDDED_RATIO_PATTERN})\s*"
    rf"(?P<units>{_EMBEDDED_UNIT_PATTERN})\s*$",
    re.IGNORECASE,
)


def _last_top_level_group(text: str) -> str:
    protected = _protect_internal_chemical_commas(text)
    depth = 0
    last_boundary = -1
    for index, character in enumerate(protected):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and character in ",;":
            last_boundary = index
    return _restore_internal_chemical_commas(protected[last_boundary + 1 :]).strip()


def _extract_embedded_ratio_annotation(
    text: str,
) -> Tuple[str, Optional[Dict[str, str]]]:
    match = None
    for pattern in (
        _EMBEDDED_RATIO_PAREN_RE,
        _EMBEDDED_RATIO_BARE_UNITS_RE,
        _EMBEDDED_RATIO_RE,
    ):
        match = pattern.fullmatch(text.strip())
        if match is not None:
            break
    if match is None:
        return text, None
    body = match.group("body").strip(" ,;")
    return body, {
        "ratio_raw": re.sub(r"\s+", "", match.group("ratio")),
        "units_raw": (match.group("units") or "").strip(),
        "scope_raw": _last_top_level_group(body),
    }


def _canonicalize_scope(scope_raw: str) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for unsplit_token in _split_top_level_delimiters(scope_raw):
        for token in _split_unresolved_dash_token(unsplit_token):
            canonical, _, status = _resolve_solvent_name(token)
            if status == "unmapped":
                return []
            canonical_text = str(canonical)
            key = _alphanumeric_key(canonical_text)
            if key not in seen:
                seen.add(key)
                result.append(canonical_text)
    return result


_AQUEOUS_SALT_NAMES = {
    "nacl": "Sodium chloride",
    "sodium chloride": "Sodium chloride",
    "kcl": "Potassium chloride",
    "potassium chloride": "Potassium chloride",
    "lithium chloride": "Lithium chloride",
    "licl": "Lithium chloride",
    "ammonium acetate": "Ammonium acetate",
}
_AQUEOUS_CONCENTRATION_RE = re.compile(
    rf"(?<![A-Za-z0-9.]){_CONCENTRATION_NUMBER}\s*"
    rf"{_CONCENTRATION_UNIT}(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _resolve_salt_additive(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    base = _CONCENTRATION_PREFIX_RE.sub("", text).strip()
    base_key = _alphanumeric_key(base)
    for token, canonical in _AQUEOUS_SALT_NAMES.items():
        if base_key == _alphanumeric_key(token):
            concentration = _AQUEOUS_CONCENTRATION_RE.search(text)
            return {
                "raw": text,
                "name": canonical,
                "concentration": concentration.group(0) if concentration else None,
                "standardization_status": (
                    "exact" if _display_key(base) == _display_key(canonical) else "alias"
                ),
            }
    return None


def _record_aqueous_modifier(cond: Dict[str, Any], modifier: str) -> None:
    aqueous = cond.get("aqueous_parameters")
    if not isinstance(aqueous, dict):
        aqueous = {}
        cond["aqueous_parameters"] = aqueous

    modifier_lower = modifier.casefold()
    for token, canonical in _AQUEOUS_SALT_NAMES.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", modifier_lower):
            aqueous["salt_added"] = True
            aqueous["salt_type"] = _append_unique_text(
                aqueous.get("salt_type"),
                canonical,
            )
            concentration = _AQUEOUS_CONCENTRATION_RE.search(modifier)
            if concentration:
                aqueous["salt_concentration"] = _append_unique_text(
                    aqueous.get("salt_concentration"),
                    concentration.group(0),
                )
            return

    aqueous["pH_modifier"] = _append_unique_text(
        aqueous.get("pH_modifier"),
        modifier,
    )


def _normalize_aqueous_parameters_value(
    raw: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Normalize schema-native aqueous salt fields and reclassify modifiers."""
    if _is_nullish(raw):
        return {}, [], "unmapped"
    if not isinstance(raw, dict):
        return {}, [], "invalid"

    aqueous = copy.deepcopy(raw)
    additives: List[Dict[str, Any]] = []

    flag_raw = aqueous.get("salt_added")
    if isinstance(flag_raw, bool):
        salt_added: Optional[bool] = flag_raw
    elif isinstance(flag_raw, str) and flag_raw.strip().casefold() in {"true", "yes", "1"}:
        salt_added = True
    elif isinstance(flag_raw, str) and flag_raw.strip().casefold() in {"false", "no", "0"}:
        salt_added = False
    else:
        salt_added = None

    concentration_raw = aqueous.get("salt_concentration")
    normalized_concentration: Optional[str] = None
    if not _is_nullish(concentration_raw):
        concentration_text = str(concentration_raw).strip()
        concentration_match = _AQUEOUS_CONCENTRATION_RE.fullmatch(concentration_text)
        if concentration_match is not None:
            normalized_concentration = concentration_match.group(0).strip()
            aqueous["salt_concentration"] = normalized_concentration
            aqueous["salt_concentration_standardization_status"] = (
                "exact"
                if _display_key(concentration_text) == _display_key(normalized_concentration)
                else "alias"
            )
        else:
            aqueous["salt_concentration"] = concentration_raw
            aqueous["salt_concentration_standardization_status"] = "unmapped"
    else:
        aqueous["salt_concentration"] = None
        aqueous["salt_concentration_standardization_status"] = "unmapped"

    salt_type_raw = aqueous.get("salt_type")
    if _is_nullish(salt_type_raw):
        aqueous["salt_type"] = None
        aqueous["salt_added"] = bool(salt_added) if salt_added is not None else False
        aqueous["salt_type_standardization_status"] = (
            "ambiguous" if salt_added is True else "unmapped"
        )
        overall = "ambiguous" if salt_added is True or normalized_concentration else "exact"
        return aqueous, additives, overall

    salt_text = str(salt_type_raw).strip()
    salt = _resolve_salt_additive(salt_text)
    if salt is not None:
        aqueous["salt_added"] = True
        aqueous["salt_type"] = salt["name"]
        aqueous["salt_type_standardization_status"] = salt["standardization_status"]
        if normalized_concentration is None and salt.get("concentration"):
            aqueous["salt_concentration"] = salt["concentration"]
            aqueous["salt_concentration_standardization_status"] = "alias"
        overall = "exact" if salt["standardization_status"] == "exact" else "alias"
        if salt_added is False:
            overall = "alias"
        return aqueous, additives, overall

    additive = _resolve_additive(salt_text)
    if additive is not None:
        additive = copy.deepcopy(additive)
        if normalized_concentration is not None:
            additive["concentration"] = normalized_concentration
        additives.append(additive)
        modifier = " ".join(
            part for part in (normalized_concentration, str(additive["name"])) if part
        )
        aqueous["pH_modifier"] = _append_unique_text(
            aqueous.get("pH_modifier"),
            modifier,
        )
        aqueous["salt_added"] = False
        aqueous["salt_type"] = None
        aqueous["salt_concentration"] = None
        aqueous["salt_type_standardization_status"] = "reclassified_additive"
        aqueous["salt_concentration_standardization_status"] = "reclassified_additive"
        return aqueous, additives, "reclassified_additive"

    aqueous["salt_added"] = bool(salt_added) if salt_added is not None else True
    aqueous["salt_type"] = salt_type_raw
    aqueous["salt_type_standardization_status"] = "unmapped"
    return aqueous, additives, "unmapped"


def _normalize_solvents(cond: Dict[str, Any]) -> None:
    raw_aqueous = _capture_raw(cond, "aqueous_parameters")
    aqueous, initial_additives, aqueous_status = _normalize_aqueous_parameters_value(
        raw_aqueous
    )
    cond["aqueous_parameters"] = aqueous
    cond["aqueous_parameters_standardization_status"] = aqueous_status

    raw_solvents = _capture_raw(cond, "mobile_phase_solvents")
    cond["mobile_phase_solvent_details"] = []
    cond["mobile_phase_solvent_qualifiers"] = []
    cond["mobile_phase_additives"] = copy.deepcopy(initial_additives)
    cond["mobile_phase_component_order"] = []
    cond.pop("mobile_phase_embedded_ratio_annotations", None)
    cond.pop("mobile_phase_ratio_scope", None)

    if _is_nullish(raw_solvents):
        cond["mobile_phase_solvents"] = None
        cond["mobile_phase_solvent_standardization_status"] = "unmapped"
        return

    raw_items: Sequence[Any]
    if isinstance(raw_solvents, (list, tuple)):
        raw_items = raw_solvents
    else:
        raw_items = [raw_solvents]

    normalized: List[str] = []
    qualifiers: List[Optional[str]] = []
    details: List[Dict[str, Any]] = []
    additives: List[Dict[str, Any]] = copy.deepcopy(initial_additives)
    annotations: List[Dict[str, str]] = []
    component_order: List[Dict[str, str]] = []
    standalone_modifier_texts: List[str] = []
    seen: set[str] = set()
    seen_components: set[Tuple[str, str]] = set()

    def add_solvent(raw_token: str, canonical: Any, qualifier: Optional[str], status: str) -> None:
        canonical_text = str(canonical).strip()
        if not canonical_text:
            return
        details.append(
            {
                "raw": raw_token,
                "solvent": canonical_text if status != "unmapped" else None,
                "standardization_status": status,
                "qualifier": qualifier,
            }
        )
        output_text = canonical_text
        key = _alphanumeric_key(output_text)
        if key and key not in seen:
            seen.add(key)
            normalized.append(output_text)
            qualifiers.append(qualifier)
            component_key = ("solvent", key)
            if component_key not in seen_components:
                seen_components.add(component_key)
                component_order.append(
                    {"component": output_text, "component_type": "solvent"}
                )

    for raw_item in raw_items:
        if _is_nullish(raw_item):
            continue
        text = str(raw_item).strip()
        text_without_ratio, annotation = _extract_embedded_ratio_annotation(text)
        if annotation is not None:
            ratio_component_count = len(
                re.findall(r"\d+(?:\.\d+)?", annotation["ratio_raw"])
            )
            full_scope = _canonicalize_scope(text_without_ratio)
            fallback_scope = _canonicalize_scope(annotation["scope_raw"])
            if full_scope and len(full_scope) == ratio_component_count:
                annotation["scope_raw"] = text_without_ratio
            elif fallback_scope and len(fallback_scope) == ratio_component_count:
                pass
            elif full_scope:
                # Preserve the full scope so the ratio parser records a
                # component-count ambiguity instead of silently dropping a
                # solvent.
                annotation["scope_raw"] = text_without_ratio
            annotations.append(annotation)

        for unsplit_token in _split_top_level_delimiters(text_without_ratio):
            for token in _split_unresolved_dash_token(unsplit_token):
                water_with = re.fullmatch(r"water\s+with\s+(.+)", token, re.IGNORECASE)
                if water_with:
                    canonical, qualifier, status = _resolve_solvent_name("Water")
                    add_solvent("Water", canonical, qualifier, status)
                    modifier = water_with.group(1).strip()
                    additive = _resolve_additive(modifier) or _resolve_salt_additive(modifier)
                    if additive is None:
                        additive = {
                            "raw": modifier,
                            "name": None,
                            "concentration": (
                                _CONCENTRATION_PREFIX_RE.match(modifier).group("concentration")
                                if _CONCENTRATION_PREFIX_RE.match(modifier)
                                else None
                            ),
                            "standardization_status": "unmapped",
                        }
                    additives.append(additive)
                    _record_aqueous_modifier(cond, modifier)
                    continue

                additive = _resolve_additive(token) or _resolve_salt_additive(token)
                if additive is not None:
                    additives.append(additive)
                    standalone_modifier_texts.append(token)
                    if additive.get("concentration") is None:
                        component_name = str(additive["name"])
                        component_key = ("additive", _alphanumeric_key(component_name))
                        if component_key not in seen_components:
                            seen_components.add(component_key)
                            component_order.append(
                                {
                                    "component": component_name,
                                    "component_type": "additive",
                                }
                            )
                    continue

                canonical, qualifier, status = _resolve_solvent_name(token)
                add_solvent(token, canonical, qualifier, status)

    cond["mobile_phase_solvents"] = normalized
    cond["mobile_phase_solvent_qualifiers"] = qualifiers
    cond["mobile_phase_solvent_details"] = details
    deduplicated_additives: List[Dict[str, Any]] = []
    primary_additive_index: Dict[str, int] = {}
    seen_additives: set[Tuple[str, str]] = set()
    for additive in additives:
        name_key = _alphanumeric_key(additive.get("name") or additive.get("raw"))
        concentration_key = _display_key(additive.get("concentration") or "")
        key = (name_key, concentration_key)
        if key in seen_additives:
            continue
        seen_additives.add(key)

        existing_index = primary_additive_index.get(name_key)
        if existing_index is None:
            primary_additive_index[name_key] = len(deduplicated_additives)
            deduplicated_additives.append(copy.deepcopy(additive))
            continue

        existing = deduplicated_additives[existing_index]
        existing_concentration = existing.get("concentration")
        new_concentration = additive.get("concentration")
        if _is_nullish(existing_concentration) and not _is_nullish(new_concentration):
            existing["concentration"] = new_concentration
            continue
        if not _is_nullish(existing_concentration) and _is_nullish(new_concentration):
            continue

        # Distinct non-null concentrations may describe genuinely different
        # mobile-phase conditions, so preserve both instead of merging them.
        deduplicated_additives.append(copy.deepcopy(additive))
    cond["mobile_phase_additives"] = deduplicated_additives
    cond["mobile_phase_component_order"] = component_order

    if _alphanumeric_key("Water") in {
        _alphanumeric_key(solvent) for solvent in normalized
    }:
        for modifier in standalone_modifier_texts:
            registered_modifier = _resolve_additive(modifier)
            existing_modifier_text = str(
                cond.get("aqueous_parameters", {}).get("pH_modifier") or ""
            )
            if (
                registered_modifier is not None
                and _alphanumeric_key(registered_modifier["name"])
                in _alphanumeric_key(existing_modifier_text)
            ):
                continue
            _record_aqueous_modifier(cond, modifier)

    mapped_statuses = [detail["standardization_status"] for detail in details]
    if not mapped_statuses or any(status == "unmapped" for status in mapped_statuses):
        overall_status = "unmapped"
    elif any(status == "alias" for status in mapped_statuses):
        overall_status = "alias"
    else:
        overall_status = "exact"
    cond["mobile_phase_solvent_standardization_status"] = overall_status

    if annotations:
        cond["mobile_phase_embedded_ratio_annotations"] = annotations
        unique_annotations = {
            (item["ratio_raw"], item["units_raw"], item["scope_raw"])
            for item in annotations
        }
        if len(unique_annotations) == 1:
            annotation = annotations[0]
            scope = _canonicalize_scope(annotation["scope_raw"])
            existing_ratio_raw = cond.get("mobile_phase_ratio_raw")
            existing_ratio_value = (
                existing_ratio_raw
                if not _is_nullish(existing_ratio_raw)
                else cond.get("mobile_phase_ratio")
            )
            embedded_marker_matches = (
                cond.get("mobile_phase_ratio_source") == "embedded"
                and re.sub(r"\s+", "", str(existing_ratio_raw))
                == re.sub(r"\s+", "", annotation["ratio_raw"])
            )
            ratio_values_match = (
                not _is_nullish(existing_ratio_value)
                and re.sub(r"\s+", "", str(existing_ratio_value)).casefold()
                == re.sub(r"\s+", "", annotation["ratio_raw"]).casefold()
            )
            existing_units_value = (
                cond.get("mobile_phase_ratio_units_raw")
                if not _is_nullish(cond.get("mobile_phase_ratio_units_raw"))
                else cond.get("mobile_phase_ratio_units")
            )
            units_agree = True
            if annotation["units_raw"] and not _is_nullish(existing_units_value):
                units_agree = (
                    _normalized_ratio_units_for_comparison(annotation["units_raw"])
                    == _normalized_ratio_units_for_comparison(existing_units_value)
                )
            matching_top_level_ratio = ratio_values_match and units_agree
            adopt_embedded_ratio = (
                _is_nullish(existing_ratio_raw)
                and _is_nullish(cond.get("mobile_phase_ratio"))
            ) or embedded_marker_matches
            if scope and (adopt_embedded_ratio or matching_top_level_ratio):
                cond["mobile_phase_ratio_scope"] = scope
            if adopt_embedded_ratio:
                cond["mobile_phase_ratio_raw"] = annotation["ratio_raw"]
                cond["mobile_phase_ratio"] = annotation["ratio_raw"]
                cond["mobile_phase_ratio_source"] = "embedded"
            if (
                adopt_embedded_ratio
                and annotation["units_raw"]
                and _is_nullish(cond.get("mobile_phase_ratio_units_raw"))
                and _is_nullish(cond.get("mobile_phase_ratio_units"))
            ):
                cond["mobile_phase_ratio_units_raw"] = annotation["units_raw"]
                cond["mobile_phase_ratio_units"] = annotation["units_raw"]


def canonicalize_solvent_list(solvents: Any) -> List[str]:
    temporary = {"mobile_phase_solvents": copy.deepcopy(solvents)}
    _normalize_solvents(temporary)
    normalized = temporary.get("mobile_phase_solvents")
    return normalized if isinstance(normalized, list) else []


# =============================================================================
# MOBILE-PHASE RATIO AND UNIT STANDARDIZATION
# =============================================================================


def _normalize_ratio_units(cond: Dict[str, Any]) -> None:
    raw = _capture_raw(cond, "mobile_phase_ratio_units")
    cond["mobile_phase_ratio_units"] = None
    cond["mobile_phase_ratio_units_standardization_status"] = "unmapped"
    cond.pop("mobile_phase_ratio_component_hint", None)

    if _is_nullish(raw):
        return

    text = unicodedata.normalize("NFKC", str(raw)).strip().casefold()
    unit_text = text
    unit_with_label = re.fullmatch(
        r"\s*([vw]\s*[/ :]\s*[vw])\s+(.+?)\s*",
        text,
        re.IGNORECASE,
    )
    if unit_with_label is not None:
        unit_text = unit_with_label.group(1)
        component, _ = _resolve_mobile_phase_component(unit_with_label.group(2))
        if component is not None:
            cond["mobile_phase_ratio_component_hint"] = component

    compact = re.sub(r"[\s()_.-]+", "", unit_text)
    compact = compact.replace(":", "/")

    exact_values = {"v/v", "w/w", "w/v", "v/w"}
    if unit_text in exact_values:
        cond["mobile_phase_ratio_units"] = unit_text
        cond["mobile_phase_ratio_units_standardization_status"] = (
            "alias" if unit_with_label is not None else "exact"
        )
        return

    mixed_weight_volume = {
        "w/v": "w/v",
        "wt/vol": "w/v",
        "weight/volume": "w/v",
        "weightbyvolume": "w/v",
        "mass/volume": "w/v",
        "massbyvolume": "w/v",
        "v/w": "v/w",
        "vol/wt": "v/w",
        "volume/weight": "v/w",
        "volumebyweight": "v/w",
        "volume/mass": "v/w",
        "volumebymass": "v/w",
    }
    volume_volume = {
        "v/v",
        "vol/vol",
        "volume/volume",
        "volumebyvolume",
        "byvolume",
        "%byvolume",
        "%v/v",
        "v%",
        "%v",
        "vol%",
        "volume%",
        "volumepercent",
        "vv",
        "v/v%",
        "%,v/v",
        "v/v/v",
    }
    weight_weight = {
        "w/w",
        "wt/wt",
        "weight/weight",
        "weightbyweight",
        "mass/mass",
        "massbymass",
        "%w/w",
        "w%",
        "%w",
        "wt%",
        "weight%",
        "ww",
    }

    if compact in mixed_weight_volume:
        canonical = mixed_weight_volume[compact]
    elif compact in volume_volume:
        canonical = "v/v"
    elif compact in weight_weight:
        canonical = "w/w"
    elif compact == "%" or re.fullmatch(r"%[a-z0-9]+/%[a-z0-9]+", compact):
        cond["mobile_phase_ratio_units_standardization_status"] = "ambiguous"
        return
    else:
        cond["mobile_phase_ratio_units_standardization_status"] = "unmapped"
        return

    cond["mobile_phase_ratio_units"] = canonical
    cond["mobile_phase_ratio_units_standardization_status"] = "alias"


def _normalized_ratio_units_for_comparison(value: Any) -> Optional[str]:
    """Return a canonical unit only when the normalizer can resolve it."""
    if _is_nullish(value):
        return None
    temporary: Dict[str, Any] = {"mobile_phase_ratio_units": value}
    _normalize_ratio_units(temporary)
    if temporary.get("mobile_phase_ratio_units_standardization_status") in {
        "exact",
        "alias",
    }:
        return temporary.get("mobile_phase_ratio_units")
    return f"unresolved:{_alphanumeric_key(value)}"


def _ratio_scope(cond: Dict[str, Any]) -> List[str]:
    scoped = cond.get("mobile_phase_ratio_scope")
    if isinstance(scoped, list) and scoped:
        return [str(value).strip() for value in scoped if str(value).strip()]
    components = cond.get("mobile_phase_component_order")
    if isinstance(components, list) and components:
        names = [
            str(item.get("component", "")).strip()
            for item in components
            if isinstance(item, dict) and str(item.get("component", "")).strip()
        ]
        if names:
            return names
    solvents = cond.get("mobile_phase_solvents")
    if isinstance(solvents, list):
        return [str(value).strip() for value in solvents if str(value).strip()]
    return []


def _set_composition(
    cond: Dict[str, Any],
    solvents: Sequence[str],
    values: Sequence[float],
) -> None:
    cleaned_values = [round(float(value), 6) for value in values]
    cond["mobile_phase_ratio_components"] = cleaned_values
    type_by_key = {
        _alphanumeric_key(item.get("component")): item.get("component_type", "solvent")
        for item in cond.get("mobile_phase_component_order", [])
        if isinstance(item, dict)
    }
    composition: List[Dict[str, Any]] = []
    for component, value in zip(solvents, cleaned_values):
        component_type = type_by_key.get(_alphanumeric_key(component), "solvent")
        item: Dict[str, Any] = {
            "component": component,
            "component_type": component_type,
            "value": value,
            "units": cond.get("mobile_phase_ratio_units"),
        }
        if component_type == "solvent":
            item["solvent"] = component
        else:
            item["additive"] = component
        composition.append(item)
    cond["mobile_phase_composition"] = composition


def _normalize_relative_values(values: Sequence[float]) -> Optional[List[float]]:
    if not values or any(not _finite_number(value) or value < 0 for value in values):
        return None
    total = sum(values)
    if total <= 0:
        return None
    if abs(total - 100.0) <= 0.01:
        return [round(value, 6) for value in values]
    return [round(value * 100.0 / total, 6) for value in values]


_NAMED_PERCENT_PREFIX_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*%\s*(?P<label>.+?)\s*$",
    re.IGNORECASE,
)
_NAMED_PERCENT_SUFFIX_RE = re.compile(
    r"^\s*(?P<label>.+?)\s*(?P<value>\d+(?:\.\d+)?)\s*%\s*$",
    re.IGNORECASE,
)


def _normalize_inline_ratio_notation(text: str) -> str:
    """Reduce inline wt/vol percent spellings to a plain percent marker."""
    return re.sub(
        r"(?<=\d)\s*(?:"
        r"wt\s*\.?\s*-?\s*%|weight\s*%|mass\s*%|"
        r"vol\s*\.?\s*-?\s*%|volume\s*%"
        r")",
        "%",
        text,
        flags=re.IGNORECASE,
    )


def _resolve_mobile_phase_component(value: str) -> Tuple[Optional[str], str]:
    solvent, _, solvent_status = _resolve_solvent_name(value)
    if solvent_status != "unmapped":
        return str(solvent), solvent_status
    additive = _resolve_additive(value)
    if additive is not None:
        return str(additive["name"]), str(additive["standardization_status"])
    return None, "unmapped"


def _split_named_ratio_clauses(text: str) -> List[str]:
    protected = _protect_internal_chemical_commas(text)
    protected = re.sub(r"\s*,?\s+and\s+", ",", protected, flags=re.IGNORECASE)
    clauses = [
        _restore_internal_chemical_commas(part).strip()
        for part in re.split(r"\s*[,;:]\s*", protected)
        if part.strip()
    ]
    return clauses


def _parse_named_percentages(text: str) -> Optional[List[Tuple[str, float]]]:
    # A colon between two named percentage clauses is a clause separator, not
    # a numeric ratio separator.
    clauses = _split_named_ratio_clauses(text)
    if not clauses:
        return None

    parsed: List[Tuple[str, float]] = []
    for clause in clauses:
        match = _NAMED_PERCENT_PREFIX_RE.fullmatch(clause)
        if match is None:
            match = _NAMED_PERCENT_SUFFIX_RE.fullmatch(clause)
        if match is None:
            return None

        canonical, status = _resolve_mobile_phase_component(match.group("label"))
        if canonical is None or status == "unmapped":
            return None
        value = float(match.group("value"))
        if not _finite_number(value) or not 0 <= value <= 100:
            return None
        parsed.append((canonical, value))
    return parsed


def _parse_named_binary_percentage(text: str) -> Optional[List[Tuple[str, float]]]:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%\s+(.+?)\s+in\s+(.+?)\s*", text, re.IGNORECASE)
    if match is not None:
        value = float(match.group(1))
        first, _ = _resolve_mobile_phase_component(match.group(2))
        second, _ = _resolve_mobile_phase_component(match.group(3))
        if first is not None and second is not None and 0 <= value <= 100:
            return [(first, value), (second, 100.0 - value)]

    prefix = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%\s+(.+?)\s*", text)
    if prefix is None:
        return None
    value = float(prefix.group(1))
    remainder = prefix.group(2)
    if not 0 <= value <= 100:
        return None

    # Try every dash as a potential separator and accept only when both sides
    # independently resolve. This avoids splitting the alias "n-hexane".
    matches: List[Tuple[str, str]] = []
    for dash in re.finditer(r"[-–—−]", remainder):
        left = remainder[: dash.start()].strip()
        right = remainder[dash.end() :].strip()
        first, _ = _resolve_mobile_phase_component(left)
        second, _ = _resolve_mobile_phase_component(right)
        if first is not None and second is not None:
            matches.append((first, second))
    if len(matches) == 1:
        first, second = matches[0]
        return [(first, value), (second, 100.0 - value)]
    return None


def _assign_named_percentages(
    cond: Dict[str, Any],
    scope: Sequence[str],
    named: Sequence[Tuple[str, float]],
) -> bool:
    value_by_key: Dict[str, float] = {}
    display_by_key: Dict[str, str] = {}
    for solvent, value in named:
        key = _alphanumeric_key(solvent)
        if key in value_by_key:
            return False
        value_by_key[key] = value
        display_by_key[key] = solvent

    if scope:
        scope_keys = [_alphanumeric_key(solvent) for solvent in scope]
        if any(key not in scope_keys for key in value_by_key):
            return False

        missing_keys = [key for key in scope_keys if key not in value_by_key]
        if len(missing_keys) == 1:
            complement = 100.0 - sum(value_by_key.values())
            if complement < -0.01 or complement > 100.01:
                return False
            value_by_key[missing_keys[0]] = max(0.0, min(100.0, complement))
        elif missing_keys:
            return False

        values = [value_by_key[key] for key in scope_keys]
        if abs(sum(values) - 100.0) > 0.02 and len(values) > 1:
            return False
        _set_composition(cond, scope, values)
        return True

    # Named values can still produce a structured result without a separate
    # solvent list, but no complement is invented when a component is absent.
    if len(value_by_key) != len(named):
        return False
    if len(named) < 2 or abs(sum(value_by_key.values()) - 100.0) > 0.02:
        return False
    solvents = [display_by_key[_alphanumeric_key(solvent)] for solvent, _ in named]
    values = [value for _, value in named]
    _set_composition(cond, solvents, values)
    return True


def _normalize_ratio(cond: Dict[str, Any]) -> None:
    raw = _capture_raw(cond, "mobile_phase_ratio")
    cond["mobile_phase_ratio_components"] = None
    cond["mobile_phase_ratio_min"] = None
    cond["mobile_phase_ratio_max"] = None
    cond["mobile_phase_composition"] = None
    cond.pop("mobile_phase_ratio_values_unassigned", None)
    cond.pop("mobile_phase_ratio_range_component", None)
    cond["mobile_phase_ratio_standardization_status"] = "unmapped"

    if _is_nullish(raw):
        cond["mobile_phase_ratio"] = None
        return

    cond["mobile_phase_ratio"] = raw
    text = unicodedata.normalize("NFKC", str(raw)).strip()
    text_without_units = re.sub(
        r"\s*\((?:v\s*/\s*v|w\s*/\s*w|w\s*/\s*v|v\s*/\s*w)\)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text_without_units = _normalize_inline_ratio_notation(text_without_units)
    scope = _ratio_scope(cond)

    if " to " in text_without_units.casefold():
        named_range = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*(?:to|[-–—−])\s*"
            r"(\d+(?:\.\d+)?)\s*%\s*(.+?)\s*",
            text_without_units,
            re.IGNORECASE,
        )
        if named_range is not None:
            component, _ = _resolve_mobile_phase_component(named_range.group(3))
            minimum = float(named_range.group(1))
            maximum = float(named_range.group(2))
            if component is not None and 0 <= minimum <= maximum <= 100:
                cond["mobile_phase_ratio_min"] = minimum
                cond["mobile_phase_ratio_max"] = maximum
                cond["mobile_phase_ratio_range_component"] = component
                cond["mobile_phase_ratio_standardization_status"] = "range_named"
                return
        # A multi-clause gradient such as "47% acetone to 96% acetone" is not
        # one critical composition and must not be reduced to two numbers.
        cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
        return

    named_binary = _parse_named_binary_percentage(text_without_units)
    if named_binary is not None:
        if _assign_named_percentages(cond, scope, named_binary):
            cond["mobile_phase_ratio_standardization_status"] = "parsed_named"
        else:
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
            cond["mobile_phase_ratio_values_unassigned"] = [
                {"solvent": solvent, "value": value}
                for solvent, value in named_binary
            ]
        return

    named = _parse_named_percentages(text_without_units)
    if named is not None:
        if _assign_named_percentages(cond, scope, named):
            cond["mobile_phase_ratio_standardization_status"] = "parsed_named"
        else:
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
            cond["mobile_phase_ratio_values_unassigned"] = [
                {"solvent": solvent, "value": value} for solvent, value in named
            ]
        return

    range_match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*%?\s*(?:to)\s*"
        r"(\d+(?:\.\d+)?)\s*%?\s*",
        text_without_units,
        re.IGNORECASE,
    )
    if range_match is not None:
        minimum = float(range_match.group(1))
        maximum = float(range_match.group(2))
        if 0 <= minimum <= maximum <= 100:
            cond["mobile_phase_ratio_min"] = minimum
            cond["mobile_phase_ratio_max"] = maximum
            cond["mobile_phase_ratio_standardization_status"] = "range"
        else:
            cond["mobile_phase_ratio_standardization_status"] = "invalid"
        return

    multi_hyphen = re.fullmatch(
        r"\s*\d+(?:\.\d+)?(?:\s*[-–—−]\s*\d+(?:\.\d+)?){2,}\s*",
        text_without_units,
    )
    if multi_hyphen is not None:
        values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text_without_units)]
        values = _normalize_relative_values(values)
        if values is None:
            cond["mobile_phase_ratio_standardization_status"] = "invalid"
        elif scope and len(values) != len(scope):
            cond["mobile_phase_ratio_values_unassigned"] = values
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
        else:
            cond["mobile_phase_ratio_components"] = values
            if scope:
                _set_composition(cond, scope, values)
            cond["mobile_phase_ratio_standardization_status"] = (
                "parsed" if scope else "parsed_unscoped"
            )
        return

    two_hyphen = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*[-–—−]\s*(\d+(?:\.\d+)?)\s*",
        text_without_units,
    )
    if two_hyphen is not None:
        values = [float(two_hyphen.group(1)), float(two_hyphen.group(2))]
        if len(scope) == 2 and abs(sum(values) - 100.0) <= 0.01:
            cond["mobile_phase_ratio_values_unassigned"] = values
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
        elif 0 <= values[0] <= values[1] <= 100:
            cond["mobile_phase_ratio_min"] = values[0]
            cond["mobile_phase_ratio_max"] = values[1]
            cond["mobile_phase_ratio_standardization_status"] = "range"
        else:
            cond["mobile_phase_ratio_standardization_status"] = "invalid"
        return

    explicit_ratio = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?%?)"
        r"(?:\s*([:/,])\s*(\d+(?:\.\d+)?%?))+\s*",
        text_without_units,
    )
    if explicit_ratio is not None:
        separators = re.findall(r"[:/,]", text_without_units)
        if len(set(separators)) > 1:
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
            return
        values = [
            float(number)
            for number in re.findall(r"\d+(?:\.\d+)?", text_without_units)
        ]
        values = _normalize_relative_values(values)
        if values is None:
            cond["mobile_phase_ratio_standardization_status"] = "invalid"
            return
        if scope and len(values) != len(scope):
            cond["mobile_phase_ratio_values_unassigned"] = values
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
            return
        cond["mobile_phase_ratio_components"] = values
        if scope:
            _set_composition(cond, scope, values)
        cond["mobile_phase_ratio_standardization_status"] = (
            "parsed" if scope else "parsed_unscoped"
        )
        return

    single_percent = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%\s*", text_without_units)
    if single_percent is not None:
        value = float(single_percent.group(1))
        component_hint = cond.get("mobile_phase_ratio_component_hint")
        if not 0 <= value <= 100:
            cond["mobile_phase_ratio_standardization_status"] = "invalid"
        elif isinstance(component_hint, str) and component_hint:
            if _assign_named_percentages(cond, scope, [(component_hint, value)]):
                cond["mobile_phase_ratio_standardization_status"] = "parsed_named"
            else:
                cond["mobile_phase_ratio_values_unassigned"] = [
                    {"solvent": component_hint, "value": value}
                ]
                cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
        elif len(scope) == 1:
            _set_composition(cond, scope, [value])
            cond["mobile_phase_ratio_standardization_status"] = "parsed"
        elif len(scope) == 2:
            # A complement is mathematically available, but an unlabeled value
            # does not establish which listed solvent it belongs to.
            cond["mobile_phase_ratio_values_unassigned"] = [value, 100.0 - value]
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
        else:
            cond["mobile_phase_ratio_values_unassigned"] = [value]
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
        return

    single_number = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", text_without_units)
    if single_number is not None:
        value = float(single_number.group(1))
        component_hint = cond.get("mobile_phase_ratio_component_hint")
        if not 0 <= value <= 100:
            cond["mobile_phase_ratio_standardization_status"] = "invalid"
        elif isinstance(component_hint, str) and component_hint:
            if _assign_named_percentages(cond, scope, [(component_hint, value)]):
                cond["mobile_phase_ratio_standardization_status"] = "parsed_named"
            else:
                cond["mobile_phase_ratio_values_unassigned"] = [
                    {"solvent": component_hint, "value": value}
                ]
                cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
        elif len(scope) == 1:
            _set_composition(cond, scope, [value])
            cond["mobile_phase_ratio_standardization_status"] = "parsed_assumed_percent"
        elif len(scope) == 2:
            cond["mobile_phase_ratio_values_unassigned"] = [value, 100.0 - value]
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
        else:
            cond["mobile_phase_ratio_values_unassigned"] = [value]
            cond["mobile_phase_ratio_standardization_status"] = "ambiguous"
        return

    # Deliberately reject arbitrary pairs of numbers such as column dimensions
    # or temperatures. They are not solvent compositions without ratio syntax.
    cond["mobile_phase_ratio_standardization_status"] = "unmapped"


# =============================================================================
# FLOW RATE, PORE SIZE, AND TEMPERATURE
# =============================================================================


_FLOW_UNSIGNED_NUMBER = r"(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
_FLOW_NUMBER = rf"[+-]?{_FLOW_UNSIGNED_NUMBER}"
_FLOW_VOLUME = r"(?:u\s*l|m\s*l|l)"
_FLOW_TIME = r"(?:hours?|hrs?|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)"
_FLOW_TERM_RE = re.compile(
    rf"(?P<value>{_FLOW_UNSIGNED_NUMBER})\s*(?P<volume>{_FLOW_VOLUME})\s*"
    rf"(?:"
    rf"(?:/|\.|\bper\b)\s*(?P<time_direct>{_FLOW_TIME})(?![A-Za-z])"
    rf"|(?P<time_inverse>{_FLOW_TIME})(?![A-Za-z])\s*\^?\s*-\s*1"
    rf")",
    re.IGNORECASE,
)


def _normalize_flow_text(value: str) -> str:
    return (
        unicodedata.normalize("NFKC", value)
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("−", "-")
        .replace("⁻", "-")
        .replace("·", ".")
    )


def _flow_term_factor_to_ml_per_min(match: re.Match[str]) -> float:
    volume = re.sub(r"\s+", "", match.group("volume")).casefold()
    if volume == "ul":
        volume_factor = 0.001
    elif volume == "ml":
        volume_factor = 1.0
    elif volume == "l":
        volume_factor = 1000.0
    else:  # Guarded by _FLOW_TERM_RE.
        raise ValueError(f"Unsupported flow volume unit: {volume!r}")

    time = (match.group("time_direct") or match.group("time_inverse")).casefold()
    if time.startswith("h"):
        time_factor = 1.0 / 60.0
    elif time.startswith("s"):
        time_factor = 60.0
    else:
        time_factor = 1.0
    return volume_factor * time_factor


def _flow_term_to_ml_per_min(match: re.Match[str]) -> float:
    return float(match.group("value")) * _flow_term_factor_to_ml_per_min(match)


def _normalize_flow_rate(cond: Dict[str, Any]) -> None:
    raw = _capture_raw(cond, "flow_rate")
    cond["flow_rate"] = raw
    cond["flow_rate_ml_per_min"] = None
    cond["flow_rate_min_ml_per_min"] = None
    cond["flow_rate_max_ml_per_min"] = None
    cond["flow_rate_uncertainty_ml_per_min"] = None
    cond["flow_rate_standardization_status"] = "unmapped"
    cond.pop("flow_rate_candidates_ml_per_min", None)
    cond.pop("flow_rate_standardization_reason", None)

    if _is_nullish(raw):
        cond["flow_rate"] = None
        return

    text = _normalize_flow_text(str(raw).strip())
    terms = list(_FLOW_TERM_RE.finditer(text))

    # Repeated-unit range: "0.5 mL/min to 1.0 mL/min". It is
    # authoritative only when it accounts for every complete flow term.
    range_pairs = list(zip(terms, terms[1:])) if len(terms) == 2 else []
    for first, second in reversed(range_pairs):
        bridge = text[first.end() : second.start()]
        if re.fullmatch(r"\s*(?:to|[-–—])\s*", bridge, re.IGNORECASE):
            minimum = _flow_term_to_ml_per_min(first)
            maximum = _flow_term_to_ml_per_min(second)
            if (
                not _finite_number(minimum)
                or not _finite_number(maximum)
                or minimum <= 0
                or maximum < minimum
            ):
                cond["flow_rate_standardization_status"] = "invalid"
                return
            cond["flow_rate_min_ml_per_min"] = minimum
            cond["flow_rate_max_ml_per_min"] = maximum
            cond["flow_rate_standardization_status"] = "parsed_range"
            return

    if len(terms) > 1:
        candidates = [_flow_term_to_ml_per_min(term) for term in terms]
        cond["flow_rate_candidates_ml_per_min"] = candidates
        cond["flow_rate_standardization_status"] = "ambiguous"
        cond["flow_rate_standardization_reason"] = (
            "Multiple complete flow rates were reported without recognized range syntax."
        )
        return

    if terms:
        term = terms[-1]
        factor = _flow_term_factor_to_ml_per_min(term)
        prefix = text[: term.start()]

        # Shared-unit uncertainty: "0.5 ± 0.1 mL/min".
        uncertainty = re.search(
            rf"({_FLOW_NUMBER})\s*(?:±|\+/-)\s*$",
            prefix,
            re.IGNORECASE,
        )
        if uncertainty is not None:
            value = float(uncertainty.group(1)) * factor
            delta = abs(float(term.group("value")) * factor)
            if not _finite_number(value) or not _finite_number(delta) or value <= 0:
                cond["flow_rate_standardization_status"] = "invalid"
                return
            cond["flow_rate_ml_per_min"] = value
            cond["flow_rate_uncertainty_ml_per_min"] = delta
            cond["flow_rate_standardization_status"] = "parsed_with_uncertainty"
            return

        # Shared-unit range: "0.5 to 1.0 mL/min".
        shared_range = re.search(
            rf"({_FLOW_NUMBER})\s*(?:to|[-–—])\s*$",
            prefix,
            re.IGNORECASE,
        )
        if shared_range is not None:
            minimum = float(shared_range.group(1)) * factor
            maximum = _flow_term_to_ml_per_min(term)
            if (
                not _finite_number(minimum)
                or not _finite_number(maximum)
                or minimum <= 0
                or maximum < minimum
            ):
                cond["flow_rate_standardization_status"] = "invalid"
                return
            cond["flow_rate_min_ml_per_min"] = minimum
            cond["flow_rate_max_ml_per_min"] = maximum
            cond["flow_rate_standardization_status"] = "parsed_range"
            return

        if re.search(
            rf"{_FLOW_UNSIGNED_NUMBER}\s*/\s*$",
            prefix,
            re.IGNORECASE,
        ):
            cond["flow_rate_standardization_status"] = "ambiguous"
            cond["flow_rate_standardization_reason"] = (
                "A fractional flow-rate value was reported; fraction parsing is not inferred."
            )
            return

        # A leading minus immediately attached to the selected term is a
        # negative value, not punctuation.
        if re.search(r"-\s*$", prefix) and not re.search(
            rf"{_FLOW_UNSIGNED_NUMBER}\s*-\s*$",
            prefix,
            re.IGNORECASE,
        ):
            cond["flow_rate_standardization_status"] = "invalid"
            return

        value = _flow_term_to_ml_per_min(term)
        if not _finite_number(value) or value <= 0:
            cond["flow_rate_standardization_status"] = "invalid"
            return
        cond["flow_rate_ml_per_min"] = value
        cond["flow_rate_standardization_status"] = "parsed"
        return

    # Unitless values are accepted only when the entire field is numeric; the
    # schema names this field as a flow rate, so mL/min is a documented
    # assumption rather than a substring guess.
    unitless_range = re.fullmatch(
        rf"\s*({_FLOW_NUMBER})\s*(?:to|[-–—])\s*({_FLOW_NUMBER})\s*",
        text,
        re.IGNORECASE,
    )
    if unitless_range is not None:
        minimum = float(unitless_range.group(1))
        maximum = float(unitless_range.group(2))
        if minimum <= 0 or maximum < minimum:
            cond["flow_rate_standardization_status"] = "invalid"
            return
        cond["flow_rate_min_ml_per_min"] = minimum
        cond["flow_rate_max_ml_per_min"] = maximum
        cond["flow_rate_standardization_status"] = "assumed_ml_per_min_range"
        return

    unitless = re.fullmatch(rf"\s*({_FLOW_NUMBER})\s*", text)
    if unitless is not None:
        value = float(unitless.group(1))
        if not _finite_number(value) or value <= 0:
            cond["flow_rate_standardization_status"] = "invalid"
            return
        cond["flow_rate_ml_per_min"] = value
        cond["flow_rate_standardization_status"] = "assumed_ml_per_min"
        return

    if re.search(r"(?<![A-Za-z])(?:u\s*l|m\s*l|l)\b", text, re.IGNORECASE):
        cond["flow_rate_standardization_status"] = "incomplete_unit"


_PORE_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
_PORE_UNIT = r"(?:angstroms?|Å|A|nm|[uµμ]m)"
_PORE_VALUE_RE = re.compile(
    rf"(?P<value>{_PORE_NUMBER})\s*(?P<unit>{_PORE_UNIT})?",
    re.IGNORECASE,
)


def _canonical_pore_unit(unit: Optional[str]) -> str:
    normalized = (unit or "A").casefold().replace("µ", "u").replace("μ", "u")
    if normalized == "nm":
        return "nm"
    if normalized == "um":
        return "um"
    return "A"


def _pore_value_to_angstrom(value: float, unit: Optional[str]) -> float:
    normalized = _canonical_pore_unit(unit)
    if normalized == "nm":
        return value * 10.0
    if normalized == "um":
        return value * 10000.0
    return value


def _valid_pore_angstrom(value: float) -> bool:
    # Broad enough for polymeric packings but rejects particle diameters and
    # physically impossible negative/zero values.
    return _finite_number(value) and 1.0 <= value <= 10000.0


def _normalize_pore_size(cond: Dict[str, Any]) -> None:
    raw = _capture_raw(cond, "pore_size")
    cond["pore_size"] = raw
    cond["pore_size_angstrom"] = None
    cond["pore_size_min_angstrom"] = None
    cond["pore_size_max_angstrom"] = None
    cond["pore_size_uncertainty_angstrom"] = None
    cond["pore_size_standardization_status"] = "unmapped"

    if _is_nullish(raw):
        cond["pore_size"] = None
        return

    text = unicodedata.normalize("NFKC", str(raw)).strip()
    text = text.replace("Ã…", "Å").replace("Âµ", "µ")
    text = re.sub(r"(\d)\s*-\s*(Å|A|nm|[uµμ]m)\b", r"\1 \2", text)

    if re.search(r"\d\s*m\b", text, re.IGNORECASE) and not re.search(
        r"\d\s*(?:n|u|µ|μ)m\b", text, re.IGNORECASE
    ):
        cond["pore_size_standardization_status"] = "unmapped"
        return

    pore_context = r"(?:nominal\s+)?pore(?:s|\s*size)?"
    uncertainty_match = re.fullmatch(
        rf"\s*(?:{pore_context}\s*[:=]?\s*)?"
        rf"({_PORE_NUMBER})\s*({_PORE_UNIT})?\s*(?:±|\+/-)\s*"
        rf"(\d+(?:\.\d+)?)\s*({_PORE_UNIT})?"
        rf"(?:\s*{pore_context})?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if uncertainty_match is not None:
        base_unit = uncertainty_match.group(2) or uncertainty_match.group(4)
        uncertainty_unit = uncertainty_match.group(4) or uncertainty_match.group(2)
        value = _pore_value_to_angstrom(
            float(uncertainty_match.group(1)), base_unit
        )
        uncertainty = _pore_value_to_angstrom(
            float(uncertainty_match.group(3)), uncertainty_unit
        )
        if _valid_pore_angstrom(value) and _finite_number(uncertainty) and uncertainty >= 0:
            cond["pore_size_angstrom"] = value
            cond["pore_size_uncertainty_angstrom"] = uncertainty
            cond["pore_size_standardization_status"] = "parsed_with_uncertainty"
        else:
            cond["pore_size_standardization_status"] = "invalid"
        return

    # Handle labeled and unlabeled ranges before single labeled values so an
    # endpoint is never silently discarded.
    range_match = re.fullmatch(
        rf"\s*(?:{pore_context}\s*[:=]?\s*)?"
        rf"({_PORE_NUMBER})\s*({_PORE_UNIT})?\s*(?:to|[-–—−])\s*"
        rf"({_PORE_NUMBER})\s*({_PORE_UNIT})?"
        rf"(?:\s*{pore_context})?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if range_match is not None:
        unit_one = range_match.group(2) or range_match.group(4)
        unit_two = range_match.group(4) or range_match.group(2)
        minimum = _pore_value_to_angstrom(float(range_match.group(1)), unit_one)
        maximum = _pore_value_to_angstrom(float(range_match.group(3)), unit_two)
        if (
            _valid_pore_angstrom(minimum)
            and _valid_pore_angstrom(maximum)
            and minimum <= maximum
        ):
            cond["pore_size_min_angstrom"] = minimum
            cond["pore_size_max_angstrom"] = maximum
            cond["pore_size_standardization_status"] = "parsed_range"
        else:
            cond["pore_size_standardization_status"] = "invalid"
        return

    # Unsupported uncertainty syntax must not fall through and be treated as
    # two independent pore sizes.
    if re.search(r"(?:±|\+/-)", text):
        cond["pore_size_standardization_status"] = "ambiguous"
        return

    # If particle and pore values are both reported, use only values explicitly
    # labeled as pores.
    pore_labeled = re.findall(
        rf"({_PORE_NUMBER})\s*({_PORE_UNIT})\s*(?:pore|pores|pore\s*size)",
        text,
        flags=re.IGNORECASE,
    )
    pore_labeled.extend(
        re.findall(
            rf"\bpore(?:s|\s*size)?\s*[:=]?\s*({_PORE_NUMBER})\s*({_PORE_UNIT})",
            text,
            flags=re.IGNORECASE,
        )
    )
    if pore_labeled:
        converted = [
            _pore_value_to_angstrom(float(value), unit)
            for value, unit in pore_labeled
        ]
        if all(_valid_pore_angstrom(value) for value in converted):
            cond["pore_size_angstrom"] = converted[0] if len(converted) == 1 else converted
            cond["pore_size_standardization_status"] = "parsed"
        else:
            cond["pore_size_standardization_status"] = "invalid"
        return

    if re.search(r"\bparticle", text, re.IGNORECASE):
        cond["pore_size_standardization_status"] = "ambiguous"
        return

    matches = list(_PORE_VALUE_RE.finditer(text))
    if not matches:
        return

    # Reject mixed explicit units outside a real range; this is commonly a
    # particle-size/pore-size mixture such as "5 µm, 300 Å".
    explicit_units = {
        _canonical_pore_unit(match.group("unit"))
        for match in matches
        if match.group("unit")
    }
    if len(explicit_units) > 1:
        cond["pore_size_standardization_status"] = "ambiguous"
        return

    inherited_unit = next(iter(explicit_units), None)
    values = [
        _pore_value_to_angstrom(
            float(match.group("value")),
            match.group("unit") or inherited_unit,
        )
        for match in matches
    ]
    if not all(_valid_pore_angstrom(value) for value in values):
        cond["pore_size_standardization_status"] = "invalid"
        return
    cond["pore_size_angstrom"] = values[0] if len(values) == 1 else values
    cond["pore_size_standardization_status"] = "parsed"


def _temperature_to_celsius(value: float, unit: Optional[str]) -> Optional[float]:
    normalized = (unit or "C").strip().casefold().replace("°", "")
    if normalized in {"c", "celsius"}:
        result = value
    elif normalized in {"k", "kelvin"}:
        if value < 0:
            return None
        result = value - 273.15
    elif normalized in {"f", "fahrenheit"}:
        result = (value - 32.0) * 5.0 / 9.0
    else:
        return None
    return result if -150.0 <= result <= 400.0 else None


def _temperature_delta_to_celsius(value: float, unit: Optional[str]) -> Optional[float]:
    normalized = (unit or "C").strip().casefold().replace("°", "")
    if normalized in {"c", "celsius", "k", "kelvin"}:
        return value
    if normalized in {"f", "fahrenheit"}:
        return value * 5.0 / 9.0
    return None


def _normalize_temperature(cond: Dict[str, Any]) -> None:
    raw = _capture_raw(cond, "temperature_celsius")
    cond["temperature_celsius"] = None
    cond["temperature_min_celsius"] = None
    cond["temperature_max_celsius"] = None
    cond["temperature_uncertainty_celsius"] = None
    cond["temperature_standardization_status"] = "unmapped"

    if _is_nullish(raw):
        return

    text = unicodedata.normalize("NFKC", str(raw)).strip()
    number = r"[+-]?\d+(?:\.\d+)?"
    unit = r"(?:°?\s*(?:C|K|F)|celsius|kelvin|fahrenheit)"

    uncertainty_match = re.fullmatch(
        rf"\s*({number})\s*({unit})?\s*(?:±|\+/-)\s*"
        rf"(\d+(?:\.\d+)?)\s*({unit})?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if uncertainty_match is not None:
        base_unit = uncertainty_match.group(2) or uncertainty_match.group(4)
        uncertainty_unit = uncertainty_match.group(4) or uncertainty_match.group(2)
        value = _temperature_to_celsius(
            float(uncertainty_match.group(1)), base_unit
        )
        uncertainty = _temperature_delta_to_celsius(
            float(uncertainty_match.group(3)), uncertainty_unit
        )
        if value is None or uncertainty is None or uncertainty < 0:
            cond["temperature_standardization_status"] = "invalid"
            return
        cond["temperature_celsius"] = round(value, 6)
        cond["temperature_uncertainty_celsius"] = round(uncertainty, 6)
        cond["temperature_standardization_status"] = "parsed_with_uncertainty"
        return

    range_match = re.fullmatch(
        rf"\s*({number})\s*({unit})?\s*(?:to|[-–—−])\s*"
        rf"({number})\s*({unit})?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if range_match is not None:
        unit_one = range_match.group(2) or range_match.group(4)
        unit_two = range_match.group(4) or range_match.group(2)
        minimum = _temperature_to_celsius(float(range_match.group(1)), unit_one)
        maximum = _temperature_to_celsius(float(range_match.group(3)), unit_two)
        if minimum is None or maximum is None or minimum > maximum:
            cond["temperature_standardization_status"] = "invalid"
            return
        cond["temperature_min_celsius"] = round(minimum, 6)
        cond["temperature_max_celsius"] = round(maximum, 6)
        cond["temperature_standardization_status"] = (
            "parsed" if unit_one or unit_two else "assumed_celsius"
        )
        return

    single_match = re.fullmatch(
        rf"\s*({number})\s*({unit})?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if single_match is None:
        # Qualitative values such as "room temperature" remain auditable in
        # temperature_celsius_raw but cannot enter numeric comparisons.
        return

    unit_text = single_match.group(2)
    value = _temperature_to_celsius(float(single_match.group(1)), unit_text)
    if value is None:
        cond["temperature_standardization_status"] = "invalid"
        return
    cond["temperature_celsius"] = round(value, 6)
    cond["temperature_standardization_status"] = (
        "parsed" if unit_text else "assumed_celsius"
    )


# =============================================================================
# COLUMN MODE, ARCHITECTURE, AND PUBLICATION YEAR
# =============================================================================


def _normalize_column_mode(cond: Dict[str, Any]) -> None:
    raw = _capture_raw(cond, "column_mode")
    cond["column_mode"] = None
    cond["column_mode_standardization_status"] = "unmapped"
    if _is_nullish(raw):
        return

    text = re.sub(r"[-_]+", " ", str(raw).strip().casefold())
    text = re.sub(r"\s+", " ", text).strip()
    mapping = {
        "reverse": "Reverse",
        "reverse phase": "Reverse",
        "reversed": "Reverse",
        "reversed phase": "Reverse",
        "rp": "Reverse",
        "normal": "Normal",
        "normal phase": "Normal",
        "np": "Normal",
        "hilic": "HILIC",
        "hydrophilic interaction": "HILIC",
        "hydrophilic interaction chromatography": "HILIC",
        "sec": "SEC",
        "size exclusion": "SEC",
        "size exclusion chromatography": "SEC",
        "ion exchange": "Ion Exchange",
        "ion exchange chromatography": "Ion Exchange",
        "iec": "Ion Exchange",
    }
    canonical = mapping.get(text)
    if canonical is None:
        return
    cond["column_mode"] = canonical
    cond["column_mode_standardization_status"] = (
        "exact" if _display_key(raw) == _display_key(canonical) else "alias"
    )


ARCHITECTURE_MAP = {
    "linear": "linear",
    "linear homopolymer": "linear",
    "linear homopolymer with various end groups": "linear",
    "ring": "cyclic",
    "cyclic": "cyclic",
    "star": "star",
    "graft": "graft",
    "diblock": "diblock",
    "ab block copolymer": "diblock",
    "diblock (ab)": "diblock",
    "eo-po diblock": "diblock",
    "triblock": "triblock",
    "triblock copolymer": "triblock",
    "triblock (eo-po-eo)": "triblock",
    "eo-po-eo triblock": "triblock",
    "peo-ppo-peo triblock": "triblock",
    "aba triblock": "triblock",
    "eo-po-eo": "triblock",
    "peo-ppo-peo": "triblock",
    "bab block copolymer": "triblock",
    "triblock (bab)": "triblock",
    "block copolymer": "block copolymer",
    "block": "block copolymer",
    "ab block": "block copolymer",
    "tetrablock copolymer": "tetrablock",
    "tetrablock": "tetrablock",
    "random copolymer": "random copolymer",
    "null": None,
    "none": None,
    "difunktionell": None,
    "monomethyl ether": None,
}


def _normalize_architecture(cond: Dict[str, Any]) -> None:
    raw = _capture_raw(cond, "architecture")
    cond["architecture"] = None
    cond["architecture_standardization_status"] = "unmapped"
    cond.pop("architecture_standardization_candidates", None)
    if _is_nullish(raw):
        return

    text = unicodedata.normalize("NFKC", str(raw)).strip().casefold()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)

    detected: List[str] = []
    detection_rules = (
        ("tetrablock", r"\btetrablock\b"),
        ("triblock", r"\btriblock\b|\bbab\s+block\b|\b(?:p?eo)-p?po-(?:p?eo)\b"),
        ("diblock", r"\bdiblock\b|\bab\s+block\b"),
        ("random copolymer", r"\brandom\s+copolymer\b"),
        ("cyclic", r"\bcyclic\b|\bring\b"),
        ("star", r"\bstar\b"),
        ("graft", r"\bgraft\b"),
        ("linear", r"\blinear\b"),
    )
    for canonical, pattern in detection_rules:
        if re.search(pattern, text, re.IGNORECASE):
            detected.append(canonical)
    if len(detected) > 1:
        cond["architecture_standardization_status"] = "ambiguous"
        cond["architecture_standardization_candidates"] = detected
        return
    if text in ARCHITECTURE_MAP:
        cond["architecture"] = ARCHITECTURE_MAP[text]
        cond["architecture_standardization_status"] = "exact"
        return

    for key, canonical in ARCHITECTURE_MAP.items():
        if canonical is not None and text.startswith(f"{key} "):
            cond["architecture"] = canonical
            cond["architecture_standardization_status"] = "alias"
            return

    if "block" in text:
        cond["architecture"] = "block copolymer"
        cond["architecture_standardization_status"] = "ambiguous"


def _normalize_year(cond: Dict[str, Any]) -> None:
    raw = _capture_raw(cond, "publication_year")
    cond["publication_year"] = None
    cond["publication_year_standardization_status"] = "unmapped"
    if _is_nullish(raw):
        return

    match = re.search(r"(?<!\d)(\d{4})(?!\d)", str(raw))
    if match is None:
        return
    year = int(match.group(1))
    current_year = datetime.now(timezone.utc).year
    if not 1800 <= year <= current_year + 1:
        cond["publication_year_standardization_status"] = "invalid"
        return
    cond["publication_year"] = year
    cond["publication_year_standardization_status"] = "parsed"


# =============================================================================
# HPLC COLUMN / STATIONARY-PHASE PRODUCT NAME STANDARDIZATION
# =============================================================================


_COLUMN_CANONICAL_TO_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Bridged Ethane Hybrid Particle C18": (
        "BEH C18",
        "UPLC BEH C18",
        "Waters BEH C18",
        "ACQUITY UPLC BEH C18",
    ),
    "Chrompak RP-18": ("Chrompak RP18",),
    "Discovery HS-PEG": (),
    "Equisil C18": ("octadecyl-silica Equisil", "octadecylsilica Equisil"),
    "Jordi Gel DVB 500 RP": ("Jordi Gel DVB 500A RP",),
    "Jupiter C18": ("Juptier C18",),
    "LiChrosorb Diol": (),
    "LiChrosorb Si": (),
    "LiChrospher Si": (),
    "Luna C18": ("Phenomenex, Luna C18",),
    "Luna HILIC": (),
    "Novapak C18": ("Nova-Pak C18",),
    "Nucleosil C18": (
        "Knauer Nucleosil RP18-100",
        "Macherey Nagel Nucleosil 5C18",
        "Macherey-Nagel Nucleosil 5C18",
        "Nucleosil 5C18",
        "Nucleosil RP18",
        "Nucleosil RP-18",
        "Nucleosil 100 RP18",
        "Nucleosil-100 RP-18",
    ),
    "Nucleosil C8": ("Nucleosil RP8", "Nucleosil RP-8"),
    "Nucleosil C8 HD": (
        "Macherey-Nagel 250 mm x 4.6 mm Nucleosil C8 HD",
        "Macherey-Nagel 250 mm × 4.6 mm Nucleosil C8 HD",
    ),
    "Nucleosil CN": (),
    "Nucleosil Diol": (
        "Nucleosil 100-5 OH",
        "Nucleosil 100-5 OH 5 mm",
        "Nucleosil 100-5 OH 5 um",
        "Nucleosil 100-5 OH 5 µm",
        "Nucleosil 100-5 OH 5 μm",
    ),
    "Nucleosil NH2": ("amino-silica NUCLEOSIL", "Nucleosil amino"),
    "Nucleosil Si": (
        "300 A Nucleosil Si",
        "300 Å Nucleosil Si",
        "Nucleosil 300 A",
        "Nucleosil 300 Å",
        "Nucleosil 300A",
        "Nucleosil bare silica",
        "Nucleosil Si 300",
        "Nucleosil Si 300A",
    ),
    "Onyx Monolithic C18": ("Onyx C18",),
    "PhenoSphere-NEXT Si": (
        "PhenoSphere-Next 5u Sil",
        "PhenoSphere-Next 5 um Sil",
        "PhenoSphere-Next 5 µm Sil",
        "PhenoSphere-Next 5 μm Sil",
    ),
    "PLRP-S": (),
    "Prodigy ODS-3": ("Prodigy ODS3", "Prodigy ODS(3)"),
    "PS/DVB gel": ("Phenogel", "Phenogel (linear)"),
    "Soloza K-0": (),
    "Soloza K-33": (),
    "Soloza KG-8": (),
    "Sphereclone C6": (),
    "Spherisorb ODS2": ("SpHEROSIL ODS2",),
    "Spherisorb S5P": (),
    "Spherisorb S5X C6": (),
    "Synergi Fusion RP": (),
    "Toyopearl Butyl-650M": (),
    "Ultisil Diol": (),
    "Ultisil XB-Phenyl": ("XB-Phenyl", "XB phenyl"),
    "XB-C18": (),
    "YMC C18": ("YMC RP 18", "YMC RP18", "YMC RP 2-3 18"),
    "Zorbax 300 C18": (),
}


_COLUMN_AMBIGUOUS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "C18": (
        (),
        "C18 identifies bonded-phase chemistry but not a unique column product.",
    ),
    "RP-18": (
        (),
        "RP-18 is used by multiple manufacturers and column products.",
    ),
    "ODS2": (
        (),
        "ODS2 is a phase designation; the manufacturer/product is missing.",
    ),
    "Si": (
        (),
        "Si identifies bare-silica chemistry but not a unique column product.",
    ),
    "Nucleosil": (
        (
            "Nucleosil C18",
            "Nucleosil C8",
            "Nucleosil C8 HD",
            "Nucleosil CN",
            "Nucleosil Diol",
            "Nucleosil NH2",
            "Nucleosil Si",
        ),
        "NUCLEOSIL is a product family; the bonded phase is missing.",
    ),
    "Nucleosil C8 HD, Nucleosil C8": (
        ("Nucleosil C8 HD", "Nucleosil C8"),
        "The value contains two distinct column products and must be compared as a set.",
    ),
    "octadecyl silica column": (
        (),
        "Only C18 chemistry is reported; no commercial product is identified.",
    ),
    "column 5": ((), "Paper-local column number; source context is required."),
    "column 8": ((), "Paper-local column number; source context is required."),
    "column 9": ((), "Paper-local column number; source context is required."),
    "Macherey-Nagel 5C;": ((), "Truncated or OCR-corrupted column name."),
    "GROM ODS 120 and Zorbax SBC 18": (
        ("GROM ODS 120", "Zorbax SB-C18"),
        "Two distinct physical columns were combined into one field.",
    ),
}


_COLUMN_UNMAPPED: Dict[str, str] = {
    "Chromolith Si": "Specific product is absent from the Pharma workbook vocabulary.",
    "FAD column": "The abbreviation FAD is not defined well enough to identify a product.",
    "Lichrosorb-10 RP-18": "Specific product is absent from the Pharma workbook vocabulary.",
    "PerfectSil": "The product family is incomplete and absent from the Pharma workbook vocabulary.",
    "Shodex-C18": "Specific product is absent from the Pharma workbook vocabulary.",
    "Si-120 silica column": "No manufacturer or unique product identity can be established.",
    "Spherisorb Si": "Specific product is absent from the Pharma workbook vocabulary.",
    "Symmetry 300": "Specific product is absent from the Pharma workbook vocabulary.",
    "Symmetry 300 A, 5 um, 4.6 x 250 mm, Waters, Milford, MA, USA": (
        "Specific product is absent from the Pharma workbook vocabulary."
    ),
    "Symmetry 300 Å, 5 µm, 4.6 x 250 mm, Waters, Milford, MA, USA": (
        "Specific product is absent from the Pharma workbook vocabulary."
    ),
    "Symmetry C18": "Specific product is absent from the Pharma workbook vocabulary.",
    "Synergi MAX RP": "Distinct product; do not collapse it to Synergi Fusion RP.",
    "ZORBAX ODS": "Distinct 70-angstrom ODS product; do not map it to Zorbax 300 C18.",
}


def _build_column_lookups() -> Tuple[
    Dict[str, str],
    Dict[str, str],
    Dict[str, Tuple[Tuple[str, ...], str]],
    Dict[str, str],
]:
    exact: Dict[str, str] = {}
    targets: Dict[str, set[str]] = defaultdict(set)
    for canonical, aliases in _COLUMN_CANONICAL_TO_ALIASES.items():
        exact_key = _display_key(canonical)
        if not exact_key:
            raise ValueError(f"Empty canonical column name: {canonical!r}")
        exact[exact_key] = canonical
        for value in (canonical, *aliases):
            key = _alphanumeric_key(value)
            if not key:
                raise ValueError(f"Empty column alias key: {value!r}")
            targets[key].add(canonical)

    collisions = {
        key: tuple(sorted(values, key=str.casefold))
        for key, values in targets.items()
        if len(values) > 1
    }
    if collisions:
        raise ValueError(f"Column alias collisions: {collisions}")

    aliases = {key: next(iter(values)) for key, values in targets.items()}
    ambiguous: Dict[str, Tuple[Tuple[str, ...], str]] = {}
    for raw, specification in _COLUMN_AMBIGUOUS.items():
        key = _alphanumeric_key(raw)
        previous = ambiguous.get(key)
        if previous is not None and previous != specification:
            raise ValueError(f"Conflicting ambiguous column rules for {raw!r}")
        ambiguous[key] = specification
    unmapped = {_alphanumeric_key(raw): reason for raw, reason in _COLUMN_UNMAPPED.items()}

    overlap = set(aliases) & set(ambiguous)
    overlap |= set(aliases) & set(unmapped)
    overlap |= set(ambiguous) & set(unmapped)
    if overlap:
        raise ValueError(f"Column standardization rule overlap: {sorted(overlap)}")
    return exact, aliases, ambiguous, unmapped


(
    _COLUMN_EXACT_LOOKUP,
    _COLUMN_ALIAS_LOOKUP,
    _COLUMN_AMBIGUOUS_LOOKUP,
    _COLUMN_UNMAPPED_LOOKUP,
) = _build_column_lookups()


def _normalize_column_name(cond: Dict[str, Any]) -> None:
    raw = _capture_raw(cond, "column_name")
    cond.pop("column_name_standardization_candidates", None)
    cond.pop("column_name_standardization_reason", None)

    if _is_nullish(raw):
        cond["column_name"] = None
        cond["column_name_standardization_status"] = "unmapped"
        cond["column_name_standardization_reason"] = "Column name was not reported."
        return
    if not isinstance(raw, str):
        cond["column_name"] = None
        cond["column_name_standardization_status"] = "unmapped"
        cond["column_name_standardization_reason"] = (
            "Expected column_name to be a string or null; "
            f"received {type(raw).__name__}."
        )
        return

    raw_text = re.sub(r"\s+", " ", raw.strip())
    canonical = _COLUMN_EXACT_LOOKUP.get(_display_key(raw_text))
    if canonical is not None:
        cond["column_name"] = canonical
        cond["column_name_standardization_status"] = "exact"
        return

    key = _alphanumeric_key(raw_text)
    canonical = _COLUMN_ALIAS_LOOKUP.get(key)
    if canonical is not None:
        cond["column_name"] = canonical
        cond["column_name_standardization_status"] = "alias"
        return

    ambiguous = _COLUMN_AMBIGUOUS_LOOKUP.get(key)
    if ambiguous is not None:
        candidates, reason = ambiguous
        cond["column_name"] = None
        cond["column_name_standardization_status"] = "ambiguous"
        cond["column_name_standardization_candidates"] = list(candidates)
        cond["column_name_standardization_reason"] = reason
        return

    reason = _COLUMN_UNMAPPED_LOOKUP.get(key)
    cond["column_name"] = None
    cond["column_name_standardization_status"] = "unmapped"
    cond["column_name_standardization_reason"] = (
        reason or "No hard-coded column-name rule exists for this value."
    )


# =============================================================================
# PUBLIC API AND FILE PROCESSING
# =============================================================================


def standardize_condition(cond: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cond, dict):
        raise TypeError("A condition must be a dictionary.")
    out = copy.deepcopy(cond)

    # Solvents run first because they can recover an embedded ratio and its
    # scope from strings such as "THF/DMF 96:4 (v/v)".
    _normalize_solvents(out)
    _normalize_ratio_units(out)
    _normalize_ratio(out)
    _normalize_flow_rate(out)
    _normalize_pore_size(out)
    _normalize_temperature(out)
    _normalize_column_name(out)
    _normalize_column_mode(out)
    _normalize_polymer_fields(out)
    _normalize_architecture(out)
    _normalize_year(out)
    return out


def _validate_input_document(data: Any, input_path: Path) -> List[Dict[str, Any]]:
    if not isinstance(data, dict):
        raise TypeError(f"Top-level JSON value must be an object: {input_path}")

    extracted_data = data.get("extracted_data")
    if not isinstance(extracted_data, dict):
        raise ValueError(f"Missing or invalid extracted_data object: {input_path}")
    if "conditions" not in extracted_data:
        raise ValueError(f"Missing extracted_data.conditions: {input_path}")

    conditions = extracted_data["conditions"]
    if not isinstance(conditions, list):
        raise TypeError(f"extracted_data.conditions must be a list: {input_path}")

    invalid_indices = [
        index for index, condition in enumerate(conditions) if not isinstance(condition, dict)
    ]
    if invalid_indices:
        raise TypeError(
            "Every extracted_data.conditions item must be an object; "
            f"invalid indices in {input_path}: {invalid_indices}"
        )
    return conditions


def _atomic_json_dump(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def standardize_file(input_path: Path, output_path: Path) -> int:
    """Standardize one consensus file and return its condition count."""
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        conditions = _validate_input_document(data, input_path)
        standardized_conditions = [
            standardize_condition(condition) for condition in conditions
        ]

        output = copy.deepcopy(data)
        metadata = output.get("metadata")
        if metadata is None:
            metadata = {}
            output["metadata"] = metadata
        elif not isinstance(metadata, dict):
            raise TypeError(f"metadata must be an object: {input_path}")
        metadata.setdefault("source_pdf", input_path.stem.removesuffix("_consensus"))
        metadata["standardized_by"] = "pipeline/standardizer.py"
        metadata["standardization_date"] = datetime.now(timezone.utc).isoformat()

        summary = output.get("summary")
        if summary is None:
            summary = {}
            output["summary"] = summary
        elif not isinstance(summary, dict):
            raise TypeError(f"summary must be an object: {input_path}")
        summary["total_conditions"] = len(standardized_conditions)
        output["extracted_data"]["conditions"] = standardized_conditions

        _atomic_json_dump(output, output_path)
        return len(standardized_conditions)
    except Exception:
        logger.exception("Failed to process %s", input_path)
        raise


def standardize_all(consensus_dir: Path, output_dir: Path) -> None:
    if not consensus_dir.is_dir():
        raise FileNotFoundError(f"Consensus directory does not exist: {consensus_dir}")

    input_paths = sorted(consensus_dir.rglob("*_consensus.json"))
    if not input_paths:
        raise FileNotFoundError(
            f"No *_consensus.json files found in: {consensus_dir}"
        )

    total_conditions = 0
    for input_path in input_paths:
        relative_path = input_path.relative_to(consensus_dir)
        base_name = input_path.stem.removesuffix("_consensus")
        output_path = output_dir / relative_path.parent / f"{base_name}_standardized.json"
        count = standardize_file(input_path, output_path)
        print(f"Processing {relative_path} ... {count} conditions")
        total_conditions += count

    print("=" * 54)
    print(
        f"Standardized {total_conditions} conditions across "
        f"{len(input_paths)} files."
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardize consensus JSON files with conservative PolyCrit rules."
    )
    parser.add_argument(
        "consensus_dir",
        nargs="?",
        default="results/consensus",
        help="Directory containing *_consensus.json files",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="results/standardized",
        help="Directory for *_standardized.json files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = _build_argument_parser().parse_args(argv)
    standardize_all(Path(arguments.consensus_dir), Path(arguments.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
