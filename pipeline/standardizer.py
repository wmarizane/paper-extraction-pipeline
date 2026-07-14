"""
Post-processing standardizer for consensus JSON output.

Reads:   results/consensus/**/*_consensus.json
Writes:  results/standardized/**/*_standardized.json
         (same subdirectory tree as consensus/)

Source JSONs are never modified.
Each standardized condition preserves the original raw value alongside
the cleaned value (with a _raw suffix) for full auditability.
"""

import argparse
import json
import logging
import copy
import re
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

ARCHITECTURE_MAP = {
    "linear": "linear",
    "linear homopolymer": "linear",
    "homopolymer": "linear",
    "linear homopolymer with various end groups": "linear",
    "ring": "cyclic",
    "cyclic": "cyclic",
    "star": "star",
    "2-arm": "star",
    "graft": "graft",
    "diblock": "diblock",
    "ab block copolymer": "diblock",
    "diblock (ab)": "diblock",
    "eo-po diblock": "diblock",
    "triblock": "triblock",
    "triblock copolymer": "triblock",
    "triblock (eo-po-eo)": "triblock",
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

_POLYMER_CANONICAL_TO_ALIASES = {

    "Poly(propylene glycol)": ["PPG"],
    
    "Poly(ethylene glycol)": ["PEG", "mPEG", "PEG-MME", "PEG-DME", "MeO-PEG-DME", "MeO-PEG"],


    "Poly(L-lactide)": ["PLLA", "PLA"],

    "Poly(ethylene oxide)": ["PEO"],
    "Poly(propylene oxide)": ["PPO"],

}


def _normalize_polymer_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


_POLYMER_COMPATIBILITY_ALIASES = {
    # PolyCrit rows use PEG and PPG, while papers often use oxide terminology.
    "Poly(ethylene glycol)": [
        "PEO",
        "Poly(ethylene oxide)",
        "Polyethylene oxide",
        "Polyoxyethylene",
    ],
    
    "Poly(propylene glycol)": [
        "PPO",
        "Poly(propylene oxide)",
        "Polypropylene oxide",
        "Polyoxypropylene",
    ],
    
    "Poly(L-lactide)": [
        "PLLA"
        "Poly(L-lactic acid)",
        "Poly-L-lactide",
        ],

    "Poly(lactide)": [
        "PLA"
        "Poly(lactic acid)",
        "Polylactide",
        ],

    "Poly(Ethylene-co-Propylene)": [
        "EP Copolymer",
        ],
    
    "Poloxamer": [
        "Pluronic",
        "Kolliphor",
        "Synperonic"
        "PEO-PPO-PEO"
        "ABA triblock"
        ],
}


_POLYMER_AMBIGUOUS_ALIAS_POLICIES: Dict[str, Optional[str]] = {}


def _build_polymer_lookup() -> Tuple[
    Dict[str, str],
    Dict[str, str],
    Dict[str, Tuple[str, ...]],
]:
    canonical_display: Dict[str, str] = {}
    alias_targets: Dict[str, set[str]] = defaultdict(set)

    for canonical, aliases in _POLYMER_CANONICAL_TO_ALIASES.items():
        canonical_key = _normalize_polymer_key(canonical)
        if not canonical_key:
            raise ValueError(f"Empty canonical polymer key: {canonical!r}")
        existing = canonical_display.get(canonical_key)
        if existing and existing != canonical:
            raise ValueError(
                f"Canonical polymer key collision: {existing!r} and {canonical!r}"
            )
        canonical_display[canonical_key] = canonical
        for value in (canonical, *aliases):
            alias_key = _normalize_polymer_key(value)
            if alias_key:
                alias_targets[alias_key].add(canonical_key)

    for canonical, aliases in _POLYMER_COMPATIBILITY_ALIASES.items():
        canonical_key = _normalize_polymer_key(canonical)
        if canonical_key not in canonical_display:
            raise ValueError(f"Unknown compatibility canonical: {canonical!r}")
        for alias in aliases:
            alias_key = _normalize_polymer_key(alias)
            if alias_key:
                alias_targets[alias_key].add(canonical_key)

    conflicts = {
        alias_key: tuple(
            sorted(
                (canonical_display[key] for key in target_keys),
                key=str.casefold,
            )
        )
        for alias_key, target_keys in alias_targets.items()
        if len(target_keys) > 1
    }
    normalized_policies = {
        _normalize_polymer_key(alias): canonical
        for alias, canonical in _POLYMER_AMBIGUOUS_ALIAS_POLICIES.items()
    }


    alias_to_canonical: Dict[str, str] = {}
    for alias_key, target_keys in alias_targets.items():
        if len(target_keys) == 1:
            alias_to_canonical[alias_key] = next(iter(target_keys))
            continue

        selected = normalized_policies.get[alias_key]
        if selected is None:
            continue
        selected_key = _normalize_polymer_key(selected)
        if selected_key not in target_keys:
            raise ValueError(
                f"Invalid polymer alias policy for {alias_key!r}: {selected!r}"
            )
        alias_to_canonical[alias_key] = selected_key

    return alias_to_canonical, canonical_display, conflicts


(
    _POLYMER_ALIAS_TO_CANONICAL,
    _POLYMER_CANONICAL_DISPLAY,
    _POLYMER_ALIAS_CONFLICTS,
) = _build_polymer_lookup()


_POLYMER_DESCRIPTOR_RE = re.compile(
    r"^(?:linear|cyclic|ring|star|graft|branched|isotactic|syndiotactic|atactic)\s+",
    re.IGNORECASE,
)


def _extract_polymer_candidates(raw: str) -> list[str]:
    text = str(raw).strip()
    if not text:
        return []

    candidates = []
    seen = set()

    def _add(candidate: str) -> None:
        item = candidate.strip()
        if not item:
            return
        key = item.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(item)

        stripped = _POLYMER_DESCRIPTOR_RE.sub("", item).strip()
        if stripped and stripped.lower() not in seen:
            seen.add(stripped.lower())
            candidates.append(stripped)

    _add(text)

    no_parenthetical = re.sub(r"\([^()]*\)", "", text).strip()
    if no_parenthetical and no_parenthetical != text:
        _add(no_parenthetical)

    for match in re.finditer(r"\(([^()]+)\)", text):
        _add(match.group(1))

    for token in re.split(r"\s+and\s+|[,;]", no_parenthetical):
        _add(token)

    return candidates


def _normalize_polymer_name(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text

    for candidate in _extract_polymer_candidates(text):
        canonical_key = _normalize_polymer_key(candidate)
        if not canonical_key:
            continue
        mapped_key = _POLYMER_ALIAS_TO_CANONICAL.get(canonical_key)
        if mapped_key:
            canonical = _POLYMER_CANONICAL_DISPLAY.get(mapped_key)
            if canonical:
                return canonical
    return text


def _normalize_polymer_fields(cond: Dict[str, Any]) -> None:
    for field in ("analyte_polymer", "critical_component"):
        value = cond.get(field)
        raw = value.strip() if isinstance(value, str) else value
        if raw is None:
            continue

        # Keep the original extracted polymer text for downstream human review.
        raw_field = f"{field}_raw"
        if raw_field not in cond:
            cond[raw_field] = value

        source_text = str(cond.get(raw_field, "")).strip()
        source_keys = [_normalize_polymer_key(source_text)]
        stripped_source = _POLYMER_DESCRIPTOR_RE.sub("", source_text).strip()
        if stripped_source != source_text:
            source_keys.append(_normalize_polymer_key(stripped_source))
        conflict_key = next(
            (key for key in source_keys if key in _POLYMER_ALIAS_CONFLICTS),
            None,
        )
        if conflict_key:
            cond[f"{field}_canonical_candidates"] = list(
                _POLYMER_ALIAS_CONFLICTS[conflict_key]
            )
            status = (
                "resolved_ambiguous_alias"
                if conflict_key in _POLYMER_ALIAS_TO_CANONICAL
                else "ambiguous_alias"
            )
            cond[f"{field}_standardization_status"] = status

        if isinstance(raw, str):
            normalized = _normalize_polymer_name(raw)
            cond[field] = normalized


_SOLVENT_CANONICAL_TO_ALIASES = {



    
    "1,4-Dioxane": ["1,4-Dioxan"],
    
    "1-Propanol": ["n-Propanol", "Propan-1-ol", "n-Propyl alcohol"],

    
    "2-Propanol": ["IPA", "Isopropanol", "Isopropyl alcohol", "i-PrOH"],
    
    "Acetic Acid": ["Ethanoic acid", "AcOH", "HOAc"],
    
    "Acetone": ["Propanone", "2-Propanone"],
    
    "Acetonitrile": ["ACN", "CH3CN", "MeCN", "Methyl cyanide"],

    
    
    "Chloroform": ["CHCl3", "Trichloromethane"],




    "Deuterated Acetone": ["Acetone-d6", "d6-Acetone"],


    
    "Dimethylformamide": ["DMF", "N,N-Dimethylformamide", "Dimethyl Formamide"],
    
    "Dioxane": ["1,4-Dioxane"],

    "Ethyl Acetate": ["EtOAc", "Ethyl ethanoate", "Ethylacetat"],
    
    "Heptane": ["n-Heptane"],
    "Hexane": ["n-Hexane"],
    "Methanol": ["MeOH", "CH3OH", "Methyl alcohol"],
    
    "Methyl Ethyl Ketone": [
        "MEK",
        "Butanone",
        "2-Butanone",
        "Methylethylketon",
        "Methyl ethyl ketone (MEK)",
    ],

    "Tetrachloromethane": ["CCl4", "Carbon tetrachloride"],
    
    "Tetrahydrofuran": ["THF", "Tetrahydrofuran (THF)", "Tetrahydrofurane"],

    "Triethylamine": ["TEA", "Et3N"],
    "Trimethylamine": ["TMA", "Me3N"],
    
    "Dimethoxyethane": [
        "DME",
        "Glyme",
        "Monoglyme",
        "Dimethyl glycol",
        "Ethylene glycol dimethyl ether",
        "Dimethyl cellosolve",
    ],
    
    "Water": ["H2O", "Deionized water", "DI water", "Aqueous buffer", "Aqueous"],
}

# Acetonitrile x
# Water x
# Methanol x
# Dimethylformamide x
# Acetone x
# Hexane x
# Tetrahydrofuran x
# Chloroform x
# Heptane x
# Dimethoxyethane x
# Tetrachloromethane x
# Ethyl Acetate x
# Methyl Ethyl Ketone x
# 2-Propanol x
# 1,4-Dioxane x


_SOLVENT_AMBIGUOUS_ALIASES = {}


def _normalize_solvent_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _build_solvent_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    canonical_display: Dict[str, str] = {}
    alias_targets: Dict[str, set[str]] = defaultdict(set)

    for canonical, aliases in _SOLVENT_CANONICAL_TO_ALIASES.items():
        canonical_key = _normalize_solvent_key(canonical)
        if not canonical_key:
            raise ValueError(f"Empty canonical solvent key: {canonical!r}")
        existing = canonical_display.get(canonical_key)
        if existing and existing != canonical:
            raise ValueError(
                f"Canonical solvent key collision: {existing!r} and {canonical!r}"
            )
        canonical_display[canonical_key] = canonical
        for value in (canonical, *aliases):
            alias_key = _normalize_solvent_key(value)
            if alias_key:
                alias_targets[alias_key].add(canonical_key)

    conflicts = {
        alias_key: target_keys
        for alias_key, target_keys in alias_targets.items()
        if len(target_keys) > 1
    }
    if conflicts:
        raise ValueError(f"Unresolved solvent alias collisions: {sorted(conflicts)}")

    alias_to_canonical = {
        alias_key: next(iter(target_keys))
        for alias_key, target_keys in alias_targets.items()
    }
    return alias_to_canonical, canonical_display


_SOLVENT_ALIAS_TO_CANONICAL, _SOLVENT_CANONICAL_DISPLAY = _build_solvent_lookup()

_NEAR_CRITICAL_RE = re.compile(
    r"\s*\(?\s*near[\s_-]*crit(?:ical)?\s*\)?\s*",
    re.IGNORECASE,
)
_SOLVENT_CONCENTRATION_PREFIX_RE = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(?:%|mM|M|mol(?:ar)?|g/?L|mg/?mL)\s+",
    re.IGNORECASE,
)


def _resolve_solvent_name(value: Any) -> Tuple[Any, Optional[str], bool]:
    if not isinstance(value, str):
        return value, None, False
    text = value.strip()
    if not text:
        return text, None, False

    qualifier = "near crit" if _NEAR_CRITICAL_RE.search(text) else None
    base_text = _NEAR_CRITICAL_RE.sub(" ", text).strip(" ,;()")
    candidates = [base_text]

    without_parenthetical = re.sub(r"\([^()]*\)", "", base_text).strip()
    if without_parenthetical and without_parenthetical != base_text:
        candidates.append(without_parenthetical)
    for match in re.finditer(r"\(([^()]+)\)", base_text):
        candidates.append(match.group(1).strip())

    without_concentration = _SOLVENT_CONCENTRATION_PREFIX_RE.sub("", base_text).strip()
    if without_concentration and without_concentration != base_text:
        candidates.append(without_concentration)

    seen = set()
    for candidate in candidates:
        key = _normalize_solvent_key(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        mapped_key = _SOLVENT_ALIAS_TO_CANONICAL.get(key)
        if mapped_key:
            return _SOLVENT_CANONICAL_DISPLAY[mapped_key], qualifier, True
    return base_text, qualifier, False


def _normalize_solvent_name(value: Any) -> Any:
    canonical, _, _ = _resolve_solvent_name(value)
    return canonical


def _split_solvent_commas(value: str) -> List[str]:
    comma_positions = [index for index, character in enumerate(value) if character == ","]
    if not comma_positions:
        return [value.strip()] if value.strip() else []

    memo: Dict[int, Optional[List[str]]] = {}

    def partition(start: int) -> Optional[List[str]]:
        if start in memo:
            return memo[start]

        boundaries = [position for position in comma_positions if position >= start]
        boundaries.append(len(value))
        for boundary in boundaries:
            candidate = value[start:boundary].strip()
            if not candidate or not _resolve_solvent_name(candidate)[2]:
                continue
            if boundary == len(value):
                memo[start] = [candidate]
                return memo[start]
            remainder = partition(boundary + 1)
            if remainder:
                memo[start] = [candidate, *remainder]
                return memo[start]

        memo[start] = None
        return None

    resolved = partition(0)
    return resolved if resolved and len(resolved) > 1 else [value.strip()]


def _split_solvent_item(value: Any) -> List[str]:
    text = str(value).strip()
    if not text:
        return []

    _, _, direct_match = _resolve_solvent_name(text)
    if direct_match:
        return [text]

    comma_parts = _split_solvent_commas(text)
    if len(comma_parts) > 1:
        return comma_parts

    for pattern in (r"\s*;\s*", r"\s*/\s*", r"\s+and\s+", r"\s*&\s*", r"\s+\+\s+"):
        parts = [part.strip() for part in re.split(pattern, text, flags=re.IGNORECASE) if part.strip()]
        if len(parts) > 1 and all(_resolve_solvent_name(part)[2] for part in parts):
            return parts
    return [text]


def canonicalize_solvent_list(solvents: Any) -> List[str]:
    if solvents is None:
        return []
    raw_items = solvents if isinstance(solvents, (list, tuple)) else [solvents]
    normalized: List[str] = []
    seen = set()

    for raw_item in raw_items:
        text = str(raw_item).strip()
        if not text:
            continue
        water_match = re.match(r"water\s+with\s+(.+)", text, re.IGNORECASE)
        if water_match:
            modifier = water_match.group(1).strip()
            modifier_canonical, _, modifier_known = _resolve_solvent_name(modifier)
            components = ["Water"]
            if modifier_known and not _SOLVENT_CONCENTRATION_PREFIX_RE.match(modifier):
                components.append(modifier_canonical)
        else:
            components = _split_solvent_item(text)
        for component in components:
            canonical, _, _ = _resolve_solvent_name(component)
            canonical_text = str(canonical).strip()
            if canonical_text.lower() in {
                "solvent",
                "solvents",
                "n/a",
                "na",
                "none",
                "null",
            }:
                continue
            key = _normalize_solvent_key(canonical_text)
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(canonical_text)
    return normalized


def _parse_first_number(text: str) -> Optional[float]:
    match = re.search(r"\d+(?:\.\d+)?", str(text))
    if not match:
        return None
    return float(match.group(0))


def _is_nullish(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "null", "none"}


def _count_mobile_phase_solvents(cond: Dict[str, Any]) -> int:
    solvents = cond.get("mobile_phase_solvents")
    if not solvents or not isinstance(solvents, list):
        return 0
    return len([s for s in solvents if str(s).strip()])


def _extract_ratio_values(val_str: str, separator: str) -> Optional[List[float]]:
    sep_pattern = rf"\s*{re.escape(separator)}\s*"
    parts = [p.strip() for p in re.split(sep_pattern, val_str)]
    if len(parts) < 2:
        return None

    values = []
    for part in parts:
        if not re.fullmatch(r"\d+(?:\.\d+)?%?", part):
            return None
        values.append(float(part.rstrip("%")))
    return values


def _normalize_ratio_units(cond: Dict[str, Any]) -> None:
    val = cond.get("mobile_phase_ratio_units")
    if "mobile_phase_ratio_units_raw" not in cond:
        cond["mobile_phase_ratio_units_raw"] = val
    if _is_nullish(val):
        cond["mobile_phase_ratio_units"] = None
        return

    val_str = str(val).strip()
    val_lower = val_str.lower()

    # Strip everything that usually appears after the solvent field extraction
    val_clean = re.sub(r"\(.*?\)", "", val_lower).strip()
    val_clean = re.sub(r"[\s,_]", "", val_clean)
    val_clean = val_clean.replace(",", "").replace(".", "").replace("by", "")

    if any(
        token in val_clean
        for token in ("w/v", "v/w", "wt/vol", "vol/wt", "weight/volume", "volume/weight")
    ):
        cond["mobile_phase_ratio_units"] = None
    elif (
        "wt" in val_clean
        or "weight" in val_clean
        or "w/w" in val_clean
    ):
        cond["mobile_phase_ratio_units"] = "w/w"
    elif "v/v" in val_clean or "vol" in val_clean or "byvolume" in val_clean:
        cond["mobile_phase_ratio_units"] = "v/v"
    elif val_clean in {"%v", "v", "vv"}:
        cond["mobile_phase_ratio_units"] = "v/v"
    elif val_clean == "%":
        cond["mobile_phase_ratio_units"] = None
    else:
        warnings.warn(f"Unrecognized ratio unit: {val}")
        cond["mobile_phase_ratio_units"] = None


def _normalize_ratio(cond: Dict[str, Any]) -> None:
    """Backward-compatible entry point for the current ratio parser."""
    _normalize_ratio_v2(cond)


def _normalize_ratio_v2(cond: Dict[str, Any]) -> None:
    val = cond.get("mobile_phase_ratio")
    if "mobile_phase_ratio_raw" not in cond:
        cond["mobile_phase_ratio_raw"] = val
    cond["mobile_phase_ratio_components"] = None
    cond["mobile_phase_ratio_min"] = None
    cond["mobile_phase_ratio_max"] = None
    solvent_count = _count_mobile_phase_solvents(cond)

    if _is_nullish(val):
        return

    val_str = str(val).strip()
    normalized = re.sub(r"[\u2013\u2014\u2212-]", "-", val_str)
    compact = normalized.strip()

    if re.fullmatch(r"\d+(?:\.\d+)?%?", compact):
        f1 = float(compact.rstrip("%"))
        if solvent_count != 1:
            cond["mobile_phase_ratio_components"] = [f1, round(100.0 - f1, 2)]
        else:
            cond["mobile_phase_ratio_components"] = [f1]
        return

    for sep in (":", "/", "-"):
        vals = _extract_ratio_values(compact, sep)
        if vals is None:
            continue

        relative_ratio = sep in {":", "/"} or (sep == "-" and len(vals) >= 3)
        if relative_ratio and sum(vals) > 0 and abs(sum(vals) - 100.0) > 0.01:
            total = sum(vals)
            vals = [round(value * 100.0 / total, 6) for value in vals]

        if len(vals) == 2:
            if sep == "-" and abs(sum(vals) - 100.0) > 2.0:
                cond["mobile_phase_ratio_min"] = vals[0]
                cond["mobile_phase_ratio_max"] = vals[1]
            else:
                cond["mobile_phase_ratio_components"] = vals
            return

        cond["mobile_phase_ratio_components"] = vals
        return

    # Handle mixed strings like "43.4% THF : 56.6% n-hexane".
    # If exactly two numeric values exist and it is not a gradient range,
    # treat them as component percentages.
    if "to" not in compact.lower() and "gradient" not in compact.lower():
        nums = re.findall(r"\d+(?:\.\d+)?", compact)
        if len(nums) == 2:
            vals = [float(x) for x in nums]
            cond["mobile_phase_ratio_components"] = vals
            return

    if "and" in compact.lower() or "," in compact:
        floats = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", compact)]
        if len(floats) >= 2:
            if sum(floats) > 100.0:
                warnings.warn(f"Ternary components sum > 100, cannot parse: {val_str}")
                return
            if len(floats) >= 3 and sum(floats) > 0 and abs(sum(floats) - 100.0) > 0.01:
                total = sum(floats)
                floats = [round(value * 100.0 / total, 6) for value in floats]
            cond["mobile_phase_ratio_components"] = floats
            return

    f1 = _parse_first_number(compact)
    if f1 is not None and len(re.findall(r"\d+(?:\.\d+)?", compact)) == 1:
        if solvent_count != 1:
            cond["mobile_phase_ratio_components"] = [f1, round(100.0 - f1, 2)]
        else:
            cond["mobile_phase_ratio_components"] = [f1]
        return

    warnings.warn(f"Unparseable ratio: {val_str}")


def _flow_rate_factor_to_ml_per_min(value: str) -> Optional[float]:
    text = (
        value.lower()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("\u2212", "-")
        .replace("\u207b", "-")
    )
    if re.search(r"\bu\s*l\b", text):
        volume_factor = 0.001
    elif re.search(r"\bm\s*l\b", text):
        volume_factor = 1.0
    elif re.search(r"\bl\b", text):
        volume_factor = 1000.0
    elif not re.search(r"[a-z]", text):
        volume_factor = 1.0
    else:
        return None

    if re.search(
        r"(?:/|per\s+)(?:h|hr|hour)s?\b|(?:h|hr|hour)s?\s*\^?\s*-\s*[1¹]",
        text,
    ):
        time_factor = 1.0 / 60.0
    elif re.search(
        r"(?:/|per\s+)(?:s|sec|second)s?\b|(?:s|sec|second)s?\s*\^?\s*-\s*[1¹]",
        text,
    ):
        time_factor = 60.0
    elif re.search(r"(?:/|per\s+)(?:m|min|minute)s?\b|min\s*[-^]?[1¹]", text):
        time_factor = 1.0
    elif "/" in text or re.search(r"\bper\b", text):
        return None
    else:
        time_factor = 1.0
    return volume_factor * time_factor


def _normalize_flow_rate(cond: Dict[str, Any]) -> None:
    val = cond.get("flow_rate")
    if "flow_rate_raw" not in cond:
        cond["flow_rate_raw"] = val
    cond["flow_rate_ml_per_min"] = None
    cond["flow_rate_min_ml_per_min"] = None
    cond["flow_rate_max_ml_per_min"] = None

    if _is_nullish(val):
        return

    val_str = str(val).strip()
    factor = _flow_rate_factor_to_ml_per_min(val_str)
    if factor is None:
        warnings.warn(f"Unrecognized flow rate unit: {val_str}")
        return

    try:
        number = r"[+-]?\d+(?:\.\d+)?"
        range_match = re.search(
            rf"({number})\s*(?:to|[-\u2013\u2014\u2212])\s*({number})",
            val_str,
            re.IGNORECASE,
        )
        if range_match:
            cond["flow_rate_min_ml_per_min"] = float(range_match.group(1)) * factor
            cond["flow_rate_max_ml_per_min"] = float(range_match.group(2)) * factor
            return

        match = re.search(number, val_str)
        if match:
            cond["flow_rate_ml_per_min"] = float(match.group(0)) * factor
    except Exception as exc:
        warnings.warn(f"Error parsing flow rate '{val_str}': {exc}")


_PORE_VALUE_RE = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\s*(angstroms?|\u00c5|A|nm|[u\u00b5\u03bc]m)?",
    re.IGNORECASE,
)


def _pore_value_to_angstrom(value: float, unit: Optional[str]) -> float:
    normalized_unit = (unit or "A").lower().replace("µ", "u").replace("μ", "u")
    if normalized_unit == "nm":
        return value * 10.0
    if normalized_unit == "um":
        return value * 10000.0
    return value


def _parse_pore_values(value: str) -> List[float]:
    matches = list(_PORE_VALUE_RE.finditer(value))
    if not matches:
        return []
    units = [match.group(2) for match in matches]
    explicit_units = [unit for unit in units if unit]
    inherited_unit = explicit_units[0] if len(set(explicit_units)) == 1 else None
    return [
        _pore_value_to_angstrom(
            float(match.group(1)),
            match.group(2) or inherited_unit,
        )
        for match in matches
    ]


def _normalize_pore_size(cond: Dict[str, Any]) -> None:
    val = cond.get("pore_size")
    if "pore_size_raw" not in cond:
        cond["pore_size_raw"] = val
    cond["pore_size_angstrom"] = None
    cond["pore_size_min_angstrom"] = None
    cond["pore_size_max_angstrom"] = None

    if _is_nullish(val):
        return

    val_str = str(val).strip()
    val_str = val_str.replace("\u00c3\u2026", "\u00c5").replace("\u00c2\u00b5", "\u00b5")
    val_str = re.sub(r"(\d)\s*-\s*(Å|A|nm|[uµμ]m)\b", r"\1 \2", val_str)

    try:
        number = r"[+-]?\d+(?:\.\d+)?"
        unit = r"(?:angstroms?|\u00c5|A|nm|[u\u00b5\u03bc]m)?"
        range_match = re.search(
            rf"({number})\s*({unit})\s*(?:to|[-\u2013\u2014\u2212])\s*({number})\s*({unit})",
            val_str,
            re.IGNORECASE,
        )
        if range_match:
            unit_one = range_match.group(2) or range_match.group(4)
            unit_two = range_match.group(4) or range_match.group(2)
            cond["pore_size_min_angstrom"] = _pore_value_to_angstrom(
                float(range_match.group(1)),
                unit_one,
            )
            cond["pore_size_max_angstrom"] = _pore_value_to_angstrom(
                float(range_match.group(3)),
                unit_two,
            )
            return

        values = _parse_pore_values(val_str)
        if not values:
            return
        if len(values) == 1:
            cond["pore_size_angstrom"] = values[0]
        else:
            cond["pore_size_angstrom"] = values
    except Exception as exc:
        warnings.warn(f"Error parsing pore size '{val_str}': {exc}")


def _normalize_temperature(cond: Dict[str, Any]) -> None:
    val = cond.get("temperature_celsius")
    if "temperature_celsius_raw" not in cond:
        cond["temperature_celsius_raw"] = val
    raw_val = cond.get("temperature_celsius_raw")
    parse_val = raw_val if _is_nullish(val) and not _is_nullish(raw_val) else val
    cond["temperature_min_celsius"] = None
    cond["temperature_max_celsius"] = None
    if _is_nullish(parse_val):
        cond["temperature_celsius"] = None
        return

    val_str = str(parse_val).strip()
    try:
        number = r"[+-]?\d+(?:\.\d+)?"
        range_match = re.search(
            rf"({number})\s*(?:to|[-\u2013\u2014\u2212])\s*({number})",
            val_str,
            re.IGNORECASE,
        )
        if range_match:
            cond["temperature_celsius"] = None
            cond["temperature_min_celsius"] = float(range_match.group(1))
            cond["temperature_max_celsius"] = float(range_match.group(2))
            return

        match = re.search(number, val_str)
        if match:
            cond["temperature_celsius"] = float(match.group(0))
    except Exception as exc:
        warnings.warn(f"Error parsing temperature '{val_str}': {exc}")


def _normalize_column_mode(cond: Dict[str, Any]) -> None:
    val = cond.get("column_mode")
    if "column_mode_raw" not in cond:
        cond["column_mode_raw"] = val
    if _is_nullish(val):
        cond["column_mode"] = None
        return

    val_str = re.sub(r"[-_]+", " ", str(val).strip().lower())
    val_str = re.sub(r"\s+", " ", val_str).strip()

    if val_str in ("reverse", "reverse phase", "reversed", "reversed phase", "rp"):
        cond["column_mode"] = "Reversed Phase"
    elif val_str in ("normal", "normal phase", "np"):
        cond["column_mode"] = "Normal Phase"
    elif val_str in ("hilic", "hydrophilic interaction", "hydrophilic interaction chromatography"):
        cond["column_mode"] = "HILIC"
    elif val_str in ("sec", "size exclusion", "size exclusion chromatography"):
        cond["column_mode"] = "SEC"
    elif val_str in ("ion exchange", "ion exchange chromatography", "iec"):
        cond["column_mode"] = "Ion Exchange"
    else:
        warnings.warn(f"Unrecognized column mode: {val}")


def _normalize_architecture(cond: Dict[str, Any]) -> None:
    val = cond.get("architecture")
    if "architecture_raw" not in cond:
        cond["architecture_raw"] = val
    if _is_nullish(val):
        cond["architecture"] = None
        return

    val_str = str(val).strip().lower()

    if val_str in ARCHITECTURE_MAP:
        new_val = ARCHITECTURE_MAP[val_str]
        if new_val != val:
            if "architecture_raw" not in cond:
                cond["architecture_raw"] = val
            cond["architecture"] = new_val
        return

    for key, mapped in ARCHITECTURE_MAP.items():
        if val_str.startswith(key):
            if mapped != val:
                if "architecture_raw" not in cond:
                    cond["architecture_raw"] = val
                cond["architecture"] = mapped
            return

    if "block" in val_str:
        if "architecture_raw" not in cond:
            cond["architecture_raw"] = val
        cond["architecture"] = "block copolymer"
        return

    warnings.warn(f"Unrecognized architecture: {val}")

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
    r"\b\d+(?:\.\d+)?\s*(?:mM|M|mol(?:ar)?|mmol/?L|mol/?L|g/?L|mg/?mL|%)\b",
    re.IGNORECASE,
)


def _append_unique_text(existing: Any, value: str) -> str:
    values = [part.strip() for part in str(existing or "").split(",") if part.strip()]
    if value and value.casefold() not in {part.casefold() for part in values}:
        values.append(value)
    return ", ".join(values)


def _record_aqueous_modifier(cond: Dict[str, Any], modifier: str) -> None:
    aqueous = cond.get("aqueous_parameters")
    if not isinstance(aqueous, dict):
        aqueous = {}
        cond["aqueous_parameters"] = aqueous

    modifier_lower = modifier.lower()
    matched_salt = None
    for token, canonical in _AQUEOUS_SALT_NAMES.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", modifier_lower):
            matched_salt = canonical
            break

    if matched_salt:
        aqueous["salt_added"] = True
        aqueous["salt_type"] = _append_unique_text(
            aqueous.get("salt_type"),
            matched_salt,
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


def _normalize_solvents(cond: Dict[str, Any]) -> None:
    if "aqueous_parameters_raw" not in cond:
        cond["aqueous_parameters_raw"] = copy.deepcopy(
            cond.get("aqueous_parameters")
        )

    solvents = cond.get("mobile_phase_solvents")
    if solvents is None:
        return

    if "mobile_phase_solvents_raw" not in cond:
        cond["mobile_phase_solvents_raw"] = copy.deepcopy(solvents)

    raw_items = solvents if isinstance(solvents, (list, tuple)) else [solvents]
    previous_qualifiers = cond.get("mobile_phase_solvent_qualifiers")
    if not isinstance(previous_qualifiers, list):
        previous_qualifiers = []

    normalized_solvents: List[str] = []
    normalized_qualifiers: List[Optional[str]] = []
    solvent_ambiguities: List[Dict[str, Any]] = []
    modifiers: List[str] = []
    seen = set()

    for item_index, raw_item in enumerate(raw_items):
        solvent_text = str(raw_item).strip()
        if not solvent_text:
            continue

        water_match = re.match(r"water\s+with\s+(.+)", solvent_text, re.IGNORECASE)
        if water_match:
            modifier = water_match.group(1).strip()
            modifier_canonical, _, modifier_known = _resolve_solvent_name(modifier)
            has_concentration = bool(_SOLVENT_CONCENTRATION_PREFIX_RE.match(modifier))
            components: List[Any] = ["Water"]
            if modifier_known and not has_concentration:
                components.append(modifier_canonical)
            elif modifier:
                modifiers.append(modifier)
        else:
            components = _split_solvent_item(solvent_text)

        for component_index, component in enumerate(components):
            canonical, qualifier, _ = _resolve_solvent_name(component)
            canonical_text = str(canonical).strip()
            if not canonical_text:
                continue
            if canonical_text.lower() in {
                "solvent",
                "solvents",
                "n/a",
                "na",
                "none",
                "null",
            }:
                continue

            ambiguous_key = _normalize_solvent_key(canonical_text)
            for alias, candidates in _SOLVENT_AMBIGUOUS_ALIASES.items():
                if ambiguous_key == _normalize_solvent_key(alias):
                    solvent_ambiguities.append(
                        {
                            "raw": str(component).strip(),
                            "canonical_candidates": list(candidates),
                        }
                    )
                    break

            if (
                qualifier is None
                and len(components) == 1
                and component_index == 0
                and item_index < len(previous_qualifiers)
            ):
                qualifier = previous_qualifiers[item_index]

            canonical_key = _normalize_solvent_key(canonical_text)
            if not canonical_key or canonical_key in seen:
                continue
            seen.add(canonical_key)
            normalized_solvents.append(canonical_text)
            normalized_qualifiers.append(qualifier)

    cond["mobile_phase_solvents"] = normalized_solvents
    cond["mobile_phase_solvent_qualifiers"] = normalized_qualifiers
    if solvent_ambiguities:
        cond["mobile_phase_solvent_ambiguities"] = solvent_ambiguities
    for modifier in modifiers:
        _record_aqueous_modifier(cond, modifier)


def _normalize_year(cond: Dict[str, Any]) -> None:
    val = cond.get("publication_year")
    if "publication_year_raw" not in cond:
        cond["publication_year_raw"] = val
    if _is_nullish(val):
        cond["publication_year"] = None
        return

    val_str = str(val).strip()
    try:
        match = re.search(r"(\d{4})", val_str)
        if match:
            cond["publication_year"] = int(match.group(1))
    except Exception:
        pass


def standardize_condition(cond: dict) -> dict:
    """
    Takes one raw consensus condition dict.
    Returns a new dict (never mutates input) with all fields standardized.
    """
    out = copy.deepcopy(cond)
    _normalize_ratio_units(out)
    _normalize_solvents(out)
    _normalize_ratio(out)
    _normalize_flow_rate(out)
    _normalize_pore_size(out)
    _normalize_temperature(out)
    _normalize_column_mode(out)
    _normalize_polymer_fields(out)
    _normalize_architecture(out)
    _normalize_year(out)
    return out


def standardize_file(input_path: Path, output_path: Path) -> int:
    """
    Reads one *_consensus.json, standardizes all conditions, writes *_standardized.json.
    Returns the number of conditions written.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conditions = data.get("extracted_data", {}).get("conditions", [])
        standardized_conditions = [standardize_condition(c) for c in conditions]

        output = copy.deepcopy(data)
        metadata = output.setdefault("metadata", {})
        metadata.setdefault(
            "source_pdf",
            input_path.stem.replace("_consensus", ""),
        )
        metadata["standardized_by"] = "pipeline/standardizer.py"
        metadata["standardization_date"] = datetime.now(timezone.utc).isoformat()
        summary = output.setdefault("summary", {})
        summary["total_conditions"] = len(standardized_conditions)
        extracted_data = output.setdefault("extracted_data", {})
        extracted_data["conditions"] = standardized_conditions

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return len(standardized_conditions)
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return 0


def standardize_all(consensus_dir: Path, output_dir: Path) -> None:
    """
    Walks consensus_dir recursively for *_consensus.json files.
    Mirrors directory structure under output_dir.
    Prints a summary per file and a grand total.
    """
    total_conds = 0
    total_files = 0

    for input_path in consensus_dir.rglob("*_consensus.json"):
        rel_path = input_path.relative_to(consensus_dir)
        output_name = f"{input_path.stem.replace('_consensus', '')}_standardized.json"
        output_path = output_dir / rel_path.parent / output_name

        count = standardize_file(input_path, output_path)
        print(f"Processing {rel_path} ... {count} conditions")

        total_conds += count
        total_files += 1

    print("=" * 54)
    print(f"Standardized {total_conds} conditions across {total_files} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standardize consensus JSON files with the PolyCrit vocabulary."
    )
    parser.add_argument(
        "consensus_dir",
        nargs="?",
        default="results/consensus",
        help="Directory containing consensus JSON files",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="results/standardized",
        help="Directory for standardized JSON files",
    )
    arguments = parser.parse_args()
    standardize_all(
        consensus_dir=Path(arguments.consensus_dir),
        output_dir=Path(arguments.output_dir),
    )
