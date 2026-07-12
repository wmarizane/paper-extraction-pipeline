"""Export standardized JSON outputs to a flat summary CSV."""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from pipeline import standardizer as _standardizer  # type: ignore
except Exception:  # pragma: no cover - fallback when package import path differs
    import standardizer as _standardizer  # type: ignore


def _clean(val):
    """Convert None and null-like values to an empty string."""
    if val is None:
        return ""
    if isinstance(val, float) and val != val:
        return ""
    if isinstance(val, str):
        value = val.strip()
        if value.lower() in ("null", "none", ""):
            return ""
        return value
    if isinstance(val, list):
        cleaned = [_clean(v) for v in val]
        deduped = []
        seen = set()
        for item in cleaned:
            if not item:
                continue
            key = item.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return ", ".join(deduped)
    return str(val)


def _preserve_raw_value(val: Any) -> str:
    """Serialize an original value without deduplicating or reordering it."""
    if val is None:
        return ""
    if isinstance(val, float) and val != val:
        return ""
    if isinstance(val, (list, tuple, dict)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _raw_condition_value(condition: Dict[str, Any], field: str) -> Any:
    """Return a preserved raw field, falling back for older standardized JSON."""
    raw_field = f"{field}_raw"
    if raw_field in condition:
        return condition.get(raw_field)
    return condition.get(field)


_POLYMER_REFERENCE_ALIASES = getattr(
    _standardizer,
    "_POLYMER_CANONICAL_TO_ALIASES",
    {
        "Poly(ethylene glycol)": ["PEG", "PEO", "mPEG", "PEG-MME", "PEG-DME", "MeO-PEG", "MeO-PEG-DME"],
        "Poly(propylene glycol)": ["PPG"],
        "Poly(L-lactide)": ["PLLA", "PLA"],
    },
)

_POLYMER_ARCHITECTURE_TOKENS = (
    "linear homopolymer",
    "random copolymer",
    "block copolymer",
    "diblock copolymer",
    "triblock copolymer",
    "tetrablock copolymer",
    "single component",
    "linear",
    "cyclic",
    "ring",
    "star",
    "graft",
    "random",
    "block",
    "copolymer",
    "homopolymer",
)


def _normalize_polymer_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _append_unique(values: list[str], value: str, seen: set[str], normalize: bool = True) -> None:
    item = _clean(value)
    if not item:
        return
    key = _normalize_polymer_key(item) if normalize else item.lower()
    if not key:
        return
    if key in seen:
        return
    seen.add(key)
    values.append(item)


def _register_polymer_alias(
    alias_to_canonical: dict[str, str],
    canonical_aliases: dict[str, list[str]],
    canonical_display: dict[str, str],
    alias: str,
    canonical: str,
) -> None:
    alias_key = _normalize_polymer_key(alias)
    canonical_key = _normalize_polymer_key(canonical)
    if not alias_key or not canonical_key:
        return

    canonical_display.setdefault(canonical_key, _clean(canonical))
    alias_to_canonical.setdefault(alias_key, canonical_key)
    aliases = canonical_aliases.setdefault(canonical_key, [])
    seen = {_normalize_polymer_key(value) for value in aliases}
    _append_unique(aliases, alias, seen)


def _split_alternates(raw: str) -> list[str]:
    if not raw:
        return []

    result: list[str] = []
    for token in re.split(r"\s*[,;]\s*", str(raw).strip()):
        token = token.strip()
        if not token:
            continue
        for sub_token in re.split(r"\s+and\s+", token, flags=re.IGNORECASE):
            cleaned = _clean(sub_token)
            if cleaned:
                result.append(cleaned)
    return result


def _load_polymer_reference_dictionary(
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """
    Build two lookup maps:
      - alias_to_canonical_key
      - canonical_aliases_by_canonical_key
      - canonical_display_by_key

    Kept small and deterministic; duplicates are de-duped.
    """
    alias_to_canonical: dict[str, str] = {}
    canonical_aliases: dict[str, list[str]] = {}
    canonical_display: dict[str, str] = {}

    for canonical, aliases in _POLYMER_REFERENCE_ALIASES.items():
        canonical = _clean(canonical)
        if not canonical:
            continue

        canonical_key = _normalize_polymer_key(canonical)
        if not canonical_key:
            continue

        _register_polymer_alias(
            alias_to_canonical,
            canonical_aliases,
            canonical_display,
            canonical,
            canonical,
        )

        for alias in aliases:
            _register_polymer_alias(
                alias_to_canonical,
                canonical_aliases,
                canonical_display,
                alias,
                canonical,
            )

    # Use the standardizer's collision-checked lookup so JSON and CSV cannot
    # disagree for ambiguous aliases such as dPS, PC, and Epoxy Resin.
    shared_aliases = getattr(_standardizer, "_POLYMER_ALIAS_TO_CANONICAL", None)
    shared_display = getattr(_standardizer, "_POLYMER_CANONICAL_DISPLAY", None)
    if isinstance(shared_aliases, dict) and isinstance(shared_display, dict):
        alias_to_canonical = dict(shared_aliases)
        canonical_display = dict(shared_display)

    return alias_to_canonical, canonical_aliases, canonical_display


def _strip_polymer_architecture_terms(
    value: str,
    architecture: Optional[str] = None,
) -> str:
    text = _clean(value)
    if not text:
        return ""

    normalized = _clean(architecture)
    architecture_tokens = tuple(dict.fromkeys(_POLYMER_ARCHITECTURE_TOKENS))
    for token in architecture_tokens:
        text = re.sub(
            rf"^\s*{re.escape(token)}\b[\s,\-/:;]*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"\s*\(?\b{re.escape(token)}\b\)?\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )

    if normalized:
        text = re.sub(
            rf"^\s*{re.escape(normalized)}\b[\s,\-/:;]*",
            "",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"^\s*[,/;:\-\s]+|[,/;:\-\s]+$", "", text)
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_polymer_case(value: str) -> str:
    if not value:
        return ""
    if "(" in value or re.search(r"[A-Z0-9]", value):
        return value
    if value.islower():
        return " ".join(part[:1].upper() + part[1:] for part in value.split())
    return value


def _extract_polymer_candidates(raw: str) -> list[str]:
    candidates: list[str] = []
    if not raw:
        return candidates

    seen = set()
    for part in _split_by_top_level_comma(raw):
        part = _clean(part)
        if not part:
            continue
        _append_unique(candidates, part, seen, normalize=False)
        without_parentheses = re.sub(r"\([^()]+\)", "", part).strip()
        if without_parentheses:
            _append_unique(
                candidates,
                _strip_polymer_architecture_terms(without_parentheses),
                seen,
                normalize=False,
            )
        for match in re.finditer(r"\(([^()]+)\)", part):
            for sub in re.split(r"[;/,]", match.group(1)):
                token = _clean(sub)
                if not token:
                    continue
                _append_unique(candidates, token, seen, normalize=False)
                _append_unique(
                    candidates,
                    _strip_polymer_architecture_terms(token),
                    seen,
                    normalize=False,
                )
    return candidates


def _derive_polymer_columns(
    analyte_polymer: Optional[str],
    critical_component: Optional[str],
    architecture: Optional[str],
    alias_to_canonical: dict[str, str],
    canonical_aliases: dict[str, list[str]],
    canonical_display: dict[str, str],
) -> tuple[str, str, str]:
    analyte = _clean(analyte_polymer)
    critical = _clean(critical_component)
    polymer_raw = analyte or critical
    if not polymer_raw:
        return "", "", ""

    candidate_list = (
        _extract_polymer_candidates(polymer_raw)
        + _extract_polymer_candidates(critical)
    )
    lookup_candidates: list[str] = []
    lookup_seen = set()
    for candidate in candidate_list:
        _append_unique(lookup_candidates, candidate, lookup_seen, normalize=False)
        stripped = _strip_polymer_architecture_terms(candidate, architecture)
        if stripped:
            _append_unique(lookup_candidates, stripped, lookup_seen, normalize=False)

    primary_candidates = [c for c in lookup_candidates if not _looks_like_alias(c)]
    alias_candidates = [c for c in lookup_candidates if _looks_like_alias(c)]

    def _find_dictionary_hit(candidates: list[str]) -> str:
        for candidate in candidates:
            if not candidate:
                continue
            key = _normalize_polymer_key(candidate)
            if key and key in alias_to_canonical:
                return alias_to_canonical[key]
        return ""

    # Prefer explicit full polymer labels over short acronym-like tokens.
    canonical_key = _find_dictionary_hit(primary_candidates)
    if not canonical_key:
        canonical_key = _find_dictionary_hit(alias_candidates)

    # Fallback when dictionary mapping is absent.
    if canonical_key:
        polymer = _clean(canonical_display.get(canonical_key, ""))
        if not polymer:
            polymer = _clean(alias_to_canonical.get(canonical_key, ""))
        alias_values = []
        alias_seen: set[str] = set()
        source_alias_by_key: dict[str, str] = {}
        for alias in canonical_aliases.get(canonical_key, []):
            if not alias:
                continue
            alias_key = _normalize_polymer_key(alias)
            polymer_key = _normalize_polymer_key(polymer)
            if alias_key == polymer_key:
                continue
            source_alias_by_key[_normalize_polymer_key(alias)] = alias

        # Emit only aliases that were present in the extracted text. The
        # reference dictionary identifies equivalence but must not manufacture
        # evidence that the model never extracted.
        for candidate in lookup_candidates:
            source_alias = source_alias_by_key.get(_normalize_polymer_key(candidate))
            if source_alias:
                _append_unique(alias_values, source_alias, alias_seen, normalize=False)

        _, extracted_aliases = _extract_polymer_aliases(analyte_polymer, critical_component)
        for alias in extracted_aliases.split(","):
            alias = _clean(alias)
            if not alias:
                continue
            alias_key = _normalize_polymer_key(alias)
            polymer_key = _normalize_polymer_key(polymer)
            source_alias = source_alias_by_key.get(alias_key)
            if source_alias:
                _append_unique(alias_values, source_alias, alias_seen, normalize=False)
                continue
            if alias_key == polymer_key or alias_key in polymer_key:
                continue
            if not _alias_compatible(polymer, alias):
                continue
            _append_unique(alias_values, alias, alias_seen, normalize=False)

        alternate = ", ".join(alias_values)
    else:
        fallback_polymer, extracted_aliases = _extract_polymer_aliases(
            analyte_polymer,
            critical_component,
        )
        fallback_polymer = _strip_polymer_architecture_terms(fallback_polymer, architecture)
        if not fallback_polymer:
            fallback_polymer = _strip_polymer_architecture_terms(polymer_raw, architecture)
        polymer = _normalize_polymer_case(_clean(fallback_polymer))
        alternate = _clean(extracted_aliases)

    return polymer_raw, polymer, alternate


def _to_num_str(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return ""
    if isinstance(val, (int, float)):
        return str(float(val)).rstrip("0").rstrip(".")
    return str(val)


def _format_range(low: Optional[float], high: Optional[float]) -> str:
    low_s = _to_num_str(low)
    high_s = _to_num_str(high)
    if not low_s or not high_s:
        return low_s or high_s
    if abs(float(low) - float(high)) < 1e-12:
        return low_s
    return f"{low_s}-{high_s}"


def _extract_reference(source_pdf: Optional[str], fallback_stem: str) -> str:
    ref = (source_pdf or fallback_stem or "").replace(".pdf", "").strip()
    if not ref:
        return ""
    match = re.match(r"^\s*\[(\d+)\]", ref)
    if match:
        return match.group(1)
    return ref


def _normalize_ratio_unit(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower().replace(" ", "").replace(".", "")
    if not text or text in {"null", "none"}:
        return ""

    if any(
        token in text
        for token in ("w/v", "v/w", "wt/vol", "vol/wt", "weight/volume", "volume/weight")
    ):
        return ""
    if "v/" in text or "vol" in text:
        return "v/v"
    if "w/" in text or "wt" in text:
        return "w/w"

    return ""


def _ratio_list_to_string(values: Optional[Any]) -> str:
    if not isinstance(values, list):
        return ""

    parts = []
    for value in values:
        num = _to_num_str(value)
        if num:
            parts.append(num)
    return ", ".join(parts)


def _ratio_primary(cond: Dict[str, Any]) -> str:
    components = cond.get("mobile_phase_ratio_components")
    if isinstance(components, list) and components:
        ratio = _ratio_list_to_string(components)
        if ratio:
            return ratio

    raw = cond.get("mobile_phase_ratio_raw") or cond.get("mobile_phase_ratio")
    if raw is None:
        return ""

    nums = re.findall(r"\d+(?:\.\d+)?", str(raw))
    if not nums:
        return _clean(raw)

    raw_str = str(raw).strip()
    raw_clean = raw_str.lower()

    if len(nums) > 3:
        return _clean(raw_str)

    text = re.sub(r"\d+(?:\.\d+)?%?", "", raw_clean)
    text = re.sub(r"\b(v/v|w/w|wt%?|vol%?|weight|volume|percent|%)\b", "", text)
    text = re.sub(r"[\s\.\,\;\:\+\-\(\)/]", "", text)
    if re.search(r"\bto\b", raw_clean) or "gradient" in raw_clean:
        return _clean(raw_str)

    if len(nums) == 1:
        return _to_num_str(float(nums[0]))

    if any(ch.isalpha() for ch in text):
        return ", ".join(_to_num_str(float(v)) for v in nums)

    return ", ".join(_to_num_str(float(v)) for v in nums)


def _ratio_columns(cond: Dict[str, Any]) -> Tuple[str, str]:
    """Return values for (wt%, vol%)."""
    min_ratio = cond.get("mobile_phase_ratio_min")
    max_ratio = cond.get("mobile_phase_ratio_max")

    ratio_value = _ratio_list_to_string(cond.get("mobile_phase_ratio_components"))
    if not ratio_value:
        ratio_value = _format_range(min_ratio, max_ratio)
    if not ratio_value:
        ratio_value = _ratio_primary(cond)

    unit = _normalize_ratio_unit(cond.get("mobile_phase_ratio_units"))
    if not unit:
        unit = _normalize_ratio_unit(cond.get("mobile_phase_ratio_units_raw"))

    wt_ratio = ""
    vol_ratio = ""
    if ratio_value:
        if unit == "w/w":
            wt_ratio = ratio_value
        elif unit == "v/v":
            vol_ratio = ratio_value
    return wt_ratio, vol_ratio


def _format_polycrit_solvents(
    solvents: Any,
    qualifiers: Any = None,
) -> str:
    if not isinstance(solvents, list):
        return _clean(solvents)

    qualifier_list = qualifiers if isinstance(qualifiers, list) else []
    values = []
    for index, solvent in enumerate(solvents):
        value = _clean(solvent)
        if not value:
            continue
        qualifier = _clean(qualifier_list[index]) if index < len(qualifier_list) else ""
        if qualifier.casefold() == "near crit":
            value = f"{value} (near crit)"
        values.append(value)
    return ", ".join(values)


def _pore_size_for_export(cond: Dict[str, Any]) -> str:
    scalar_or_list = _clean(cond.get("pore_size_angstrom"))
    if scalar_or_list:
        return scalar_or_list
    return _format_range(
        cond.get("pore_size_min_angstrom"),
        cond.get("pore_size_max_angstrom"),
    )


def _split_by_top_level_comma(value: str) -> list[str]:
    if not value:
        return []

    parts: list[str] = []
    current = []
    depth = 0
    text = str(value)

    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            prev_char = text[i - 1] if i > 0 else ""
            next_char = text[i + 1] if i + 1 < len(text) else ""
            # Avoid splitting numeric designations (e.g. 1,4-PI) into fragments.
            if prev_char.isdigit() and next_char.isdigit():
                current.append(char)
                continue

            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    return parts


def _polymer_signature_tokens(text: str) -> set[str]:
    """Get compact tokens useful for quick polymer-family compatibility checks."""
    if not text:
        return set()

    text_lower = str(text).lower()
    stop_words = {
        "and",
        "with",
        "the",
        "block",
        "poly",
        "polymer",
        "copolymer",
        "linear",
        "random",
        "titration",
        "samples",
        "sample",
        "solvent",
        "solvents",
        "at",
    }

    tokens: set[str] = set()

    def add_tokens(candidate: str) -> None:
        cleaned = re.sub(r"[^a-z0-9-]+", " ", candidate.strip().lower())
        for chunk in cleaned.split():
            if not chunk or len(chunk) < 2:
                continue
            for part in chunk.split("-"):
                part = part.strip("-")
                if part and part not in stop_words and len(part) >= 2:
                    tokens.add(part)
            if chunk not in stop_words and len(chunk) >= 2:
                tokens.add(chunk)

    for match in re.finditer(r"\(([^()]+)\)", text_lower):
        for segment in re.split(r"[;,/]", match.group(1)):
            add_tokens(segment)

    add_tokens(re.sub(r"\([^()]*\)", " ", text_lower))
    return tokens


def _alias_compatible(main_polymer: str, alias: str) -> bool:
    main_tokens = _polymer_signature_tokens(main_polymer)
    alias_tokens = _polymer_signature_tokens(alias)
    if not main_tokens or not alias_tokens:
        return False

    if main_tokens.intersection(alias_tokens):
        return True

    for m in main_tokens:
        for a in alias_tokens:
            if m in a or a in m:
                return True
    return False


def _looks_like_alias(candidate: str) -> bool:
    token = candidate.strip().strip("()").strip()
    if not token or len(token) > 30:
        return False
    if len(token) < 2:
        return False
    if re.search(r"\s", token):
        return False
    if "/" in token or "\\" in token:
        return False
    if "(" in token or ")" in token:
        return False
    if token.lower() in {
        "and",
        "or",
        "the",
        "with",
        "polymer",
        "poly",
        "copolymer",
        "lccc",
        "solvent",
        "solvents",
        "sec",
    }:
        return False
    if token.lower() == "null":
        return False
    if not re.search(r"[A-Za-z]", token):
        return False

    # Prefer chemical-style short labels or explicit topology abbreviations.
    if re.match(r"^[A-Za-z]{1,4}$", token):
        return token.isupper()
    return any(ch.isdigit() for ch in token) or re.search(r"[A-Z][A-Za-z0-9\-]", token)


def _add_unique(values: list[str], value: str, seen: set[str], normalize: bool = False) -> None:
    if not value:
        return
    item = value.strip()
    if not item:
        return
    key = item.lower() if normalize else item.lower()
    if key in seen:
        return
    seen.add(key)
    values.append(item)


def _normalize_alias_token(token: str) -> str:
    token = token.strip()
    if len(token) > 2 and token.startswith("(") and token.endswith(")"):
        inner = token[1:-1].strip()
        if inner and "(" not in inner and ")" not in inner:
            return inner
    return token


def _extract_parenthetical_aliases(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\(([^()]+)\)", text):
        token = match.group(1).strip()
        if not token or len(token) > 30:
            continue
        if len(token) > 1 and re.search(r"\s", token):
            continue
        # Drop common non-alias textual descriptors that are not short abbreviations.
        if token.lower() in {"samples", "synthesized", "with"}:
            continue
        if _looks_like_alias(token):
            _add_unique(candidates, token, seen)
    return candidates


def _extract_polymer_aliases(
    analyte_polymer: str,
    critical_component: Optional[str],
) -> tuple[str, str]:
    analyte = _clean(analyte_polymer)
    if not analyte:
        analyte = _clean(critical_component)
    if not analyte:
        return "", ""

    parts = _split_by_top_level_comma(analyte)
    main_polymer = _clean(parts[0] if parts else analyte)

    alias_values: list[str] = []
    seen_alias: set[str] = set()

    # Parenthetical shorthand on the primary polymer is a strong alias signal.
    for alias in _extract_parenthetical_aliases(analyte):
        if _alias_compatible(main_polymer, alias):
            _add_unique(alias_values, alias, seen_alias, normalize=True)

    # For multi-polymer entries, treat short descriptors as alternate names.
    for part in parts[1:]:
        part_clean = _normalize_alias_token(part)
        if (
            _looks_like_alias(part_clean)
            and _alias_compatible(main_polymer, part_clean)
            and part_clean.lower() not in {"and", ""}
        ):
            _add_unique(alias_values, part_clean, seen_alias)

    if critical_component and critical_component.strip().lower() != main_polymer.lower():
        crit = _clean(critical_component)
        if crit:
            # Add compact, likely alias-like forms from critical component.
            for alias in _extract_parenthetical_aliases(crit):
                _add_unique(alias_values, alias, seen_alias, normalize=True)

            for part in _split_by_top_level_comma(crit):
                part_clean = _normalize_alias_token(part)
                if (
                    part_clean.lower() != main_polymer.lower()
                    and _looks_like_alias(part_clean)
                ):
                    _add_unique(alias_values, part_clean, seen_alias)

    return main_polymer, ", ".join(alias_values)


def _extract_end_groups(*texts: Any) -> str:
    combined = " ".join(_clean(t) for t in texts if _clean(t))
    if not combined:
        return ""

    text = combined.lower()
    groups: list[str] = []
    seen: set[str] = set()

    if re.search(r"\b(cooh|carboxy|carboxyl|carb\W*acid)\b", text):
        _add_unique(groups, "COOH", seen)
    if re.search(r"\b(nh2|amino|amide)\b", text):
        _add_unique(groups, "NH2", seen)
    if re.search(r"\b(cl|chloride)\b", text):
        _add_unique(groups, "Cl", seen)
    if re.search(r"\b(br|bromide)\b", text):
        _add_unique(groups, "Br", seen)

    if re.search(r"\bmeo\b|\bmPEG\b|m-?peg|mPEO|PEG-MME|PEGMEE|mpego|mpeg|mpeg|MeO-PEG|MPEG", text, flags=re.IGNORECASE):
        _add_unique(groups, "CH3", seen)
        _add_unique(groups, "OH", seen)
    if re.search(r"\b(dimethoxy|di-methoxy|methoxy)\b", text):
        _add_unique(groups, "CH3", seen)
        _add_unique(groups, "OCH3", seen)
    if re.search(r"\b(hydroxyl|hydroxy)\b", text):
        _add_unique(groups, "OH", seen)

    if re.search(r"\bch3\b", text):
        _add_unique(groups, "CH3", seen)
    if re.search(r"\b(oh|\(\s*oh\s*\)|-oh)\b", text):
        _add_unique(groups, "OH", seen)
    for m in re.finditer(r"\bh\(ch2\)\d+\b", text, flags=re.IGNORECASE):
        _add_unique(groups, m.group(0).upper(), seen)

    return ", ".join(groups)


def _parse_polymer_architecture(
    architecture: Optional[str],
    analyte_polymer: Optional[str],
    critical_component: Optional[str],
    architecture_raw: Optional[str],
) -> str:
    candidates: list[str] = []
    seen: set[str] = set()

    source = _clean(architecture)
    source_raw = _clean(architecture_raw)
    if source:
        source = source.lower()
        if source in {"single component", "single"}:
            source = ""
        if source:
            for token in [
                "cyclic",
                "ring",
                "linear",
                "diblock",
                "triblock",
                "tetrablock",
                "star",
                "graft",
                "block copolymer",
                "random copolymer",
                "telechelic",
            ]:
                if token in source and source not in seen:
                    if token == "block copolymer":
                        _append_unique(candidates, "block", seen)
                        break
                    if token == "ring":
                        _append_unique(candidates, "cyclic", seen)
                    elif token == "random copolymer":
                        _append_unique(candidates, "random copolymer", seen)
                    elif token == "telechelic":
                        _append_unique(candidates, "telechelic", seen)
                    else:
                        _append_unique(candidates, token, seen)

    if source_raw and not candidates:
        source_raw = _clean(source_raw).lower()
        if "cyclic" in source_raw or "ring" in source_raw:
            _append_unique(candidates, "cyclic", seen)

    if not candidates:
        descriptor_text = " ".join(
            _clean(v).lower()
            for v in [analyte_polymer, critical_component, architecture]
            if _clean(v)
        )
        for token in [
            ("cyclic", "cyclic"),
            ("ring", "cyclic"),
            ("linear", "linear"),
            ("star", "star"),
            ("graft", "graft"),
            ("diblock", "diblock"),
            ("triblock", "triblock"),
            ("tetrablock", "tetrablock"),
            ("random", "random copolymer"),
            ("block", "block"),
        ]:
            key, value = token
            if key in descriptor_text:
                _append_unique(candidates, value, seen)
                if key in {"cyclic", "ring", "linear"}:
                    break

    return ", ".join(candidates)


def _extract_particle_diameter_um(column_dimensions: Any) -> str:
    if not column_dimensions:
        return ""

    text = str(column_dimensions)
    match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:\u03bcm|\u00b5m|um|micro)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _to_num_str(float(match.group(1)))
    return ""


def _parse_flow(cond: Dict[str, Any]) -> str:
    min_flow = cond.get("flow_rate_min_ml_per_min")
    max_flow = cond.get("flow_rate_max_ml_per_min")
    if min_flow is not None and max_flow is not None:
        return _format_range(min_flow, max_flow)
    return _to_num_str(cond.get("flow_rate_ml_per_min"))


def _parse_temperature(cond: Dict[str, Any]) -> str:
    min_temp = cond.get("temperature_min_celsius")
    max_temp = cond.get("temperature_max_celsius")
    if min_temp is not None and max_temp is not None:
        return _format_range(min_temp, max_temp)
    return _to_num_str(cond.get("temperature_celsius"))


def _extract_base_material(phase_name: Any) -> str:
    if not phase_name:
        return ""

    phase = str(phase_name).strip()
    phase_lower = phase.lower()
    if "silica" in phase_lower:
        return "Silica"
    if "ps/dvb" in phase_lower:
        return "PS/DVB"
    if re.search(r"\bdvb\b", phase_lower):
        return "DVB"
    if "beh" in phase_lower:
        return "BEH"
    if re.search(r"c\d+", phase_lower):
        return "Silica"
    return ""


def _polycrit_phase(column_mode: Any, stationary_phase: Any = "") -> str:
    if not column_mode:
        chem = str(stationary_phase).lower()
        if re.search(r"c\d+", chem) or "octadecyl" in chem or "rp" in chem:
            return "Reverse"
        return ""

    mode = str(column_mode).strip().lower()
    if "hilic" in mode:
        return "HILIC"
    if mode.startswith("normal"):
        return "Normal"
    if mode.startswith("reversed"):
        return "Reverse"
    if mode == "reverse":
        return "Reverse"
    return str(column_mode).strip()


_POLYCRIT_FIELDNAMES_BASE = [
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
    "Particle diameter (\u03bcm)",
    "Pore size (\u00c5)",
    "Temperature (Celsius)",
    "Flow Rate (mL/min)",
    "Injected Polymer Concentration (g/L)",
    "Detector",
]

POLYCRIT_FIELDNAMES = _POLYCRIT_FIELDNAMES_BASE
POLYCRIT_REVIEW_AUDIT_FIELDNAMES = [
    "Reference (Raw)",
    "Paper DOI",
    "Corresponding Author Name",
    "Corresponding Email Address",
    "Physical Address",
    "Publication Year (Raw)",
    "Critical Condition Basis",
    "Critical Condition Confidence",
    "Model Confidences",
    "Qwen Confidence",
    "Mistral Confidence",
    "Analyte Polymer (Raw)",
    "Analyte Polymer (Standardized)",
    "Critical Component (Raw)",
    "Critical Component (Standardized)",
    "Architecture (Raw)",
    "Architecture (Standardized)",
    "Polymer (Raw)",
    "Alternate Polymer Names (Parsed from Raw)",
    "Polymer Parsing",
    "End Groups (Raw)",
    "End Groups (Parsed)",
    "Solvents (Raw)",
    "Solvents (Standardized)",
    "Solvent Qualifiers (Standardized)",
    "Solvent Count (Standardized)",
    "Solvent Ratio (Raw)",
    "Solvent Ratio Units (Raw)",
    "Solvent Ratio Components (Standardized)",
    "Solvent Ratio Units (Standardized)",
    "Solvent Ratio Minimum (Standardized)",
    "Solvent Ratio Maximum (Standardized)",
    "Stationary Phase (Raw)",
    "Manufacturer (Raw)",
    "Base Material Source (Raw)",
    "Base Material Modification (Raw)",
    "Column Name (Raw)",
    "Column Mode (Raw)",
    "Column Mode (Standardized)",
    "Column Dimensions (Raw)",
    "Pore Size (Raw)",
    "Pore Size (Standardized)",
    "Temperature (Raw)",
    "Temperature (Standardized)",
    "Flow Rate (Raw)",
    "Flow Rate (Standardized)",
    "Injected Polymer Concentration (Raw)",
    "Detector (Raw)",
    "Aqueous Parameters (Raw)",
    "Aqueous Parameters (Standardized)",
    "Evidence Text",
    "Notes",
]
POLYCRIT_FIELDNAMES_REVIEW = [
    *POLYCRIT_FIELDNAMES,
    *POLYCRIT_REVIEW_AUDIT_FIELDNAMES,
]


LEGACY_FIELDNAMES = [
    "Paper",
    "DOI",
    "Publication Year",
    "Corresponding Author",
    "Email",
    "Physical Address",
    "Analyte Polymer",
    "Critical Component",
    "Architecture (Raw)",
    "Architecture",
    "Critical Condition Basis",
    "Column Name",
    "Stationary Phase Chemistry",
    "Column Mode",
    "Pore Size (Raw)",
    "Pore Size (Angstrom)",
    "Pore Size Min (Angstrom)",
    "Pore Size Max (Angstrom)",
    "Column Dimensions",
    "Mobile Phase Solvents",
    "Mobile Phase Ratio (Raw)",
    "Mobile Phase Ratio Components",
    "Mobile Phase Ratio Min",
    "Mobile Phase Ratio Max",
    "Mobile Phase Ratio Units (Raw)",
    "Mobile Phase Ratio Units",
    "Aqueous pH",
    "Aqueous pH Modifier",
    "Aqueous Salt Added",
    "Aqueous Salt Type",
    "Aqueous Salt Concentration",
    "Temperature (\u00b0C) (Raw)",
    "Temperature (\u00b0C)",
    "Temperature Min (\u00b0C)",
    "Temperature Max (\u00b0C)",
    "Flow Rate (Raw)",
    "Flow Rate (mL/min)",
    "Flow Rate Min (mL/min)",
    "Flow Rate Max (mL/min)",
    "Detector",
    "Consensus Confidence",
    "Qwen Confidence",
    "Mistral Confidence",
    "Evidence Text",
    "Notes",
]


def export_folder_to_csv(
    folder_path: str,
    output_csv: str,
    mode: str = "polycrit",
    *,
    polymer_reference_xlsx: Optional[str] = None,
    include_review_columns: bool = False,
) -> None:
    """Export all JSON files in a folder recursively to a single summary CSV."""
    folder = Path(folder_path)
    output = Path(output_csv)
    fieldnames = POLYCRIT_FIELDNAMES if mode == "polycrit" else LEGACY_FIELDNAMES
    if mode == "polycrit" and include_review_columns:
        # Keep a stable default schema unless review mode is requested.
        fieldnames = POLYCRIT_FIELDNAMES_REVIEW
    if polymer_reference_xlsx:
        # Retained for compatibility. The exporter now uses internal alias rules.
        # This avoids dependency on any PolyCrit spreadsheet at runtime.
        pass

    alias_to_canonical, canonical_aliases, canonical_display = _load_polymer_reference_dictionary()

    rows = []

    for json_file in sorted(folder.rglob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            paper_name = data.get("metadata", {}).get("source_pdf", json_file.stem)
            reference = _extract_reference(
                source_pdf=paper_name,
                fallback_stem=json_file.stem,
            )
            conditions = data.get("extracted_data", {}).get("conditions", [])

            for condition in conditions:
                aq = condition.get("aqueous_parameters") or {}
                solvents = condition.get("mobile_phase_solvents") or []
                solvent_qualifiers = condition.get("mobile_phase_solvent_qualifiers") or []
                conf = condition.get("model_confidences") or {}
                wt_ratio, vol_ratio = _ratio_columns(condition)
                analyte_polymer_raw = (
                    _clean(condition.get("analyte_polymer_raw"))
                    or _clean(condition.get("analyte_polymer"))
                )
                critical_component_raw = (
                    _clean(condition.get("critical_component_raw"))
                    or _clean(condition.get("critical_component"))
                )
                polymer_raw, polymer, alternate = _derive_polymer_columns(
                    analyte_polymer_raw,
                    critical_component_raw,
                    condition.get("architecture"),
                    alias_to_canonical,
                    canonical_aliases,
                    canonical_display,
                )
                polymer_parsing = _parse_polymer_architecture(
                    condition.get("architecture"),
                    condition.get("analyte_polymer"),
                    condition.get("critical_component"),
                    condition.get("architecture_raw"),
                )
                parsed_end_groups = _extract_end_groups(
                    condition.get("analyte_polymer"),
                    condition.get("critical_component"),
                    condition.get("notes"),
                    condition.get("evidence_text"),
                )
                _, raw_alternate = _extract_polymer_aliases(
                    analyte_polymer_raw,
                    critical_component_raw,
                )

                if mode == "polycrit":
                    row = {
                        "Reference": reference,
                        "Author Year": _clean(condition.get("publication_year")),
                        "Polymer": polymer,
                        "Alternate Polymer Names": alternate,
                        "End Groups": parsed_end_groups,
                        "Solvents": _format_polycrit_solvents(
                            solvents,
                            solvent_qualifiers,
                        ),
                        "Solvent Ratio (wt%)": wt_ratio,
                        "Solvent Ratio (vol%)": vol_ratio,
                        "Stationary Phase": _clean(condition.get("stationary_phase_chemistry")),
                        "Manufacturer": "",
                        "Base Material": _extract_base_material(condition.get("stationary_phase_chemistry")),
                        "Base Material Modification": "",
                        "Phase": _polycrit_phase(
                            condition.get("column_mode"),
                            condition.get("stationary_phase_chemistry"),
                        ),
                        "Particle diameter (\u03bcm)": _extract_particle_diameter_um(
                            condition.get("column_dimensions")
                        ),
                        "Pore size (\u00c5)": _pore_size_for_export(condition),
                        "Temperature (Celsius)": _parse_temperature(condition),
                        "Flow Rate (mL/min)": _parse_flow(condition),
                        "Injected Polymer Concentration (g/L)": "",
                        "Detector": _clean(condition.get("detector")),
                    }
                    if include_review_columns:
                        raw_solvents = _raw_condition_value(
                            condition,
                            "mobile_phase_solvents",
                        )
                        raw_aqueous = condition.get(
                            "aqueous_parameters_raw",
                            condition.get("aqueous_parameters"),
                        )
                        row.update(
                            {
                                "Reference (Raw)": _preserve_raw_value(paper_name),
                                "Paper DOI": _preserve_raw_value(
                                    condition.get("paper_doi")
                                ),
                                "Corresponding Author Name": _preserve_raw_value(
                                    condition.get("corresponding_author_name")
                                ),
                                "Corresponding Email Address": _preserve_raw_value(
                                    condition.get("corresponding_email_address")
                                ),
                                "Physical Address": _preserve_raw_value(
                                    condition.get("physical_address")
                                ),
                                "Publication Year (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "publication_year")
                                ),
                                "Critical Condition Basis": _preserve_raw_value(
                                    condition.get("critical_condition_basis")
                                ),
                                "Critical Condition Confidence": _preserve_raw_value(
                                    condition.get("critical_condition_confidence")
                                ),
                                "Model Confidences": _preserve_raw_value(conf),
                                "Qwen Confidence": _preserve_raw_value(conf.get("qwen")),
                                "Mistral Confidence": _preserve_raw_value(
                                    conf.get("mistral")
                                ),
                                "Analyte Polymer (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "analyte_polymer")
                                ),
                                "Analyte Polymer (Standardized)": _clean(
                                    condition.get("analyte_polymer")
                                ),
                                "Critical Component (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "critical_component")
                                ),
                                "Critical Component (Standardized)": _clean(
                                    condition.get("critical_component")
                                ),
                                "Architecture (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "architecture")
                                ),
                                "Architecture (Standardized)": _clean(
                                    condition.get("architecture")
                                ),
                                "Polymer (Raw)": polymer_raw,
                                "Alternate Polymer Names (Parsed from Raw)": raw_alternate,
                                "Polymer Parsing": polymer_parsing,
                                "End Groups (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "end_groups")
                                ),
                                "End Groups (Parsed)": parsed_end_groups,
                                "Solvents (Raw)": _preserve_raw_value(raw_solvents),
                                "Solvents (Standardized)": _preserve_raw_value(solvents),
                                "Solvent Qualifiers (Standardized)": _preserve_raw_value(
                                    solvent_qualifiers
                                ),
                                "Solvent Count (Standardized)": str(
                                    len(solvents)
                                    if isinstance(solvents, (list, tuple))
                                    else int(bool(_clean(solvents)))
                                ),
                                "Solvent Ratio (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "mobile_phase_ratio")
                                ),
                                "Solvent Ratio Units (Raw)": _preserve_raw_value(
                                    _raw_condition_value(
                                        condition,
                                        "mobile_phase_ratio_units",
                                    )
                                ),
                                "Solvent Ratio Components (Standardized)": (
                                    _preserve_raw_value(
                                        condition.get("mobile_phase_ratio_components")
                                    )
                                ),
                                "Solvent Ratio Units (Standardized)": _clean(
                                    condition.get("mobile_phase_ratio_units")
                                ),
                                "Solvent Ratio Minimum (Standardized)": _clean(
                                    condition.get("mobile_phase_ratio_min")
                                ),
                                "Solvent Ratio Maximum (Standardized)": _clean(
                                    condition.get("mobile_phase_ratio_max")
                                ),
                                "Stationary Phase (Raw)": _preserve_raw_value(
                                    _raw_condition_value(
                                        condition,
                                        "stationary_phase_chemistry",
                                    )
                                ),
                                "Manufacturer (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "manufacturer")
                                ),
                                "Base Material Source (Raw)": _preserve_raw_value(
                                    _raw_condition_value(
                                        condition,
                                        "stationary_phase_chemistry",
                                    )
                                ),
                                "Base Material Modification (Raw)": _preserve_raw_value(
                                    _raw_condition_value(
                                        condition,
                                        "base_material_modification",
                                    )
                                ),
                                "Column Name (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "column_name")
                                ),
                                "Column Mode (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "column_mode")
                                ),
                                "Column Mode (Standardized)": _clean(
                                    condition.get("column_mode")
                                ),
                                "Column Dimensions (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "column_dimensions")
                                ),
                                "Pore Size (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "pore_size")
                                ),
                                "Pore Size (Standardized)": _pore_size_for_export(
                                    condition
                                ),
                                "Temperature (Raw)": _preserve_raw_value(
                                    _raw_condition_value(
                                        condition,
                                        "temperature_celsius",
                                    )
                                ),
                                "Temperature (Standardized)": _parse_temperature(
                                    condition
                                ),
                                "Flow Rate (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "flow_rate")
                                ),
                                "Flow Rate (Standardized)": _parse_flow(condition),
                                "Injected Polymer Concentration (Raw)": _preserve_raw_value(
                                    _raw_condition_value(
                                        condition,
                                        "injected_polymer_concentration_g_l",
                                    )
                                ),
                                "Detector (Raw)": _preserve_raw_value(
                                    _raw_condition_value(condition, "detector")
                                ),
                                "Aqueous Parameters (Raw)": _preserve_raw_value(
                                    raw_aqueous
                                ),
                                "Aqueous Parameters (Standardized)": _preserve_raw_value(
                                    condition.get("aqueous_parameters")
                                ),
                                "Evidence Text": _preserve_raw_value(
                                    condition.get("evidence_text")
                                ),
                                "Notes": _preserve_raw_value(condition.get("notes")),
                            }
                        )
                else:
                    row = {
                        "Paper": paper_name,
                        "DOI": _clean(condition.get("paper_doi")),
                        "Publication Year": _clean(condition.get("publication_year")),
                        "Corresponding Author": _clean(condition.get("corresponding_author_name")),
                        "Email": _clean(condition.get("corresponding_email_address")),
                        "Physical Address": _clean(condition.get("physical_address")),
                        "Analyte Polymer": _clean(condition.get("analyte_polymer")),
                        "Critical Component": _clean(condition.get("critical_component")),
                        "Architecture (Raw)": _clean(condition.get("architecture_raw")),
                        "Architecture": _clean(condition.get("architecture")),
                        "Critical Condition Basis": _clean(condition.get("critical_condition_basis")),
                        "Column Name": _clean(condition.get("column_name")),
                        "Stationary Phase Chemistry": _clean(condition.get("stationary_phase_chemistry")),
                        "Column Mode": _clean(condition.get("column_mode")),
                        "Pore Size (Raw)": _clean(condition.get("pore_size_raw")),
                        "Pore Size (Angstrom)": _clean(condition.get("pore_size_angstrom")),
                        "Pore Size Min (Angstrom)": _clean(condition.get("pore_size_min_angstrom")),
                        "Pore Size Max (Angstrom)": _clean(condition.get("pore_size_max_angstrom")),
                        "Column Dimensions": _clean(condition.get("column_dimensions")),
                        "Mobile Phase Solvents": _format_polycrit_solvents(
                            solvents,
                            solvent_qualifiers,
                        ),
                        "Mobile Phase Ratio (Raw)": _clean(condition.get("mobile_phase_ratio_raw")),
                        "Mobile Phase Ratio Components": _clean(condition.get("mobile_phase_ratio_components")),
                        "Mobile Phase Ratio Min": _clean(condition.get("mobile_phase_ratio_min")),
                        "Mobile Phase Ratio Max": _clean(condition.get("mobile_phase_ratio_max")),
                        "Mobile Phase Ratio Units (Raw)": _clean(condition.get("mobile_phase_ratio_units_raw")),
                        "Mobile Phase Ratio Units": _clean(condition.get("mobile_phase_ratio_units")),
                        "Aqueous pH": _clean(aq.get("pH")),
                        "Aqueous pH Modifier": _clean(aq.get("pH_modifier")),
                        "Aqueous Salt Added": _clean(aq.get("salt_added") if aq.get("salt_added") is not None else ""),
                        "Aqueous Salt Type": _clean(aq.get("salt_type")),
                        "Aqueous Salt Concentration": _clean(aq.get("salt_concentration")),
                        "Temperature (\u00b0C) (Raw)": _clean(
                            _raw_condition_value(condition, "temperature_celsius")
                        ),
                        "Temperature (\u00b0C)": _parse_temperature(condition),
                        "Temperature Min (\u00b0C)": _clean(condition.get("temperature_min_celsius")),
                        "Temperature Max (\u00b0C)": _clean(condition.get("temperature_max_celsius")),
                        "Flow Rate (Raw)": _clean(condition.get("flow_rate_raw")),
                        "Flow Rate (mL/min)": _parse_flow(condition),
                        "Flow Rate Min (mL/min)": _clean(condition.get("flow_rate_min_ml_per_min")),
                        "Flow Rate Max (mL/min)": _clean(condition.get("flow_rate_max_ml_per_min")),
                        "Detector": _clean(condition.get("detector")),
                        "Consensus Confidence": _clean(condition.get("critical_condition_confidence")),
                        "Qwen Confidence": _clean(conf.get("qwen")),
                        "Mistral Confidence": _clean(conf.get("mistral")),
                        "Evidence Text": _clean(condition.get("evidence_text")),
                        "Notes": _clean(condition.get("notes")),
                    }

                rows.append(row)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} conditions from {folder} to {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a folder of standardized JSON results to a single summary CSV."
    )
    parser.add_argument("folder", help="Folder containing JSON files (e.g. results/standardized)")
    parser.add_argument("output", help="Output CSV file path (e.g. results/standardized_summary.csv)")
    parser.add_argument(
        "--mode",
        choices=["polycrit", "legacy"],
        default="polycrit",
        help="Output schema: polycrit (default) or legacy",
    )
    parser.add_argument(
        "--polymer-reference-xlsx",
        default=None,
        help="Deprecated. Kept for compatibility; alias mapping is now internal.",
    )
    parser.add_argument(
        "--include-review-columns",
        action="store_true",
        default=False,
        help=(
            "Append raw and standardized audit columns for debugging; "
            "source fields absent from extraction remain blank."
        ),
    )
    args = parser.parse_args()

    export_folder_to_csv(
        args.folder,
        args.output,
        mode=args.mode,
        polymer_reference_xlsx=args.polymer_reference_xlsx,
        include_review_columns=args.include_review_columns,
    )
