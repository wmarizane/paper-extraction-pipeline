"""
Post-processing standardizer for consensus JSON output.

Reads:   results/consensus/**/*_consensus.json
Writes:  results/standardized/**/*_standardized.json
         (same subdirectory tree as consensus/)

Source JSONs are never modified.
Each standardized condition preserves the original raw value alongside
the cleaned value (with a _raw suffix) for full auditability.
"""

import json
import logging
import copy
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

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
    "eoâ€“po diblock": "diblock",
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
    # Full Polymer ↔ alternate-name coverage from PolyCrit seed data (used as context only).
    "Poly(methyl methacrylate)": ["PMMA", "hPMMA", "dPMMA"],
    "Polystyrene": ["PS", "hPS", "dPS"],
    "Polyisoprene": ["PI"],
    "Poly(ethyl methacrylate)": ["PEMA"],
    "Poly(vinyl chloride)": ["PVC"],
    "Polybutadiene": ["PB"],
    "Poly(propylene glycol)": ["PPG"],
    "Poly(decyl methacrylate)": ["PDMA"],
    "Poly(1,3,6-trioxocane)": ["polyTOC"],
    "Poly[(adipic acid)-co-(1,2-ethanediol)]": ["AA-1", "2ED"],
    "Poly[(adipic acid)-co-(1,2-propanediol)]": ["AA-1", "2PD"],
    "Poly[(adipic acid)-co-(1,3-butanediol)]": ["AA-1", "3BD"],
    "Poly[(adipic acid)-co-(1,3-propanediol)]": ["AA-1", "3PD"],
    "Poly[(adipic Acid)-co-(1,4-butanediol)]": ["AA-1", "4BD"],
    "Poly[(adipic acid)-co-(diethylene glycol)]": ["AA-DEG"],
    "Poly[(adipic acid)-co-(dipropylene glycol)]": ["AA-DPG"],
    "Poly[(adipic acid)-co-(neopentyl glycol)]": ["AA-NPG"],
    "Poly[(phthalic acid)-co-(diethylene glycol)]": ["PA-DEG"],
    "Poly[(phthalic acid)-co-(ethylene glycol)]": ["PA-EG"],
    "Poly[(phthalic acid)-co-(triethylene glycol)]": ["PA-TEG"],
    "Polycarbonate": ["PC"],
    "Poly[di(ethylene glycol) adipate]": ["PDEGA"],
    "Polydimethylsiloxane": ["PDMS"],
    "Poly(ethylene glycol)": ["PEG", "mPEG", "PEG-MME", "PEG-DME", "MeO-PEG-DME", "MeO-PEG"],
    "Poly(ethylene oxide)": ["PEO"],
    "Poly(t-butyl methacrylate)": ["PtBMA"],
    "Poly(n-butyl methacrylate)": ["PBMA", "PnBMA"],
    "Poly(2-vinylpyridine)": ["P2VP"],
    "Poly(propylene oxide) Adipate": ["PPOA"],
    "Polycaprolactam": ["PA6", "Nylon-6"],
    "Polybutylene Terephthalate": ["PBT", "PBTF"],
    "Polysulfone": ["PSU"],
    "Poly(phenolphthalein terephthalate)": ["PPha-tere"],
    "Poly(n-butyl acrylate)": ["PBA", "PnBA"],
    "Poly[(adipic acid)-co-(1,6-hexanediol)]": ["AA-HD", "AA-1", "6HD"],
    "Chlorinated Polyethylene": ["CPE", "PE-C"],
    "Bisphenol A": ["BPA", "DGEBA", "Epoxy Resin"],
    "Calixarene": ["Calixarene"],
    "Polyepoxides": ["Epoxy resin"],
    "Polycaprolactone": ["PCL", "PCL-ME"],
    "Polyepichlorohydrin": ["PECH", "Epoxy Resin"],
    "Poly(1,4-butylene adipate)": ["PBAG"],
    "Polyphenylsulfone": ["PPSU", "PFS"],
    "Aliphatic Polycarbonate": ["APC", "PC"],
    "Poly(L-lactide)": ["PLLA", "PLA"],
    "Poly(propylene adipate)": ["PPA"],
    "Polytetrahydrofuran": ["PTHF"],
    "Polyisobutylene": ["PIB"],
    "Poly(2-ethyl-2-oxazoline)": ["PEtOx"],
    "Poly(butylene oxide)": ["PBO"],
    "Poly(hexylene oxide)": ["PHO"],
    "Poly(isobornyl acrylate)": ["PiBoA"],
    "Poly(methyl acrylate)": ["PMA"],
    "Polyvinylpyrrolidone": ["PVP"],
    "Polyvinyl acetate": ["PVAc", "PVA"],
    "Poly(diphenolic acid)": ["PDPA"],
    "Adipic acid": ["AA", "AS"],
    "Poly(ambrettolide)": ["PAmb", "cPAmb"],
    "Polystyrene sulfonate": ["SPS", "PSS"],
    "Poly(acrylic acid)": ["PAA", "PAAS", "ACR"],
    "Deuterated Polystyrene": ["dPS"],
    "Poly(propylene phthalate)": ["PPOPA"],
    "Poly(3,6-dioxa-1,8-octanedithiol)": ["polyDODT"],
    "Poly(Ethylene-co-Propylene)": ["EP", "EP Copolomer"],
    "Polyethylene": ["PE"],
    "Polypropylene": ["PP"],
    "Polyoxyethylene sorbitan monolaurate": ["Tween 20", "Polysorbate 20"],
    "Poly(ethylene ether carbonate)": ["PEEC"],
    "Poly(propylene ether carbonate)": ["PPEC"],
    "Polyoxyethylene sorbitan monopalmitate": ["Tween 40", "Polysorbate 40"],
}

_POLYMER_ALIAS_TO_CANONICAL: Dict[str, str] = {}
_POLYMER_CANONICAL_DISPLAY: Dict[str, str] = {}

def _normalize_polymer_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


for _canonical_polymer, _aliases in _POLYMER_CANONICAL_TO_ALIASES.items():
    _canonical_key = _normalize_polymer_key(_canonical_polymer)
    if not _canonical_key:
        continue
    canonical_display = str(_canonical_polymer).strip()
    _POLYMER_CANONICAL_DISPLAY[_canonical_key] = canonical_display
    _POLYMER_ALIAS_TO_CANONICAL[_canonical_key] = _canonical_key
    for _alias in _aliases:
        alias_key = _normalize_polymer_key(_alias)
        if alias_key:
            _POLYMER_ALIAS_TO_CANONICAL[alias_key] = _canonical_key

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
        cond[f"{field}_raw"] = value

        if isinstance(raw, str):
            normalized = _normalize_polymer_name(raw)
            cond[field] = normalized

def _parse_first_number(text: str) -> Optional[float]:
    match = re.search(r"\d+(?:\.\d+)?", str(text))
    if not match:
        return None
    return float(match.group(0))


def _count_mobile_phase_solvents(cond: Dict[str, Any]) -> int:
    solvents = cond.get("mobile_phase_solvents")
    if not solvents or not isinstance(solvents, list):
        return 0
    return len([s for s in solvents if str(s).strip()])


def _extract_ratio_pair(val_str: str, separator: str) -> Optional[List[float]]:
    if separator == "-":
        parts = re.split(r"\s*-\s*", val_str, maxsplit=1)
    else:
        parts = val_str.split(separator)

    if len(parts) != 2:
        return None

    left = _parse_first_number(parts[0])
    right = _parse_first_number(parts[1])
    if left is None or right is None:
        return None

    return [left, right]


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
    cond["mobile_phase_ratio_units_raw"] = val
    if not val or str(val).strip().lower() in ("null", "none", ""):
        cond["mobile_phase_ratio_units"] = None
        return

    val_str = str(val).strip()
    val_lower = val_str.lower()

    # Strip everything that usually appears after the solvent field extraction
    val_clean = re.sub(r"\(.*?\)", "", val_lower).strip()
    val_clean = re.sub(r"[\s,_]", "", val_clean)
    val_clean = val_clean.replace(",", "").replace("by", "")

    if (
        "wt" in val_clean
        or "weight" in val_clean
        or "w/w" in val_clean
    ):
        cond["mobile_phase_ratio_units"] = "w/w"
    elif "v/v" in val_clean or "vol" in val_clean or "byvolume" in val_clean:
        cond["mobile_phase_ratio_units"] = "v/v"
    elif val_clean in {"%", "%v", "v", "vv"}:
        cond["mobile_phase_ratio_units"] = "v/v"
    else:
        warnings.warn(f"Unrecognized ratio unit: {val}")
        cond["mobile_phase_ratio_units"] = None

def _normalize_ratio(cond: Dict[str, Any]) -> None:
    # Backward compatibility: delegate to the corrected parser.
    _normalize_ratio_v2(cond)
    return

    val = cond.get("mobile_phase_ratio")
    cond["mobile_phase_ratio_raw"] = val
    cond["mobile_phase_ratio_components"] = None
    
    if not val or str(val).strip().lower() in ("null", "none", ""):
        return
        
    val_str = str(val).strip()
    
    try:
        # 1. Single float string
        if re.fullmatch(r"[\d.]+", val_str):
            f1 = float(val_str)
            cond["mobile_phase_ratio_components"] = [f1, round(100.0 - f1, 2)]
            return
            
        # 2. Colon-separated
        if ":" in val_str:
            parts = [float(p) for p in val_str.split(":")]
            cond["mobile_phase_ratio_components"] = parts
            return
            
        # 3. Slash-separated
        if "/" in val_str:
            parts = [float(p) for p in val_str.split("/")]
            cond["mobile_phase_ratio_components"] = parts
            return
            
        # 4. Dash-separated (but NOT a range)
        if "-" in val_str:
            try:
                parts = [float(p) for p in val_str.split("-")]
                if abs(sum(parts) - 100.0) <= 2.0:
                    cond["mobile_phase_ratio_components"] = parts
                    return
                if len(parts) == 2:
                    cond["mobile_phase_ratio_components"] = None
                    cond["mobile_phase_ratio_min"] = parts[0]
                    cond["mobile_phase_ratio_max"] = parts[1]
                    return
            except ValueError:
                pass
                
        # 6. Multi-component text
        if "and" in val_str.lower() or "," in val_str:
            floats = [float(x) for x in re.findall(r"([\d.]+)", val_str)]
            if len(floats) >= 2:
                if sum(floats) > 100.0:
                    # Floats are per-component absolute values already summing >100
                    # (e.g. two separate ratios on the same line) â€” unparseable
                    warnings.warn(f"Ternary components sum > 100, cannot parse: {val_str}")
                    return
                if sum(floats) < 100.0:
                    floats.append(round(100.0 - sum(floats), 2))
                cond["mobile_phase_ratio_components"] = floats
                return
                
        # 5. Embedded solvent + value
        match = re.search(r"([\d.]+)\s*(?:wt|vol|%|v/v|w/w|-)", val_str, re.IGNORECASE)
        if match:
            f1 = float(match.group(1))
            cond["mobile_phase_ratio_components"] = [f1, round(100.0 - f1, 2)]
            return
            
        match = re.match(r"^([\d.]+)", val_str)
        if match:
             f1 = float(match.group(1))
             cond["mobile_phase_ratio_components"] = [f1, round(100.0 - f1, 2)]
             return
             
        warnings.warn(f"Unparseable ratio: {val_str}")
    except Exception as e:
        warnings.warn(f"Error parsing ratio '{val_str}': {e}")

def _normalize_ratio_v2(cond: Dict[str, Any]) -> None:
    val = cond.get("mobile_phase_ratio")
    cond["mobile_phase_ratio_raw"] = val
    cond["mobile_phase_ratio_components"] = None
    cond["mobile_phase_ratio_min"] = None
    cond["mobile_phase_ratio_max"] = None
    solvent_count = _count_mobile_phase_solvents(cond)

    if not val or str(val).strip().lower() in ("null", "none", ""):
        # Pure single-solvent conditions are still explicit compositions.
        # If no ratio is provided, infer 100% for that solvent.
        if solvent_count == 1:
            # Keep raw extractor output untouched (None) but model the implied
            # composition for export and downstream consumers.
            cond["mobile_phase_ratio_components"] = [100.0]
            if cond.get("mobile_phase_ratio_units") in (None, "", "null", "none"):
                cond["mobile_phase_ratio_units"] = "v/v"
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
def _normalize_flow_rate(cond: Dict[str, Any]) -> None:
    val = cond.get("flow_rate")
    cond["flow_rate_raw"] = val
    cond["flow_rate_ml_per_min"] = None
    
    if not val or str(val).strip().lower() in ("null", "none", ""):
        return
        
    val_str = str(val).strip()
    
    try:
        range_match = re.search(r"([\d.]+)\s*(?:to|[-\u2013\u2014\u2212])\s*([\d.]+)", val_str, re.IGNORECASE)
        if range_match:
            cond["flow_rate_min_ml_per_min"] = float(range_match.group(1))
            cond["flow_rate_max_ml_per_min"] = float(range_match.group(2))
            return
            
        match = re.search(r"([\d.]+)", val_str)
        if match:
            cond["flow_rate_ml_per_min"] = float(match.group(1))
    except Exception as e:
        warnings.warn(f"Error parsing flow rate '{val_str}': {e}")

def _normalize_pore_size(cond: Dict[str, Any]) -> None:
    val = cond.get("pore_size")
    cond["pore_size_raw"] = val
    cond["pore_size_angstrom"] = None
    
    if not val or str(val).strip().lower() in ("null", "none", ""):
        return
        
    val_str = str(val).strip()
    
    try:
        if ";" in val_str or ":" in val_str:
            return
            
        range_match = re.search(r"([\d.]+)\s*(?:-|â€“)\s*([\d.]+)\s*(nm|Ã…|A|um|Âµm)?", val_str, re.IGNORECASE)
        if range_match:
            v1, v2 = float(range_match.group(1)), float(range_match.group(2))
            unit = range_match.group(3) or "A"
            unit = unit.lower()
            mult = 1.0
            if "nm" in unit:
                mult = 10.0
            elif "um" in unit or "Âµm" in unit:
                mult = 10000.0
                warnings.warn(f"Unusually large pore size range unit: {val_str}")
            
            cond["pore_size_min_angstrom"] = v1 * mult
            cond["pore_size_max_angstrom"] = v2 * mult
            return
            
        if "and" in val_str.lower() or "," in val_str:
            floats = []
            for part in re.split(r"and|,", val_str, flags=re.IGNORECASE):
                m = re.search(r"([\d.]+)", part)
                if m:
                    floats.append(float(m.group(1)))
            
            mult = 1.0
            if "nm" in val_str.lower():
                mult = 10.0
            elif "um" in val_str.lower() or "Âµm" in val_str.lower():
                mult = 10000.0
                warnings.warn(f"Unusually large pore size list unit: {val_str}")
                
            cond["pore_size_angstrom"] = [f * mult for f in floats]
            return
            
        match = re.search(r"([\d.]+)", val_str)
        if match:
            v = float(match.group(1))
            mult = 1.0
            if "nm" in val_str.lower():
                mult = 10.0
            elif "um" in val_str.lower() or "Âµm" in val_str.lower():
                mult = 10000.0
                warnings.warn(f"Unusually large pore size unit: {val_str}")
                
            cond["pore_size_angstrom"] = v * mult
            
    except Exception as e:
        warnings.warn(f"Error parsing pore size '{val_str}': {e}")

def _normalize_temperature(cond: Dict[str, Any]) -> None:
    val = cond.get("temperature_celsius")
    if not val or str(val).strip().lower() in ("null", "none", ""):
        cond["temperature_celsius"] = None
        return
        
    val_str = str(val).strip()
    
    try:
        range_match = re.search(r"([\d.]+)\s*(?:[-\u2013\u2014\u2212])\s*([\d.]+)", val_str)
        if range_match:
            cond["temperature_celsius"] = None
            cond["temperature_min_celsius"] = float(range_match.group(1))
            cond["temperature_max_celsius"] = float(range_match.group(2))
            return
            
        match = re.search(r"([\d.]+)", val_str)
        if match:
            cond["temperature_celsius"] = float(match.group(1))
    except Exception as e:
        warnings.warn(f"Error parsing temperature '{val_str}': {e}")

def _normalize_column_mode(cond: Dict[str, Any]) -> None:
    val = cond.get("column_mode")
    if not val or str(val).strip().lower() in ("null", "none", ""):
        cond["column_mode"] = None
        return
        
    val_str = str(val).strip().lower()
    
    if val_str in ("reverse phase", "reversed phase"):
        cond["column_mode"] = "Reversed Phase"
    elif val_str == "normal phase":
        cond["column_mode"] = "Normal Phase"
    elif val_str in ("hilic", "hydrophilic interaction", "hydrophilic interaction chromatography"):
        cond["column_mode"] = "HILIC"
    elif val_str in ("sec", "size exclusion", "size exclusion chromatography"):
        cond["column_mode"] = "SEC"

def _normalize_architecture(cond: Dict[str, Any]) -> None:
    val = cond.get("architecture")
    if not val:
        if str(val).lower() == "null":
            cond["architecture"] = None
        return
        
    val_str = str(val).strip().lower()
    
    if val_str in ARCHITECTURE_MAP:
        new_val = ARCHITECTURE_MAP[val_str]
        if new_val != val:
            cond["architecture_raw"] = val
            cond["architecture"] = new_val
        return
        
    for key, mapped in ARCHITECTURE_MAP.items():
        if val_str.startswith(key):
            if mapped != val:
                cond["architecture_raw"] = val
                cond["architecture"] = mapped
            return
            
    if "block" in val_str:
        cond["architecture_raw"] = val
        cond["architecture"] = "block copolymer"
        return
        
    warnings.warn(f"Unrecognized architecture: {val}")

def _normalize_solvents(cond: Dict[str, Any]) -> None:
    solvents = cond.get("mobile_phase_solvents")
    if not solvents or not isinstance(solvents, list):
        return

    new_solvents = []
    modifiers = []

    seen = set()

    for solvent in solvents:
        solvent_str = str(solvent).strip()
        if not solvent_str:
            continue

        solvent_lower = solvent_str.lower()
        if solvent_lower in {"solvent", "solvents", "n/a", "na", "none", "null"}:
            continue

        match = re.match(r"water\s+with\s+(.+)", solvent_str, re.IGNORECASE)
        if match:
            modifier = match.group(1).strip()
            if modifier:
                modifiers.append(modifier)
            normalized_solvent = "water"
        else:
            normalized_solvent = solvent_str
            if normalized_solvent.lower().startswith("methyl ethyl ketone ("):
                normalized_solvent = "methyl ethyl ketone"

        normalized_lower = normalized_solvent.strip().lower()
        if normalized_lower in seen:
            continue
        seen.add(normalized_lower)
        new_solvents.append(normalized_solvent)

    cond["mobile_phase_solvents"] = new_solvents
    if modifiers:
        if "aqueous_parameters" not in cond or cond["aqueous_parameters"] is None:
            cond["aqueous_parameters"] = {}
        cond["aqueous_parameters"]["pH_modifier"] = ", ".join(modifiers)

def _normalize_year(cond: Dict[str, Any]) -> None:
    val = cond.get("publication_year")
    if not val or str(val).strip().lower() in ("null", "none", ""):
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
        
        output = {
            "metadata": {
                "source_pdf": input_path.stem.replace("_consensus", ""),
                "model": "deepseek-r1-32b-consensus",
                "inputs": ["qwen3.5-27b", "mistral-small-24b"],
                "standardized_by": "pipeline/standardizer.py",
                "standardization_date": datetime.utcnow().isoformat() + "Z"
            },
            "summary": {"total_conditions": len(standardized_conditions)},
            "extracted_data": {
                "conditions": standardized_conditions
            }
        }
        
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
        output_path = output_dir / rel_path.parent / f"{input_path.stem.replace('_consensus', '')}_standardized.json"
        
        count = standardize_file(input_path, output_path)
        print(f"Processing {rel_path} ... {count} conditions")
        
        total_conds += count
        total_files += 1
        
    print("â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print(f"Standardized {total_conds} conditions across {total_files} files.")

if __name__ == "__main__":
    standardize_all(
        consensus_dir=Path("results/consensus"),
        output_dir=Path("results/standardized")
    )
