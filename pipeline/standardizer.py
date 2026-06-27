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
    "eo–po diblock": "diblock",
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

def _normalize_ratio_units(cond: Dict[str, Any]) -> None:
    val = cond.get("mobile_phase_ratio_units")
    cond["mobile_phase_ratio_units_raw"] = val
    if not val or str(val).strip().lower() in ("null", "none", ""):
        cond["mobile_phase_ratio_units"] = None
        return

    val_str = str(val).strip()
    val_lower = val_str.lower()
    
    # Strip anything in parentheses
    val_lower = re.sub(r'\(.*?\)', '', val_lower).strip()
    
    # Strip solvent suffix: if we split on space, and the second part has no digits or '%'
    parts = val_lower.split(maxsplit=1)
    if len(parts) > 1 and not re.search(r'[\d%]', parts[1]):
        val_lower = parts[0]
        
    val_clean = val_lower.replace(" ", "").replace(",", "")

    vv_exact = {
        "v/v", "vol%", "vol.%", "vol-%", "%v/v", "%", "v%", "v:v", "v/v%", 
        "v/v/v", "%byvolume"
    }
    ww_exact = {
        "wt%", "wt-%", "wt.%", "w/w", "%w/w", "wt.-%", "vol-%", "vol.-%" 
    }
    
    if val_clean in vv_exact or val_lower == "% by volume":
        cond["mobile_phase_ratio_units"] = "v/v"
    elif val_clean in ww_exact:
        cond["mobile_phase_ratio_units"] = "w/w"
    else:
        warnings.warn(f"Unrecognized ratio unit: {val}")
        cond["mobile_phase_ratio_units"] = None

def _normalize_ratio(cond: Dict[str, Any]) -> None:
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
                    # (e.g. two separate ratios on the same line) — unparseable
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

def _normalize_flow_rate(cond: Dict[str, Any]) -> None:
    val = cond.get("flow_rate")
    cond["flow_rate_raw"] = val
    cond["flow_rate_ml_per_min"] = None
    
    if not val or str(val).strip().lower() in ("null", "none", ""):
        return
        
    val_str = str(val).strip()
    
    try:
        range_match = re.search(r"([\d.]+)\s*(?:to|-|–)\s*([\d.]+)", val_str, re.IGNORECASE)
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
            
        range_match = re.search(r"([\d.]+)\s*(?:-|–)\s*([\d.]+)\s*(nm|Å|A|um|µm)?", val_str, re.IGNORECASE)
        if range_match:
            v1, v2 = float(range_match.group(1)), float(range_match.group(2))
            unit = range_match.group(3) or "A"
            unit = unit.lower()
            mult = 1.0
            if "nm" in unit:
                mult = 10.0
            elif "um" in unit or "µm" in unit:
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
            elif "um" in val_str.lower() or "µm" in val_str.lower():
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
            elif "um" in val_str.lower() or "µm" in val_str.lower():
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
        range_match = re.search(r"([\d.]+)\s*(?:-|–)\s*([\d.]+)", val_str)
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
    modifier = None
    
    for solvent in solvents:
        solvent_str = str(solvent).strip()
        match = re.search(r"water with (.+)", solvent_str, re.IGNORECASE)
        if match:
            new_solvents.append("water")
            modifier = match.group(1)
        else:
            new_solvents.append(solvent_str)
            
    cond["mobile_phase_solvents"] = new_solvents
    if modifier:
        if "aqueous_parameters" not in cond or cond["aqueous_parameters"] is None:
            cond["aqueous_parameters"] = {}
        cond["aqueous_parameters"]["pH_modifier"] = modifier

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
    out = dict(cond)
    _normalize_ratio_units(out)
    _normalize_ratio(out)
    _normalize_flow_rate(out)
    _normalize_pore_size(out)
    _normalize_temperature(out)
    _normalize_column_mode(out)
    _normalize_architecture(out)
    _normalize_solvents(out)
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
        
    print("─────────────────────────────────────────────────────")
    print(f"Standardized {total_conds} conditions across {total_files} files.")

if __name__ == "__main__":
    standardize_all(
        consensus_dir=Path("results/consensus"),
        output_dir=Path("results/standardized")
    )
