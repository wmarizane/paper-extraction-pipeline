"""Export extraction JSON outputs to a flat summary CSV."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

def _clean(val):
    """Convert None and string 'null' to empty string for CSV output."""
    if val is None:
        return ""
    if isinstance(val, str) and val.lower() == "null":
        return ""
    return val

FIELDNAMES = [
    "Paper",
    "DOI",
    "Publication Year",
    "Corresponding Author",
    "Email",
    "Physical Address",
    "Analyte Polymer",
    "Critical Component",
    "Architecture",
    "Critical Condition Basis",
    "Column Name",
    "Stationary Phase Chemistry",
    "Column Mode",
    "Pore Size",
    "Column Dimensions",
    "Mobile Phase Solvents",
    "Mobile Phase Ratio",
    "Mobile Phase Ratio Units",
    "Aqueous pH",
    "Aqueous Salt Added",
    "Aqueous Salt Type",
    "Aqueous Salt Concentration",
    "Temperature (°C)",
    "Flow Rate",
    "Detector",
    "Consensus Confidence",
    "Qwen Confidence",
    "Mistral Confidence",
    "Evidence Text",
    "Notes"
]

def export_folder_to_csv(folder_path: str, output_csv: str) -> None:
    """Export all JSON files in a folder to a single summary CSV."""
    folder = Path(folder_path)
    output = Path(output_csv)
    
    rows = []
    
    for json_file in folder.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            paper_name = data.get("metadata", {}).get("source_pdf", json_file.stem)
            # Handle both consensus outputs and raw extractor outputs
            if "final_consensus" in data:
                conditions = data["final_consensus"].get("extracted_conditions", [])
            else:
                conditions = data.get("extracted_data", {}).get("conditions", [])
                
            for c in conditions:
                aq = c.get("aqueous_parameters") or {}
                solvents = c.get("mobile_phase_solvents") or []
                conf = c.get("model_confidences") or {}
                
                rows.append({
                    "Paper": paper_name,
                    "DOI": _clean(c.get("paper_doi")),
                    "Publication Year": _clean(c.get("publication_year")),
                    "Corresponding Author": _clean(c.get("corresponding_author_name")),
                    "Email": _clean(c.get("corresponding_email_address")),
                    "Physical Address": _clean(c.get("physical_address")),
                    "Analyte Polymer": _clean(c.get("analyte_polymer")),
                    "Critical Component": _clean(c.get("critical_component")),
                    "Architecture": _clean(c.get("architecture")),
                    "Critical Condition Basis": _clean(c.get("critical_condition_basis")),
                    "Column Name": _clean(c.get("column_name")),
                    "Stationary Phase Chemistry": _clean(c.get("stationary_phase_chemistry")),
                    "Column Mode": _clean(c.get("column_mode")),
                    "Pore Size": _clean(c.get("pore_size")),
                    "Column Dimensions": _clean(c.get("column_dimensions")),
                    "Mobile Phase Solvents": _clean(", ".join(solvents) if isinstance(solvents, list) else (str(solvents) if solvents else "")),
                    "Mobile Phase Ratio": _clean(c.get("mobile_phase_ratio")),
                    "Mobile Phase Ratio Units": _clean(c.get("mobile_phase_ratio_units")),
                    "Aqueous pH": _clean(aq.get("pH")),
                    "Aqueous Salt Added": _clean(aq.get("salt_added") if aq.get("salt_added") is not None else ""),
                    "Aqueous Salt Type": _clean(aq.get("salt_type")),
                    "Aqueous Salt Concentration": _clean(aq.get("salt_concentration")),
                    "Temperature (°C)": _clean(c.get("temperature_celsius")),
                    "Flow Rate": _clean(c.get("flow_rate")),
                    "Detector": _clean(c.get("detector")),
                    "Consensus Confidence": _clean(c.get("critical_condition_confidence")),
                    "Qwen Confidence": _clean(conf.get("qwen")),
                    "Mistral Confidence": _clean(conf.get("mistral")),
                    "Evidence Text": _clean(c.get("evidence_text")),
                    "Notes": _clean(c.get("notes"))
                })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Exported {len(rows)} conditions from {folder} to {output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export a folder of JSON results to a single summary CSV.")
    parser.add_argument("folder", help="Folder containing JSON files (e.g. results/consensus_fuzzy_bidir)")
    parser.add_argument("output", help="Output CSV file path (e.g. results/consensus_fuzzy_bidir_summary.csv)")
    args = parser.parse_args()
    
    export_folder_to_csv(args.folder, args.output)
