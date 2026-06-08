"""Export extraction JSON outputs to a flat summary CSV."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

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
                    "DOI": c.get("paper_doi", ""),
                    "Publication Year": c.get("publication_year", ""),
                    "Corresponding Author": c.get("corresponding_author_name", ""),
                    "Email": c.get("corresponding_email_address", ""),
                    "Physical Address": c.get("physical_address", ""),
                    "Analyte Polymer": c.get("analyte_polymer", ""),
                    "Critical Component": c.get("critical_component", ""),
                    "Architecture": c.get("architecture", ""),
                    "Critical Condition Basis": c.get("critical_condition_basis", ""),
                    "Column Name": c.get("column_name", ""),
                    "Stationary Phase Chemistry": c.get("stationary_phase_chemistry", ""),
                    "Pore Size": c.get("pore_size", ""),
                    "Column Dimensions": c.get("column_dimensions", ""),
                    "Mobile Phase Solvents": ", ".join(solvents) if isinstance(solvents, list) else str(solvents),
                    "Mobile Phase Ratio": c.get("mobile_phase_ratio", ""),
                    "Mobile Phase Ratio Units": c.get("mobile_phase_ratio_units", ""),
                    "Aqueous pH": aq.get("pH", ""),
                    "Aqueous Salt Added": aq.get("salt_added", ""),
                    "Aqueous Salt Type": aq.get("salt_type", ""),
                    "Aqueous Salt Concentration": aq.get("salt_concentration", ""),
                    "Temperature (°C)": c.get("temperature_celsius", ""),
                    "Flow Rate": c.get("flow_rate", ""),
                    "Detector": c.get("detector", ""),
                    "Consensus Confidence": c.get("critical_condition_confidence", ""),
                    "Qwen Confidence": conf.get("qwen", ""),
                    "Mistral Confidence": conf.get("mistral", ""),
                    "Evidence Text": c.get("evidence_text", ""),
                    "Notes": c.get("notes", "")
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
