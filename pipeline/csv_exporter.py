"""Export extraction JSON outputs to chemistry-focused table CSV files."""

import csv
from pathlib import Path
from typing import Any, Dict, List


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_extraction_to_csv(data: Dict[str, Any], output_dir: Path, base_name: str) -> Dict[str, str]:
    """Export extraction output into three required chemistry tables."""
    extracted = data.get("extracted_data", {})

    master_rows = []
    for item in extracted.get("master_table", []):
        master_rows.append(
            {
                "Paper": item.get("paper", ""),
                "System ID*": item.get("system_id", ""),
                "Polymer system*": item.get("polymer_system", ""),
                "Target at CC*": item.get("target_at_cc", ""),
                "Architecture context*": item.get("architecture_context", ""),
                "Column*": item.get("column", ""),
                "Stationary phase*": item.get("stationary_phase", ""),
                "Mobile phase*": item.get("mobile_phase", ""),
                "Composition*": item.get("composition", ""),
                "Units*": item.get("units", ""),
                "Temp (°C)*": item.get("temp_c", ""),
                "Additives": item.get("additives", ""),
                "Notes*": item.get("notes", ""),
                "Source section": item.get("source_section", ""),
                "Source text": item.get("source_text", ""),
            }
        )

    mechanism_rows = []
    for item in extracted.get("separation_mechanism", []):
        mechanism_rows.append(
            {
                "Paper": item.get("paper", ""),
                "System": item.get("system", ""),
                "Type of criticality": item.get("type_of_criticality", ""),
                "Driving variable": item.get("driving_variable", ""),
                "Non-critical species behavior": item.get("non_critical_species_behavior", ""),
                "Source section": item.get("source_section", ""),
                "Source text": item.get("source_text", ""),
            }
        )

    metadata_rows = []
    for item in extracted.get("column_system_metadata", []):
        metadata_rows.append(
            {
                "Paper": item.get("paper", ""),
                "Column*": item.get("column", ""),
                "Brand/type*": item.get("brand_type", ""),
                "Stationary phase chemistry*": item.get("stationary_phase_chemistry", ""),
                "Pore size (Å)*": item.get("pore_size_a", ""),
                "Dimensions*": item.get("dimensions", ""),
                "Notes*": item.get("notes", ""),
                "Source section": item.get("source_section", ""),
                "Source text": item.get("source_text", ""),
            }
        )

    master_path = output_dir / f"{base_name}_master_table.csv"
    mechanism_path = output_dir / f"{base_name}_separation_mechanism.csv"
    metadata_path = output_dir / f"{base_name}_column_system_metadata.csv"

    _write_csv(
        master_path,
        [
            "Paper",
            "System ID*",
            "Polymer system*",
            "Target at CC*",
            "Architecture context*",
            "Column*",
            "Stationary phase*",
            "Mobile phase*",
            "Composition*",
            "Units*",
            "Temp (°C)*",
            "Additives",
            "Notes*",
            "Source section",
            "Source text",
        ],
        master_rows,
    )

    _write_csv(
        mechanism_path,
        [
            "Paper",
            "System",
            "Type of criticality",
            "Driving variable",
            "Non-critical species behavior",
            "Source section",
            "Source text",
        ],
        mechanism_rows,
    )

    _write_csv(
        metadata_path,
        [
            "Paper",
            "Column*",
            "Brand/type*",
            "Stationary phase chemistry*",
            "Pore size (Å)*",
            "Dimensions*",
            "Notes*",
            "Source section",
            "Source text",
        ],
        metadata_rows,
    )

    return {
        "master_table_csv": str(master_path),
        "separation_mechanism_csv": str(mechanism_path),
        "column_system_metadata_csv": str(metadata_path),
    }
