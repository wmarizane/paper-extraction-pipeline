#!/usr/bin/env python3
"""
Compares extraction results across multiple models.
Generates a Markdown table of conditions extracted per paper per model.
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found.")
        return

    # Data structure: paper_stats[paper_name][model_name] = stats
    paper_stats = defaultdict(lambda: defaultdict(dict))
    models_found = set()
    papers_found = set()

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        models_found.add(model_name)
        
        for json_file in model_dir.glob("*_latest.json"):
            paper_name = json_file.stem.replace("_latest", "")
            papers_found.add(paper_name)
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                total_conditions = data.get("summary", {}).get("total_conditions", 0)
                
                # Count total null/empty fields to measure hallucination/completeness
                null_count = 0
                total_fields = 0
                conditions = data.get("extracted_data", {}).get("conditions", [])
                
                for cond in conditions:
                    for k, v in cond.items():
                        total_fields += 1
                        if v is None or v == "" or v == []:
                            null_count += 1
                
                paper_stats[paper_name][model_name] = {
                    "conditions": total_conditions,
                    "nulls": null_count,
                    "total_fields": total_fields,
                    "success": True
                }
            except Exception as e:
                paper_stats[paper_name][model_name] = {
                    "success": False,
                    "error": str(e)
                }

    if not models_found:
        print("No model outputs found to compare.")
        return

    models = sorted(list(models_found))
    papers = sorted(list(papers_found))

    print("\n# Phase 2: Multi-Model Comparison\n")
    
    # 1. Extraction Count Table
    print("## Extracted Conditions per Paper")
    header = "| Paper | " + " | ".join(models) + " |"
    separator = "|-------|" + "|".join(["---" for _ in models]) + "|"
    print(header)
    print(separator)
    
    for paper in papers:
        row = [f"| **{paper}**"]
        for model in models:
            stats = paper_stats[paper].get(model)
            if not stats:
                row.append(" ❌ N/A ")
            elif not stats["success"]:
                row.append(" ❌ Error ")
            else:
                row.append(f" {stats['conditions']} ")
        row.append("|")
        print(" | ".join(row).replace("| |", "|"))

    print("\n## Null Fields (Lower is Better)")
    header = "| Paper | " + " | ".join(models) + " |"
    separator = "|-------|" + "|".join(["---" for _ in models]) + "|"
    print(header)
    print(separator)
    
    for paper in papers:
        row = [f"| **{paper}**"]
        for model in models:
            stats = paper_stats[paper].get(model)
            if not stats or not stats.get("success"):
                row.append(" - ")
            else:
                row.append(f" {stats['nulls']}/{stats['total_fields']} ")
        row.append("|")
        print(" | ".join(row).replace("| |", "|"))
        
    print("\n")

if __name__ == "__main__":
    main()
