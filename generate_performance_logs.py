import json
from pathlib import Path
import csv

def generate_performance_logs():
    results_dir = Path("results")
    qwen_dir = results_dir / "qwen3.5-27b"
    mistral_dir = results_dir / "mistral-small-24b"
    consensus_dir = results_dir / "consensus"

    output_csv = Path("performance_logs.csv")
    
    # Gather all papers
    papers = set()
    for f in qwen_dir.rglob("*_latest.json"):
        papers.add(f.stem.replace("_latest", ""))
        
    rows = []
    
    for paper in sorted(papers):
        row = {"Paper": paper}
        
        # Qwen
        qwen_file = list(qwen_dir.rglob(f"{paper}_latest.json"))
        if qwen_file:
            with open(qwen_file[0], "r") as f:
                q_data = json.load(f)
                pm = q_data.get("metadata", {}).get("pipeline_metrics", {})
                row["Qwen Runtime (s)"] = pm.get("stages", {}).get("llm_extraction", {}).get("time_seconds", "N/A")
                row["Qwen Extraction Calls (Chunks)"] = pm.get("stages", {}).get("chunking", {}).get("num_chunks", "N/A")
        
        # Mistral
        mistral_file = list(mistral_dir.rglob(f"{paper}_latest.json"))
        if mistral_file:
            with open(mistral_file[0], "r") as f:
                m_data = json.load(f)
                pm = m_data.get("metadata", {}).get("pipeline_metrics", {})
                row["Mistral Runtime (s)"] = pm.get("stages", {}).get("llm_extraction", {}).get("time_seconds", "N/A")
                row["Mistral Extraction Calls (Chunks)"] = pm.get("stages", {}).get("chunking", {}).get("num_chunks", "N/A")

        # Consensus
        consensus_file = list(consensus_dir.rglob(f"{paper}_consensus.json"))
        if consensus_file:
            row["Consensus Generated"] = "Yes"
        else:
            row["Consensus Generated"] = "No"
            
        rows.append(row)
        
    with open(output_csv, "w", newline="") as f:
        fieldnames = ["Paper", "Qwen Runtime (s)", "Qwen Extraction Calls (Chunks)", 
                      "Mistral Runtime (s)", "Mistral Extraction Calls (Chunks)", "Consensus Generated"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Generated {output_csv} with available metrics for {len(rows)} papers.")

if __name__ == "__main__":
    generate_performance_logs()
