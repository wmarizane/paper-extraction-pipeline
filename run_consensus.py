#!/usr/bin/env python3
"""
Driver script for Phase 3 Consensus.
Finds extracted JSON files, aggregates them, and runs the DeepSeek Judge.
"""

import json
import time
import subprocess
from pathlib import Path
from pipeline.consensus_judge import ConsensusJudge

def load_json(path: Path) -> list:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("extracted_data", {}).get("conditions", [])
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def main():
    results_dir = Path("results")
    consensus_dir = results_dir / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    
    qwen_dir = results_dir / "qwen3.5-27b"
    mistral_dir = results_dir / "mistral-small-24b"
    
    if not qwen_dir.exists() or not mistral_dir.exists():
        print("Missing required model results directories.")
        return
        
    print(f"Initializing Consensus Judge...")
    judge = ConsensusJudge()
    
    # Loop over subfolders in the Qwen directory
    subfolders = [p.name for p in qwen_dir.iterdir() if p.is_dir()]
    if not subfolders:
        subfolders = [""]  # fallback to root if no subfolders exist
        
    for subfolder in subfolders:
        print(f"\n{'='*50}\nProcessing Subfolder: {subfolder or 'ROOT'}\n{'='*50}")
        qwen_sub_dir = qwen_dir / subfolder if subfolder else qwen_dir
        mistral_sub_dir = mistral_dir / subfolder if subfolder else mistral_dir
        consensus_sub_dir = consensus_dir / subfolder if subfolder else consensus_dir
        
        consensus_sub_dir.mkdir(parents=True, exist_ok=True)
        
        papers = set()
        for f in qwen_sub_dir.glob("*_latest.json"):
            papers.add(f.stem.replace("_latest", ""))
            
        if not papers:
            print(f"No papers found in {subfolder}.")
            continue
            
        print(f"Found {len(papers)} papers in {subfolder}.")
        
        for paper in sorted(list(papers)):
            print(f"\n>>> Running Consensus for: {paper}")
            qwen_file = qwen_sub_dir / f"{paper}_latest.json"
            mistral_file = mistral_sub_dir / f"{paper}_latest.json"
            
            qwen_conds = load_json(qwen_file)
            mistral_conds = load_json(mistral_file)
            
            for cond in qwen_conds: cond["source_model"] = "qwen"
            for cond in mistral_conds: cond["source_model"] = "mistral"
            
            print(f"Loaded {len(qwen_conds)} conditions from Qwen, {len(mistral_conds)} from Mistral.")
            
            consensus_calls = 0
            validation_calls = 0
            retry_calls = 0
            
            try:
                start_time = time.time()
                final_data = judge.run_bidirectional_consensus(qwen_conds, mistral_conds)
                consensus_calls += 2
                validation_calls += 1
                
                # Phase 2 Feedback Loop Orchestration (Max 1 Retry)
                if final_data.get("requires_retry"):
                    print("⚠️ Judge requested a retry based on quality feedback!")
                    feedback_dict = final_data.get("feedback_for_models", {})
                    
                    for model_name, feedback_str in feedback_dict.items():
                        if feedback_str:
                            print(f"🔄 Retrying {model_name} with feedback: {feedback_str}")
                            if subfolder:
                                pdf_path = Path("Inputs") / subfolder / f"{paper}.pdf"
                            else:
                                pdf_path = Path("Inputs") / f"{paper}.pdf"
                                
                            if not pdf_path.exists():
                                print(f"Warning: PDF not found for retry at {pdf_path}")
                                continue
                                
                            cmd = [
                                "python", "run_local.py", 
                                str(pdf_path), 
                                "--model", model_name, 
                                "--feedback", feedback_str
                            ]
                            if subfolder:
                                cmd.extend(["--subfolder", subfolder])
                                
                            subprocess.run(cmd, check=False)
                    
                    # Reload JSONs after retry
                    print(f"🔄 Re-running consensus after retry...")
                    qwen_conds = load_json(qwen_file)
                    mistral_conds = load_json(mistral_file)
                    
                    for cond in qwen_conds: cond["source_model"] = "qwen"
                    for cond in mistral_conds: cond["source_model"] = "mistral"
                    
                    # Run consensus again (2nd pass, no further retries)
                    final_data = judge.run_bidirectional_consensus(qwen_conds, mistral_conds)
                    consensus_calls += 2
                    validation_calls += 1
                    retry_calls += 1
                
                final_conds = final_data.get("final_consensus", {}).get("extracted_conditions", [])
                
                # Hard filter: reject conditions missing too many critical experimental fields.
                # Rule 1: column_name + flow_rate + detector all absent → literature reference
                # Rule 2: 4+ of 6 critical fields absent → too vague to be real data
                CRITICAL_FIELDS = ["column_name", "mobile_phase_ratio", "temperature_celsius",
                                   "flow_rate", "detector", "pore_size"]
                pre_filter = len(final_conds)
                final_conds = [
                    c for c in final_conds
                    if not (
                        # Rule 1: classic literature reference signal
                        (not c.get("column_name") and not c.get("flow_rate") and not c.get("detector"))
                        or
                        # Rule 2: too many missing critical fields
                        sum(1 for f in CRITICAL_FIELDS if not c.get(f)) >= 4
                    )
                ]
                if len(final_conds) < pre_filter:
                    print(f"  ⚠️ Filtered {pre_filter - len(final_conds)} null-heavy conditions")

                # Sanitize: convert string "null" to actual None across all fields.
                # LLMs sometimes output the literal string "null" instead of JSON null.
                for c in final_conds:
                    for key in list(c.keys()):
                        if isinstance(c[key], str) and c[key].lower() == "null":
                            c[key] = None
                        # Also handle nested dicts (e.g. aqueous_parameters)
                        elif isinstance(c[key], dict):
                            for sub_key in list(c[key].keys()):
                                if isinstance(c[key][sub_key], str) and c[key][sub_key].lower() == "null":
                                    c[key][sub_key] = None

                # Split comma-separated analyte_polymer into separate conditions.
                # Exception: commas inside parentheses (chemical names) or single-token commas.
                import re
                split_conds = []
                for c in final_conds:
                    analyte = c.get("analyte_polymer") or ""
                    # Don't split if the comma is inside parentheses or if there's only one token
                    # Simple heuristic: split on ", " but not inside parentheses
                    if ", " in analyte:
                        # Remove content in parentheses for splitting decision
                        stripped = re.sub(r'\([^)]*\)', '', analyte)
                        parts = [p.strip() for p in stripped.split(", ") if p.strip()]
                        if len(parts) >= 2 and all(len(p) > 1 for p in parts):
                            for part in parts:
                                new_c = dict(c)
                                new_c["analyte_polymer"] = part
                                split_conds.append(new_c)
                            continue
                    split_conds.append(c)
                if len(split_conds) != len(final_conds):
                    print(f"  ℹ️ Split comma-separated analytes: {len(final_conds)} → {len(split_conds)} conditions")
                final_conds = split_conds
                print(f"✅ Consensus reached: {len(final_conds)} conditions after null-heavy filter.")
                
                output_file = consensus_sub_dir / f"{paper}_consensus.json"
                
                # Wrap in our standard format
                total_time = time.time() - start_time
                if "metadata" not in final_data:
                    final_data["metadata"] = {}
                
                final_data["metadata"] = {
                    "source_pdf": f"{paper}.pdf",
                    "model": "deepseek-r1-32b-consensus",
                    "inputs": ["qwen3.5-27b", "mistral-small-24b"],
                    "pipeline_metrics": {
                        "consensus_calls": consensus_calls,
                        "validation_calls": validation_calls,
                        "retry_calls": retry_calls,
                        "consensus_runtime_seconds": round(total_time, 2)
                    }
                }
                
                out_json = {
                    "metadata": final_data["metadata"],
                    "summary": {
                        "total_conditions": len(final_conds)
                    },
                    "extracted_data": {
                        "conditions": final_conds
                    }
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(out_json, f, indent=2)
                    
            except Exception as e:
                print(f"❌ Consensus failed for {paper}: {e}")

if __name__ == "__main__":
    main()
