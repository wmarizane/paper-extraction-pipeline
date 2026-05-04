#!/usr/bin/env python3
"""
Driver script for Phase 3 Consensus.
Finds extracted JSON files, aggregates them, and runs the DeepSeek Judge.
"""

import json
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
    llama_dir = results_dir / "llama3.3-70b"
    
    if not qwen_dir.exists() or not llama_dir.exists():
        print("Missing required model results directories.")
        return
        
    papers = set()
    for f in qwen_dir.glob("*_latest.json"):
        papers.add(f.stem.replace("_latest", ""))
        
    if not papers:
        print("No papers found to process.")
        return
        
    print(f"Initializing Consensus Judge for {len(papers)} papers...")
    judge = ConsensusJudge()
    
    for paper in sorted(list(papers)):
        print(f"\n>>> Running Consensus for: {paper}")
        qwen_file = qwen_dir / f"{paper}_latest.json"
        llama_file = llama_dir / f"{paper}_latest.json"
        
        qwen_conds = load_json(qwen_file)
        llama_conds = load_json(llama_file)
        
        print(f"Loaded {len(qwen_conds)} conditions from Qwen, {len(llama_conds)} from LLaMA.")
        
        try:
            final_data = judge.run_consensus(qwen_conds, llama_conds)
            final_conds = final_data.get("extracted_conditions", [])
            print(f"✅ Consensus reached: {len(final_conds)} merged conditions.")
            
            output_file = consensus_dir / f"{paper}_consensus.json"
            
            # Wrap in our standard format
            out_json = {
                "metadata": {
                    "source_pdf": f"{paper}.pdf",
                    "model": "deepseek-r1-32b-consensus",
                    "inputs": ["qwen3.5-27b", "llama3.3-70b"]
                },
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
