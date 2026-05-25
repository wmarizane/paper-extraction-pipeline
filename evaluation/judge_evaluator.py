#!/usr/bin/env python3
"""
Quantitative Judge Evaluator (Position Bias)

Implements the swapped-prompt methodology from the 2025 IJCNLP paper to
evaluate the Position Consistency (PC) of LLM judges.

For each paper in the results directory:
  Run A: Prompt with [Candidate A: Qwen, Candidate B: Mistral]
  Run B: Prompt with [Candidate A: Mistral, Candidate B: Qwen]

Calculates PC: percentage of papers where Run A output == Run B output.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import argparse
import sys

from pipeline.consensus_judge import ConsensusJudge

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def load_json(path: Path) -> list:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("extracted_data", {}).get("conditions", [])
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return []

def compare_json_outputs(data1: Dict, data2: Dict) -> bool:
    """Returns True if the final_consensus arrays are identical, ignoring order."""
    def _normalize(data: Dict):
        conds = data.get("final_consensus", {}).get("extracted_conditions", [])
        # Convert to string and sort to ignore list order differences
        str_conds = [json.dumps(c, sort_keys=True) for c in conds]
        return sorted(str_conds)
    
    return _normalize(data1) == _normalize(data2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Judge Position Bias")
    parser.add_argument("--models", nargs="+", default=["deepseek-r1-32b", "qwen3.5-27b", "mistral-small-24b"])
    parser.add_argument("--model", type=str, default=None, help="Evaluate a single model in an isolated process")
    args = parser.parse_args()
    
    results_dir = Path("results")
    qwen_dir = results_dir / "qwen3.5-27b"
    mistral_dir = results_dir / "mistral-small-24b"
    
    if not qwen_dir.exists() or not mistral_dir.exists():
        logger.error("Missing required model results directories.")
        sys.exit(1)
        
    papers = set()
    for f in qwen_dir.glob("*_latest.json"):
        papers.add(f.stem.replace("_latest", ""))
        
    if not papers:
        logger.error("No papers found to process.")
        sys.exit(1)
        
    if args.model:
        judge_model = args.model
        logger.info(f"==== Evaluating Judge (Isolated Process): {judge_model} ====")
        
        # Override the judge model in settings
        from config.settings import settings
        settings.llm_model = judge_model
        
        # Create ConsensusJudge with init_llm=False to avoid double-initializing the default model
        judge = ConsensusJudge(model_name=judge_model, init_llm=False)
        
        # Re-initialize vLLM exactly once for this specific model
        from vllm import LLM
        vllm_kwargs = {
            "model": judge.model_config.hf_id,
            "gpu_memory_utilization": 0.85,
            "max_model_len": judge.model_config.max_model_len,
            "trust_remote_code": True,
        }
        vllm_kwargs.update(judge.model_config.vllm_kwargs)
        
        judge.llm = LLM(**vllm_kwargs)
        
        consistent_count = 0
        total_evals = 0
        
        for paper in sorted(list(papers)):
            qwen_file = qwen_dir / f"{paper}_latest.json"
            mistral_file = mistral_dir / f"{paper}_latest.json"
            
            qwen_conds = load_json(qwen_file)
            mistral_conds = load_json(mistral_file)
            
            if not qwen_conds and not mistral_conds:
                continue # Skip papers where both failed to extract anything
                
            total_evals += 1
            
            # Run A: Qwen first, Mistral second
            logger.info(f"[{paper}] Run A: Qwen -> Mistral")
            out_A = judge.run_consensus(qwen_conds, mistral_conds)
            
            # Run B: Mistral first, Qwen second
            logger.info(f"[{paper}] Run B: Mistral -> Qwen")
            out_B = judge.run_consensus(mistral_conds, qwen_conds)
            
            # Calculate consistency
            if compare_json_outputs(out_A, out_B):
                logger.info(f"[{paper}] Result: CONSISTENT (1.0)")
                consistent_count += 1
            else:
                logger.info(f"[{paper}] Result: BIASED (0.0)")
                
        pc_score = (consistent_count / total_evals) if total_evals > 0 else 0.0
        
        result_data = {
            "model": judge_model,
            "pc_score": pc_score,
            "consistent_count": consistent_count,
            "total_evals": total_evals
        }
        
        out_path = results_dir / f"judge_eval_{judge_model}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
            
        logger.info(f"Saved isolated model results to {out_path}")
        logger.info(f"Final Score for {judge_model}: {pc_score:.2f}")
        sys.exit(0)
        
    else:
        logger.info(f"Starting Position Bias Evaluation on {len(papers)} papers.")
        logger.info(f"Models to evaluate as judge: {args.models}\n")
        
        results_matrix = {}
        import subprocess
        import os
        
        for judge_model in args.models:
            logger.info(f"==== Evaluating Judge: {judge_model} ====")
            out_path = results_dir / f"judge_eval_{judge_model}.json"
            if out_path.exists():
                out_path.unlink() # remove old run
                
            cmd = [sys.executable, "-m", "evaluation.judge_evaluator", "--model", judge_model]
            logger.info(f"Spawning subprocess: {' '.join(cmd)}")
            
            # Forward environment variables and ensure current folder is in PYTHONPATH
            env = os.environ.copy()
            env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.getcwd()
            
            # Run subprocess and wait for completion
            subprocess.run(cmd, check=True, env=env)
            
            if not out_path.exists():
                logger.error(f"Subprocess failed to generate results file: {out_path}")
                sys.exit(1)
                
            with open(out_path, "r", encoding="utf-8") as f:
                res = json.load(f)
                
            results_matrix[judge_model] = {
                "Position Consistency": f"{res['pc_score']:.2f} ({res['consistent_count']}/{res['total_evals']})"
            }
            
            # Clean up temporary results file
            out_path.unlink()
            
        logger.info("==== FINAL EVALUATION RESULTS ====")
        logger.info(f"{'Judge Model':<20} | {'Position Consistency':<20}")
        logger.info("-" * 45)
        for model, scores in results_matrix.items():
            logger.info(f"{model:<20} | {scores['Position Consistency']:<20}")

if __name__ == "__main__":
    main()
