#!/usr/bin/env python3
"""
Driver script for Phase 3 Consensus.
Finds extracted JSON files, aggregates them, and runs the DeepSeek Judge.
"""

import json
import subprocess
import os
import unicodedata
from datetime import datetime
from pathlib import Path
from pipeline.consensus_judge import ConsensusJudge, CONSENSUS_PROMPT_VERSION
from pipeline.pre_consensus_dedup import dedup_model_conditions, absorb_vague_conditions
from pipeline.provenance import build_consensus_provenance
from pipeline.telemetry import PaperTelemetry, TelemetryWriter

def load_json(path: Path) -> list:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("extracted_data", {}).get("conditions", [])
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def main():
    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    tel_writer = TelemetryWriter(output_dir=Path("logs"), job_id=job_id)

    results_dir = Path("results")
    consensus_dir = results_dir / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    
    qwen_dir = results_dir / "qwen3.5-27b"
    llama_dir = results_dir / "mistral-small-24b"
    
    if not qwen_dir.exists() or not llama_dir.exists():
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
        llama_sub_dir = llama_dir / subfolder if subfolder else llama_dir
        consensus_sub_dir = consensus_dir / subfolder if subfolder else consensus_dir
        
        consensus_sub_dir.mkdir(parents=True, exist_ok=True)
        
        # Map NFC-normalized stem -> actual on-disk stem. Collapses Unicode
        # NFC/NFD filename twins (e.g. "Krüger" as decomposed NFD from macOS vs
        # composed NFC on Linux) so the judge doesn't process the same paper
        # twice; the real on-disk stem is kept so the file still opens.
        papers = {}
        for f in qwen_sub_dir.glob("*_latest.json"):
            stem = f.stem.replace("_latest", "")
            papers.setdefault(unicodedata.normalize("NFC", stem), stem)

        if not papers:
            print(f"No papers found in {subfolder}.")
            continue

        print(f"Found {len(papers)} papers in {subfolder}.")

        for paper in sorted(papers.values()):
            print(f"\n>>> Running Consensus for: {paper}")
            qwen_file = qwen_sub_dir / f"{paper}_latest.json"
            llama_file = llama_sub_dir / f"{paper}_latest.json"
            
            qwen_conds = load_json(qwen_file)
            llama_conds = load_json(llama_file)
            
            print(f"Loaded {len(qwen_conds)} conditions from Qwen, {len(llama_conds)} from LLaMA.")

            # Pre-consensus per-model dedup (Dr. Wang 7-7 feedback, items 1 & 4)
            qwen_before, llama_before = len(qwen_conds), len(llama_conds)
            qwen_conds = dedup_model_conditions(qwen_conds)
            llama_conds = dedup_model_conditions(llama_conds)
            if (qwen_before, llama_before) != (len(qwen_conds), len(llama_conds)):
                print(f"  Pre-consensus dedup: Qwen {qwen_before}->{len(qwen_conds)}, "
                      f"Mistral {llama_before}->{len(llama_conds)}")

            tel = PaperTelemetry(paper_name=paper, model="deepseek-r1-32b", phase="consensus")
            tel.start()
            
            try:
                final_data = judge.run_bidirectional_consensus(qwen_conds, llama_conds)
                
                # TODO: expose from ConsensusJudge
                tel.record_llm_call(call_type="initial", input_tokens=0, output_tokens=0, duration_s=0.0, success=True)
                tel.record_llm_call(call_type="initial", input_tokens=0, output_tokens=0, duration_s=0.0, success=True)
                tel.record_llm_call(call_type="initial", input_tokens=0, output_tokens=0, duration_s=0.0, success=True)
                
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
                    llama_conds = load_json(llama_file)

                    # Re-dedup after retry (pre-consensus layer)
                    qwen_conds = dedup_model_conditions(qwen_conds)
                    llama_conds = dedup_model_conditions(llama_conds)

                    # Run consensus again (2nd pass, no further retries)
                    final_data = judge.run_bidirectional_consensus(qwen_conds, llama_conds)
                    tel.record_llm_call(call_type="retry", input_tokens=0, output_tokens=0, duration_s=0.0, success=True)
                
                final_conds = final_data.get("final_consensus", {}).get("extracted_conditions", [])

                # Absorb residual vague rows (generic analyte / less-precise
                # fields) that a strictly-more-specific same-paper row covers.
                absorb_before = len(final_conds)
                final_conds = absorb_vague_conditions(final_conds)
                if len(final_conds) != absorb_before:
                    print(f"  Vague-row absorb: {absorb_before}->{len(final_conds)}")

                print(f"✅ Consensus reached: {len(final_conds)} merged conditions.")
                
                tel.finish(conditions_extracted=len(final_conds), success=True)
                try:
                    tel_writer.append(tel)
                    tel_writer.flush_csv()
                except Exception:
                    pass
                
                output_file = consensus_sub_dir / f"{paper}_consensus.json"
                
                # Wrap in our standard format
                try:
                    reproducibility = build_consensus_provenance(
                        judge_model_name="deepseek-r1-32b",
                        sampling=judge.sampling_config,
                        prompt_version=CONSENSUS_PROMPT_VERSION,
                        schema_name="polymer-lccc-consensus",
                        input_files=[str(qwen_file), str(llama_file)],
                    )
                except Exception as e:
                    print(f"Provenance warning: {e}")
                    reproducibility = {"error": str(e)}

                out_json = {
                    "metadata": {
                        "source_pdf": f"{paper}.pdf",
                        "model": "deepseek-r1-32b-consensus",
                        "inputs": ["qwen3.5-27b", "mistral-small-24b"],
                        "reproducibility": reproducibility
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
                tel.finish(conditions_extracted=0, success=False, error=str(e))
                try:
                    tel_writer.append(tel)
                    tel_writer.flush_csv()
                except Exception:
                    pass

if __name__ == "__main__":
    main()
