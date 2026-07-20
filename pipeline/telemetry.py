"""
pipeline/telemetry.py

Centralized telemetry for extraction and consensus runs.
Writes one JSON Lines (.jsonl) file per SLURM job, with one record per paper.
Also writes a human-readable summary CSV at the end of a run.
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

@dataclass
class PaperTelemetry:
    paper_name: str
    model: str
    phase: str                          # "extraction" | "consensus"

    # Timestamps
    wall_start: Optional[str] = None    # ISO-8601 UTC
    wall_end: Optional[str] = None
    wall_duration_s: Optional[float] = None

    # LLM calls breakdown
    initial_calls: int = 0
    retry_calls: int = 0
    total_calls: int = 0                # = initial + retry

    # Token usage (summed across all calls for this paper)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0               # = input + output

    # Per-call details (list of dicts, for the JSONL record)
    call_log: List[Dict[str, Any]] = field(default_factory=list)

    # Output
    conditions_extracted: int = 0

    # GPU
    pytorch_peak_gb: Optional[float] = None
    nvidia_smi_pre_mb: Optional[float] = None
    nvidia_smi_post_mb: Optional[float] = None
    gpu_delta_mb: Optional[float] = None

    # Chunking
    num_chunks: int = 0
    total_input_tokens_doc: int = 0    # token count of the parsed document

    # Status
    success: bool = False
    error: Optional[str] = None

    # Derived (computed at finish())
    conditions_per_minute: Optional[float] = None
    tokens_per_second: Optional[float] = None

    def start(self):
        self.wall_start = datetime.utcnow().isoformat() + "Z"

    def record_llm_call(self, call_type: str, input_tokens: int, output_tokens: int, duration_s: float, success: bool, error: Optional[str] = None):
        self.call_log.append({
            "call_type": call_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_s": duration_s,
            "success": success,
            "error": error
        })
        if call_type == "initial":
            self.initial_calls += 1
        elif call_type == "retry":
            self.retry_calls += 1
        self.total_calls = self.initial_calls + self.retry_calls

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens = self.total_input_tokens + self.total_output_tokens

    def record_gpu_memory(self, pytorch_peak_gb: float, nvidia_smi_pre_mb: Any, nvidia_smi_post_mb: Any):
        self.pytorch_peak_gb = pytorch_peak_gb
        
        def parse_mb(val) -> Optional[float]:
            if val is None:
                return None
            if isinstance(val, str) and val.strip().upper() == "N/A":
                return None
            try:
                return float(val)
            except ValueError:
                return None
                
        self.nvidia_smi_pre_mb = parse_mb(nvidia_smi_pre_mb)
        self.nvidia_smi_post_mb = parse_mb(nvidia_smi_post_mb)
        if self.nvidia_smi_pre_mb is not None and self.nvidia_smi_post_mb is not None:
            self.gpu_delta_mb = self.nvidia_smi_post_mb - self.nvidia_smi_pre_mb

    def finish(self, conditions_extracted: int, success: bool, error: Optional[str] = None):
        self.conditions_extracted = conditions_extracted
        self.success = success
        self.error = error
        
        if self.wall_start:
            start_dt = datetime.fromisoformat(self.wall_start.rstrip("Z"))
            end_dt = datetime.utcnow()
            self.wall_end = end_dt.isoformat() + "Z"
            self.wall_duration_s = (end_dt - start_dt).total_seconds()
            
            if self.wall_duration_s > 0:
                self.conditions_per_minute = round((self.conditions_extracted / self.wall_duration_s) * 60, 2)
                self.tokens_per_second = round(self.total_tokens / self.wall_duration_s, 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_csv_row(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop("call_log", None)
        return d


class TelemetryWriter:
    def __init__(self, output_dir: Path, job_id: Optional[str] = None):
        self.output_dir = output_dir
        self.job_id = job_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def append(self, tel: PaperTelemetry) -> None:
        jsonl_path = self.output_dir / f"{tel.phase}_telemetry_{self.job_id}.jsonl"
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(tel.to_dict()) + "\n")
        except Exception as e:
            print(f"Error writing telemetry to {jsonl_path}: {e}")

    def flush_csv(self) -> None:
        """Rewrites the CSV file from the current JSONL contents."""
        for phase in ["extraction", "consensus"]:
            jsonl_path = self.output_dir / f"{phase}_telemetry_{self.job_id}.jsonl"
            csv_path = self.output_dir / f"{phase}_telemetry_{self.job_id}.csv"
            
            if not jsonl_path.exists():
                continue
                
            try:
                records = []
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            record.pop("call_log", None)
                            records.append(record)
                
                if records:
                    fieldnames = list(records[0].keys())
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(records)
            except Exception as e:
                print(f"Error flushing telemetry CSV for {phase}: {e}")
