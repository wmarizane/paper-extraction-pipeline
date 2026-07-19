"""Reproducibility provenance for the extraction & consensus pipeline.

Implements Dr. Wang's item 6 (Digital Discovery reproducibility): a durable
per-run record of *what produced each result* — exact model identity, sampling
configuration, library versions, the pipeline code commit, generation date,
and an input fingerprint (so an output can be tied to its exact input).

Intentionally free of heavy imports (no vllm/torch at import time) so it can be
built and unit-tested anywhere. Library versions and the model snapshot are
resolved best-effort and degrade to None rather than raising.
"""

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Packages worth pinning in a reproducibility appendix.
_TRACKED_PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "pydantic", "pymupdf4llm", "tiktoken")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: Optional[str]) -> Optional[str]:
    """Stable SHA-256 of a string (UTF-8). None -> None."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def library_versions() -> Dict[str, Optional[str]]:
    """Resolved versions of the packages that affect output, plus Python.
    Missing packages map to None instead of raising."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover - importlib.metadata always present on 3.8+
        return {"python": sys.version.split()[0]}

    out: Dict[str, Optional[str]] = {"python": sys.version.split()[0]}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
        except Exception:
            out[pkg] = None
    return out


def git_commit(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """The pipeline's git commit at run time (sha, short sha, dirty flag).
    Degrades to nulls if git or the repo is unavailable."""
    root = str(repo_root or Path(__file__).resolve().parent.parent)

    def _git(*args) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "-C", root, *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        "short_sha": sha[:9] if sha else None,
        "dirty": bool(status) if status is not None else None,
    }


def resolve_hf_snapshot(hf_id: str) -> Optional[str]:
    """Best-effort resolution of the local HuggingFace snapshot commit hash for
    a model id, so the exact weights are pinned. Returns None if not resolvable
    (e.g. model not in the local cache). Never raises."""
    if not hf_id or "/" not in hf_id:
        return None
    cache = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None)
        or os.path.expanduser("~/.cache/huggingface/hub")
    )
    org, name = hf_id.split("/", 1)
    model_dir = Path(cache) / f"models--{org}--{name}"
    try:
        ref = model_dir / "refs" / "main"
        if ref.is_file():
            return ref.read_text().strip() or None
        snaps = model_dir / "snapshots"
        if snaps.is_dir():
            children = [p.name for p in snaps.iterdir() if p.is_dir()]
            if len(children) == 1:
                return children[0]
    except Exception:
        return None
    return None


def _model_block(model_name: str) -> Dict[str, Any]:
    """Exact model identity from the registry + resolved snapshot."""
    from config.model_registry import get_model_config  # local import: no vllm dependency

    cfg = get_model_config(model_name)
    return {
        "short_name": cfg.short_name,
        "hf_id": cfg.hf_id,
        "revision": resolve_hf_snapshot(cfg.hf_id),
        "quantization": cfg.quantization,
        "max_model_len": cfg.max_model_len,
        "vllm_kwargs": dict(cfg.vllm_kwargs),
    }


def input_fingerprint(text: Optional[str], parsed_markdown_path: Optional[str] = None,
                      token_count: Optional[int] = None) -> Dict[str, Any]:
    """Fingerprint of the input text (hash + length) plus a pointer to the
    saved parsed markdown, so an output can be tied to its exact input."""
    return {
        "parsed_markdown_path": parsed_markdown_path,
        "sha256": sha256_text(text),
        "char_len": len(text) if text is not None else None,
        "token_count": token_count,
    }


def build_extraction_provenance(
    model_name: str,
    sampling: Dict[str, Any],
    prompt_version: str,
    schema_name: str,
    input_text: Optional[str] = None,
    parsed_markdown_path: Optional[str] = None,
    token_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Provenance block for a Phase-1 extraction output."""
    return {
        "generated_at_utc": _now_utc_iso(),
        "pipeline_git_commit": git_commit(),
        "model": _model_block(model_name),
        "sampling": dict(sampling),
        "prompt_version": prompt_version,
        "output_schema": schema_name,
        "library_versions": library_versions(),
        "input": input_fingerprint(input_text, parsed_markdown_path, token_count),
    }


def build_consensus_provenance(
    judge_model_name: str,
    sampling: Dict[str, Any],
    prompt_version: str,
    schema_name: str,
    input_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Provenance block for a Phase-2 consensus output. Inputs are the per-model
    extraction files that fed the judge."""
    return {
        "generated_at_utc": _now_utc_iso(),
        "pipeline_git_commit": git_commit(),
        "judge_model": _model_block(judge_model_name),
        "sampling": dict(sampling),
        "prompt_version": prompt_version,
        "output_schema": schema_name,
        "library_versions": library_versions(),
        "input_extraction_files": list(input_files) if input_files else [],
    }
