#!/usr/bin/env python3
"""Tests for pipeline/provenance.py (reproducibility metadata, item 6).

Runs anywhere — provenance.py has no vllm/torch import. Run from project root:
  python3 scripts/test_provenance.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import provenance as P


class TestHelpers(unittest.TestCase):
    def test_sha256_stable_and_none(self):
        self.assertIsNone(P.sha256_text(None))
        h1 = P.sha256_text("hello world")
        h2 = P.sha256_text("hello world")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)
        self.assertNotEqual(h1, P.sha256_text("hello world!"))

    def test_library_versions_has_python(self):
        v = P.library_versions()
        self.assertIn("python", v)
        self.assertRegex(v["python"], r"^\d+\.\d+")
        # tracked packages are present as keys even if not installed (value None)
        for pkg in ("vllm", "torch", "transformers"):
            self.assertIn(pkg, v)

    def test_git_commit_shape(self):
        g = P.git_commit()
        self.assertEqual(set(g), {"sha", "short_sha", "dirty"})
        if g["sha"]:  # running inside the repo
            self.assertEqual(g["short_sha"], g["sha"][:9])

    def test_git_commit_binary_free_fallback(self):
        # Simulate a compute node with no usable git binary: the subprocess
        # path fails, so git_commit must still resolve HEAD by reading .git.
        from unittest.mock import patch
        repo_root = Path(__file__).resolve().parent.parent
        expected = P._read_head_sha(repo_root)
        with patch("pipeline.provenance.subprocess.check_output", side_effect=OSError("git not found")):
            g = P.git_commit()
        self.assertEqual(g["sha"], expected)
        if expected:
            self.assertEqual(g["short_sha"], expected[:9])
            self.assertEqual(len(expected), 40)

    def test_resolve_hf_snapshot_missing_is_none(self):
        self.assertIsNone(P.resolve_hf_snapshot("nonexistent-org/nonexistent-model-xyz"))
        self.assertIsNone(P.resolve_hf_snapshot("not-a-valid-id"))

    def test_input_fingerprint(self):
        fp = P.input_fingerprint("abc", "parsed_md/x.md", token_count=5)
        self.assertEqual(fp["char_len"], 3)
        self.assertEqual(fp["token_count"], 5)
        self.assertEqual(fp["parsed_markdown_path"], "parsed_md/x.md")
        self.assertEqual(fp["sha256"], P.sha256_text("abc"))


class TestExtractionProvenance(unittest.TestCase):
    def test_block_shape_and_model_identity(self):
        block = P.build_extraction_provenance(
            model_name="qwen3.5-27b",
            sampling={"temperature": 0.1, "top_p": 0.9, "max_tokens": 2048},
            prompt_version="extraction-v2-2026-04-17",
            schema_name="polymer-lccc",
            input_text="paper text",
            parsed_markdown_path="parsed_md/paper.md",
            token_count=42,
        )
        for k in ("generated_at_utc", "pipeline_git_commit", "model", "sampling",
                  "prompt_version", "output_schema", "library_versions", "input"):
            self.assertIn(k, block)
        # exact model identity resolved from the registry
        self.assertEqual(block["model"]["hf_id"], "QuantTrio/Qwen3.5-27B-AWQ")
        self.assertEqual(block["model"]["quantization"], "awq")
        self.assertEqual(block["sampling"]["temperature"], 0.1)
        self.assertEqual(block["input"]["token_count"], 42)
        self.assertEqual(block["prompt_version"], "extraction-v2-2026-04-17")


class TestConsensusProvenance(unittest.TestCase):
    def test_block_shape(self):
        block = P.build_consensus_provenance(
            judge_model_name="deepseek-r1-32b",
            sampling={"temperature": 0.6, "top_p": 0.9, "max_tokens": 8192},
            prompt_version="consensus-bidirectional-v1",
            schema_name="polymer-lccc-consensus",
            input_files=["results/qwen3.5-27b/PEG/a_latest.json",
                         "results/mistral-small-24b/PEG/a_latest.json"],
        )
        self.assertEqual(block["judge_model"]["short_name"], "deepseek-r1-32b")
        self.assertEqual(len(block["input_extraction_files"]), 2)
        self.assertIn("library_versions", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
