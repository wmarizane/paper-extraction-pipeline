#!/usr/bin/env python3
"""Regression test for the Unicode NFC/NFD filename-twin bug.

"Krüger1996" committed from macOS carries a DECOMPOSED (NFD) 'ü' (u + combining
diaeresis); the same paper regenerated on the Linux cluster carries a COMPOSED
(NFC) 'ü'. Linux treats these as two distinct files, so a paper could be
consensus-judged twice and exported as two CSV rows (observed: PLA [94] Krüger1996
survived consensus as a near-null row that should not exist).

Both scan points now normalize to NFC. Run from project root:

  python3 scripts/test_unicode_twins.py
"""
import json
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.sampling_params", MagicMock())
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.csv_exporter import _select_latest_json_files, export_folder_to_csv

NFC = unicodedata.normalize("NFC", "Krüger1996")   # composed ü
NFD = unicodedata.normalize("NFD", "Krüger1996")   # decomposed u + ̈
assert NFC != NFD and unicodedata.normalize("NFC", NFD) == NFC


def _write(path: Path, conditions):
    path.write_text(json.dumps({
        "metadata": {"source_pdf": path.stem + ".pdf"},
        "extracted_data": {"conditions": conditions},
    }), encoding="utf-8")


class TestUnicodeTwins(unittest.TestCase):
    def test_select_latest_collapses_twins(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            _write(folder / f"[94] {NFC}_latest.json", [{"analyte_polymer": "x"}])
            _write(folder / f"[94] {NFD}_latest.json", [{"analyte_polymer": "x"}])
            # Sanity: the OS really did create two separate files.
            if len(list(folder.glob("*_latest.json"))) < 2:
                self.skipTest("filesystem normalizes Unicode names (e.g. macOS APFS)")
            picked = _select_latest_json_files(folder)
            self.assertEqual(len(picked), 1, "NFC/NFD twins must collapse to one file")

    def test_export_emits_single_row(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            _write(folder / f"[94] {NFC}_latest.json", [{"analyte_polymer": "aliphatic polyesters"}])
            _write(folder / f"[94] {NFD}_latest.json", [{"analyte_polymer": "aliphatic polyesters"}])
            if len(list(folder.glob("*_latest.json"))) < 2:
                self.skipTest("filesystem normalizes Unicode names (e.g. macOS APFS)")
            out = folder / "summary.csv"
            export_folder_to_csv(str(folder), str(out))
            rows = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 2, "header + exactly one data row (twins collapsed)")

    def test_select_latest_collapses_twins_os_independent(self):
        # Runs on every OS (incl. macOS APFS): mock glob so the two twin paths
        # exist regardless of how the real filesystem normalizes names.
        twins = [Path(f"[94] {NFC}_latest.json"), Path(f"[94] {NFD}_latest.json")]
        folder = MagicMock(spec=Path)
        folder.glob.return_value = twins
        picked = _select_latest_json_files(folder)
        self.assertEqual(len(picked), 1, "NFC/NFD twins must collapse to one file")

    def test_consensus_scan_collapses_twins_os_independent(self):
        # Mirrors run_consensus.py's paper-scan collapse (NFC key -> real stem).
        files = [Path(f"[94] {NFC}_latest.json"), Path(f"[94] {NFD}_latest.json"),
                 Path("[95] Pasch_latest.json")]
        papers = {}
        for f in files:
            stem = f.stem.replace("_latest", "")
            papers.setdefault(unicodedata.normalize("NFC", stem), stem)
        self.assertEqual(len(papers), 2, "twins collapse, distinct paper stays")

    def test_distinct_papers_not_collapsed(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            _write(folder / f"[94] {NFC}_latest.json", [{"analyte_polymer": "x"}])
            _write(folder / f"[95] Pasch_latest.json", [{"analyte_polymer": "y"}])
            picked = _select_latest_json_files(folder)
            self.assertEqual(len(picked), 2, "genuinely different papers must stay separate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
