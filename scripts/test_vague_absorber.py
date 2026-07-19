#!/usr/bin/env python3
"""Tests for the vague-row absorber (absorb_vague_conditions).

MUST absorb: a generic/less-precise row fully covered by a strictly-more-
specific same-analyte row in the same paper.
MUST NOT absorb: MW / end-group / architecture / block series (analyte guard),
genuinely different analytes sharing one critical condition, or rows that
carry mutually-unique fields.

Run from project root:
  python3 scripts/test_vague_absorber.py
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.sampling_params", MagicMock())
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pre_consensus_dedup import absorb_vague_conditions


def C(**kw):
    base = {
        "analyte_polymer": None, "critical_component": None,
        "stationary_phase_chemistry": None, "column_name": None,
        "mobile_phase_solvents": None, "mobile_phase_ratio": None,
        "temperature_celsius": None,
    }
    base.update(kw)
    return base


class TestMustAbsorb(unittest.TestCase):
    def test_generic_peg_shadow_of_mw_series(self):
        # [330] Wei2016 pattern: a summary "PEG" row shadowing PEG 2k/4k/6k
        # on the same column/ratio/temp. The generic row is absorbed; the MW
        # series stays intact.
        rows = [
            C(analyte_polymer="PEG", critical_component="PEG",
              column_name="XB-phenyl", mobile_phase_ratio="45", temperature_celsius="30"),
            C(analyte_polymer="PEG 2k", critical_component="PEG backbone",
              column_name="XB-phenyl", mobile_phase_ratio="45:55", temperature_celsius="30"),
            C(analyte_polymer="PEG 4k", critical_component="PEG backbone",
              column_name="XB-phenyl", mobile_phase_ratio="45:55", temperature_celsius="30"),
            C(analyte_polymer="PEG 6k", critical_component="PEG backbone",
              column_name="XB-phenyl", mobile_phase_ratio="45:55", temperature_celsius="30"),
        ]
        out = absorb_vague_conditions(rows)
        analytes = sorted(r["analyte_polymer"] for r in out)
        self.assertEqual(analytes, ["PEG 2k", "PEG 4k", "PEG 6k"], "generic PEG absorbed, MW series kept")

    def test_missing_field_vaguer_row(self):
        # Same analyte, one row missing temperature -> absorbed by the fuller row.
        rows = [
            C(analyte_polymer="PEG 600", critical_component="PEG",
              column_name="Onyx C18", mobile_phase_ratio="90"),
            C(analyte_polymer="PEG 600", critical_component="PEG",
              column_name="Onyx C18", mobile_phase_ratio="90", temperature_celsius="25"),
        ]
        out = absorb_vague_conditions(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["temperature_celsius"], "25", "kept the more complete row")


class TestMustNotAbsorb(unittest.TestCase):
    def test_mw_series_preserved(self):
        rows = [
            C(analyte_polymer="PEG 2k", critical_component="PEG backbone",
              column_name="XB-phenyl", mobile_phase_ratio="45:55", temperature_celsius="30"),
            C(analyte_polymer="PEG 6k", critical_component="PEG backbone",
              column_name="XB-phenyl", mobile_phase_ratio="45:55", temperature_celsius="30"),
        ]
        self.assertEqual(len(absorb_vague_conditions(rows)), 2, "MW series must never collapse")

    def test_different_analytes_same_critical_condition(self):
        # [372]/[A10] pattern: two DIFFERENT surfactant classes sharing the
        # same oxyethylene critical condition -> MULTIPLE ANALYTES, keep both.
        rows = [
            C(analyte_polymer="Fatty alcohol ethoxylates", critical_component="oxyethylene unit",
              column_name="Zorbax 300 C18", mobile_phase_ratio="85.8", temperature_celsius="25"),
            C(analyte_polymer="fatty acid polyglycol esters", critical_component="oxyethylene unit",
              column_name="Zorbax 300 C18", mobile_phase_ratio="85.8", temperature_celsius="25"),
        ]
        self.assertEqual(len(absorb_vague_conditions(rows)), 2, "distinct analytes must stay separate")

    def test_endgroup_variants_preserved(self):
        rows = [
            C(analyte_polymer="PIB-diol", critical_component="PIB-diol",
              column_name="YMC", mobile_phase_ratio="80.5/19.5", temperature_celsius="35"),
            C(analyte_polymer="PIB-diallyl", critical_component="PIB backbone",
              column_name="YMC", mobile_phase_ratio="80.5/19.5", temperature_celsius="35"),
        ]
        self.assertEqual(len(absorb_vague_conditions(rows)), 2, "end-group variants must stay separate")

    def test_homopolymer_vs_block_preserved(self):
        rows = [
            C(analyte_polymer="PEO", critical_component="PEO",
              column_name="Nucleosil Si 300", mobile_phase_ratio="96:4", temperature_celsius="30"),
            C(analyte_polymer="PS-b-PEO", critical_component="PEO",
              column_name="Nucleosil Si 300", mobile_phase_ratio="96:4", temperature_celsius="30"),
        ]
        self.assertEqual(len(absorb_vague_conditions(rows)), 2, "homopolymer vs block must stay separate")

    def test_row_with_unique_field_not_absorbed(self):
        # v has a temperature the "specific" row lacks -> v carries unique info,
        # must not be dropped.
        rows = [
            C(analyte_polymer="PEG", critical_component="PEG",
              column_name="XB-phenyl", mobile_phase_ratio="45", temperature_celsius="30"),
            C(analyte_polymer="PEG 2k", critical_component="PEG backbone",
              column_name="XB-phenyl", mobile_phase_ratio="45:55"),  # no temperature
        ]
        self.assertEqual(len(absorb_vague_conditions(rows)), 2, "row with unique field kept")

    def test_identical_rows_not_touched(self):
        # Exact duplicates are the fingerprint dedup's job, not the absorber's;
        # the absorber must leave them (no strict specificity difference).
        r = C(analyte_polymer="PEG", critical_component="PEG",
              column_name="XB-phenyl", mobile_phase_ratio="45:55", temperature_celsius="30")
        self.assertEqual(len(absorb_vague_conditions([dict(r), dict(r)])), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
