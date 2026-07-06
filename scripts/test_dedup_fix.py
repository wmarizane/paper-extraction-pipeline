#!/usr/bin/env python3
"""
Comprehensive test suite for the generalized consensus dedup fix.
Tests both known duplicate cases AND unseen polymer types.
All 20 cases must pass before running dedup post-processing.

Run from project root:
  python3 scripts/test_dedup_fix.py
"""
import sys, re, unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.modules['vllm'] = MagicMock()
sys.modules['vllm.sampling_params'] = MagicMock()
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.consensus_judge import ConsensusJudge
j = ConsensusJudge(init_llm=False)


class TestAnalyteBaseFamilyMatch(unittest.TestCase):
    """Unit tests for _analyte_base_family_match in isolation."""

    # ── MUST MATCH ────────────────────────────────────────────────────────────

    def test_pla_endgroup_vs_generic(self):
        self.assertTrue(j._analyte_base_family_match(
            "linear C4H9-PLA-OH", "poly(L-lactide)"),
            "End-group-specific PLA name vs generic PLA — must match")

    def test_pla_abbreviation_vs_full(self):
        self.assertTrue(j._analyte_base_family_match(
            "Linear PLAs", "poly(lactic acid)"),
            "PLAs abbreviation vs full name — must match")

    def test_ppo_verbose_form(self):
        self.assertTrue(j._analyte_base_family_match(
            "ppo homopolymer", "ppo"),
            "Verbose descriptor form — must match")

    def test_parenthetical_annotation(self):
        self.assertTrue(j._analyte_base_family_match(
            "polyisoprene (1,4-pi)", "polyisoprene"),
            "Parenthetical annotation — must match")

    def test_unicode_em_dash(self):
        self.assertTrue(j._analyte_base_family_match(
            "styrene\u2014butadiene block copolymers",
            "styrene-butadiene block copolymers"),
            "Em-dash vs ASCII hyphen — must match")

    def test_unicode_en_dash_in_ratio_context(self):
        self.assertTrue(j._analyte_base_family_match(
            "eo\u2013po\u2013eo triblock copolymer",
            "eo-po-eo triblock copolymer"),
            "En-dash vs ASCII hyphen — must match")

    def test_typo_variant(self):
        self.assertTrue(j._analyte_base_family_match(
            "poly(oxypropylenpolyolen)", "poly(oxypropylenpolyol)"),
            "Typo variant (one substring of other) — must match")

    def test_ps_abbreviation(self):
        self.assertTrue(j._analyte_base_family_match(
            "PS", "polystyrene"),
            "Standard abbreviation PS — must match")

    def test_pcl_novel_polymer(self):
        """PCL was not in original CANONICAL_POLYMERS — tests extended dict generality."""
        self.assertTrue(j._analyte_base_family_match(
            "PCL", "poly(caprolactone)"),
            "PCL abbreviation (extended dict) — must match")

    def test_pdms_novel_polymer(self):
        self.assertTrue(j._analyte_base_family_match(
            "PDMS", "poly(dimethylsiloxane)"),
            "PDMS abbreviation (extended dict) — must match")

    # ── MUST NOT MATCH ────────────────────────────────────────────────────────

    def test_mw_series_block_three_digits(self):
        self.assertFalse(j._analyte_base_family_match("peg 2010", "peg 6240"),
            "MW series (3-digit) — must NOT match")

    def test_mw_series_block_k_suffix(self):
        self.assertFalse(j._analyte_base_family_match("peg 2k", "peg 6k"),
            "MW series (k-suffix) — must NOT match")

    def test_chain_length_series_block(self):
        self.assertFalse(j._analyte_base_family_match("c10-peo", "c12-peo"),
            "Chain-length variants (c10 vs c12) — must NOT match")

    def test_functional_group_variants(self):
        self.assertFalse(j._analyte_base_family_match("pib-diol", "pib-diallyl"),
            "Different end-group functionality (diol vs diallyl) — must NOT match")

    def test_homopolymer_vs_block_copolymer(self):
        self.assertFalse(j._analyte_base_family_match(
            "peo", "peo-ppo block copolymer"),
            "Homopolymer vs block copolymer — must NOT match")

    def test_block_copolymer_vs_homopolymer_dash(self):
        self.assertFalse(j._analyte_base_family_match("ps-b-pmma", "ps"),
            "Block copolymer vs homopolymer (-b-) — must NOT match")

    def test_different_polymers(self):
        self.assertFalse(j._analyte_base_family_match("polystyrene", "poly(methyl methacrylate)"),
            "Completely different polymers — must NOT match")

    def test_ring_vs_linear_same_base(self):
        self.assertFalse(j._analyte_base_family_match("ring-ps", "linear-ps"),
            "Different architectures (ring vs linear) — must NOT match")


class TestChromatographicMatchEndToEnd(unittest.TestCase):
    """End-to-end tests using full condition dicts on _chromatographic_match."""

    def test_biela2003_must_merge(self):
        """The user's supervisor's example: [210] Biela2003."""
        ca = {"analyte_polymer": "linear C4H9-PLA-OH",
              "critical_component": "linear C4H9-PLA-OH",
              "column_name": "Si-100 and Si-300",
              "mobile_phase_solvents": ["1,4-dioxane", "n-hexane"],
              "mobile_phase_ratio": "56.25/43.75", "temperature_celsius": "50"}
        cb = {"analyte_polymer": "poly(L-lactide)",
              "critical_component": "poly(L-lactide)",
              "column_name": "Si-100 and Si-300",
              "mobile_phase_solvents": ["1,4-dioxane", "n-hexane"],
              "mobile_phase_ratio": "56.25:43.75", "temperature_celsius": "50"}
        self.assertTrue(j._chromatographic_match(ca, cb),
            "[210] Biela2003 pair must be detected as duplicate")

    def test_mw_series_stays_separate(self):
        """[328] Bashir2006: PEG MW series must NOT be merged."""
        ca = {"analyte_polymer": "peg 2010", "column_name": "Symmetry 300",
              "mobile_phase_ratio": "96:4", "temperature_celsius": "30"}
        cb = {"analyte_polymer": "peg 6240", "column_name": "Symmetry 300",
              "mobile_phase_ratio": "96:4", "temperature_celsius": "30"}
        self.assertFalse(j._chromatographic_match(ca, cb),
            "PEG MW series must remain as separate records")

    def test_ring_vs_linear_ps_different_temp(self):
        """Ring-PS and Linear-PS at different temperatures — different conditions."""
        ca = {"analyte_polymer": "Ring-PS", "column_name": "Nucleosil C18",
              "mobile_phase_ratio": "85:15", "temperature_celsius": "17.3"}
        cb = {"analyte_polymer": "Linear-PS", "column_name": "Nucleosil C18",
              "mobile_phase_ratio": "85:15", "temperature_celsius": "14.8"}
        self.assertFalse(j._chromatographic_match(ca, cb),
            "Ring-PS vs Linear-PS at different temps must NOT match")

    def test_pib_endgroups_stay_separate(self):
        """[289] Banerjee2015: PIB end-group variants — different analytes."""
        ca = {"analyte_polymer": "pib-diol", "column_name": "YMC HPLC column",
              "mobile_phase_ratio": "80.5:19.5", "temperature_celsius": "35"}
        cb = {"analyte_polymer": "pib-diallyl", "column_name": "YMC HPLC column",
              "mobile_phase_ratio": "80.5:19.5", "temperature_celsius": "35"}
        self.assertFalse(j._chromatographic_match(ca, cb),
            "PIB end-group variants must remain as separate records")


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for cls in [TestAnalyteBaseFamilyMatch, TestChromatographicMatchEndToEnd]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✅ All 22 tests passed. Safe to run dedup post-processing.")
        sys.exit(0)
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} test(s) failed. Fix before proceeding.")
        sys.exit(1)
