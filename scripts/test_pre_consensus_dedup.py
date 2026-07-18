#!/usr/bin/env python3
"""Test suite for the pre-consensus per-model dedup layer.

Covers the regression cases from Dr. Wang's 7-7 feedback plus the
negative cases that must NEVER merge. Run from project root:

  python3 scripts/test_pre_consensus_dedup.py
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# vllm is cluster-only; mock it so consensus_judge imports locally
sys.modules.setdefault('vllm', MagicMock())
sys.modules.setdefault('vllm.sampling_params', MagicMock())
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pre_consensus_dedup import (
    _conditions_match, _cc_match, _merge, dedup_model_conditions,
)


class TestMustMerge(unittest.TestCase):
    """Known duplicate pairs that MUST be detected."""

    def test_hiller2011_column_phrasing(self):
        """Same experiment, different column descriptions."""
        c1 = {
            'analyte_polymer': '1,4-polyisoprene',
            'critical_component': '1,4-isoprene units',
            'column_name': 'Nucleosil C18',
            'stationary_phase_chemistry': 'C18',
            'mobile_phase_solvents': ['butanone', 'cyclohexane'],
            'mobile_phase_ratio': '92/8',
            'temperature_celsius': '29',
        }
        c2 = {
            'analyte_polymer': '1,4-polyisoprene',
            'critical_component': '1,4-polyisoprene',
            'column_name': 'three RP columns of C18 (100-5, 300-5, 1000-7)',
            'stationary_phase_chemistry': 'C18',
            'mobile_phase_solvents': ['butanone', 'cyclohexane'],
            'mobile_phase_ratio': '92:8',
            'temperature_celsius': '29',
        }
        self.assertTrue(_conditions_match(c1, c2), 'Hiller2011 must match')
        result = dedup_model_conditions([c1, c2])
        self.assertEqual(len(result), 1, 'Hiller2011 must merge to 1')

    def test_kazanskii2000_polymer_naming(self):
        """Same experiment, different polymer naming (PEG methyl ether = MPEG)."""
        c3 = {
            'analyte_polymer': 'poly(ethylene glycol) methyl ether',
            'critical_component': 'poly(ethylene glycol) methyl ether',
            'column_name': 'Lichrosorb-10 RP-18',
            'stationary_phase_chemistry': 'C18',
            'mobile_phase_solvents': ['CH3CN', 'H2O'],
            'mobile_phase_ratio': '60:40',
            'temperature_celsius': None,
        }
        c4 = {
            'analyte_polymer': 'PEG monomethyl ethers (MPEG)',
            'critical_component': 'PEG monomethyl ethers',
            'column_name': 'Lichrosorb-10 RP-18',
            'stationary_phase_chemistry': 'C18',
            'mobile_phase_solvents': ['CH3CN', 'H2O'],
            'mobile_phase_ratio': '60:40',
            'temperature_celsius': None,
        }
        self.assertTrue(_conditions_match(c3, c4), 'Kazanskii2000 must match')

    def test_malik2012_analyte_phrasing(self):
        """Same experiment, different analyte phrasing."""
        c5 = {
            'analyte_polymer': 'PS, PEO',
            'critical_component': 'PS',
            'column_name': 'Symmetry 300',
            'stationary_phase_chemistry': 'C18',
            'mobile_phase_solvents': ['DMF', 'THF'],
            'mobile_phase_ratio': '18-82',
            'temperature_celsius': '30',
        }
        c6 = {
            'analyte_polymer': 'Polystyrene (PS), PS-b-PEO block copolymers',
            'critical_component': 'Polystyrene (PS) block',
            'column_name': 'Symmetry 300',
            'stationary_phase_chemistry': 'C18',
            'mobile_phase_solvents': ['DMF', 'THF'],
            'mobile_phase_ratio': '18-82',
            'temperature_celsius': '30',
        }
        self.assertTrue(_conditions_match(c5, c6), 'Malik2012 must match')

    def test_radke2005_cc_variants(self):
        """Dr. Wang item 3: three CC phrasings of the same PLA condition."""
        base = {
            'analyte_polymer': 'linear poly(lactide)',
            'column_name': 'Nucleosil',
            'stationary_phase_chemistry': 'Bare silica',
            'mobile_phase_solvents': ['1,4-dioxane', 'n-hexane'],
            'mobile_phase_ratio': '57.1/42.9',
            'temperature_celsius': '50',
        }
        variants = ['linear poly(lactides)', 'poly(lactide) repeating units',
                    'Linear poly(lactide)']
        conds = [dict(base, critical_component=v) for v in variants]
        self.assertTrue(_cc_match(variants[0], variants[1]))
        self.assertTrue(_cc_match(variants[0], variants[2]))
        self.assertTrue(_cc_match(variants[1], variants[2]))
        self.assertEqual(len(dedup_model_conditions(conds)), 1,
                         'Radke2005 CC variants must merge to 1')

    def test_null_tolerance(self):
        """Missing fields must not block a match."""
        c1 = {'critical_component': 'PEO', 'column_name': 'Nucleosil C18',
              'stationary_phase_chemistry': None, 'mobile_phase_solvents': None,
              'mobile_phase_ratio': '92/8', 'temperature_celsius': None}
        c2 = {'critical_component': 'Polyethylene oxide (PEO) block',
              'column_name': 'Nucleosil C18', 'stationary_phase_chemistry': 'C18',
              'mobile_phase_solvents': ['acetonitrile', 'water'],
              'mobile_phase_ratio': '92 vol%', 'temperature_celsius': '25'}
        self.assertTrue(_conditions_match(c1, c2), 'Nulls must auto-match')

    def test_ratio_unit_variants(self):
        """'92/8' = '92:8' = '92 vol%' (leading number)."""
        c1 = {'mobile_phase_ratio': '92/8'}
        c2 = {'mobile_phase_ratio': '92 vol%'}
        self.assertTrue(_conditions_match(c1, c2))


class TestMustNotMerge(unittest.TestCase):
    """Distinct records that must NEVER merge."""

    def test_im2008_different_temps(self):
        """Different columns AND temps (33.3 vs 30.4, diff > 2.0)."""
        c7 = {
            'analyte_polymer': 'linear PS', 'critical_component': 'linear PS',
            'column_name': 'Kromasil C18', 'stationary_phase_chemistry': 'C18',
            'mobile_phase_solvents': ['acetonitrile', 'dichloromethane'],
            'mobile_phase_ratio': '57/43', 'temperature_celsius': '33.3',
        }
        c8 = dict(c7, column_name='Nucleosil C18', temperature_celsius='30.4')
        self.assertFalse(_conditions_match(c7, c8),
                         'Im2008 must NOT match (temps differ by 2.9)')

    def test_banerjee2015_endgroup_variants(self):
        """PIB-diol vs PIB-diallyl: same CC, same chromatography — the
        analyte guard MUST keep them separate (END-GROUPS rule)."""
        c9 = {
            'analyte_polymer': 'PIB-diol', 'critical_component': 'polyisobutylene',
            'column_name': 'YMC C18', 'stationary_phase_chemistry': 'C18',
            'mobile_phase_solvents': ['THF', 'methanol'],
            'mobile_phase_ratio': '80.5/19.5', 'temperature_celsius': None,
        }
        c10 = dict(c9, analyte_polymer='PIB-diallyl')
        self.assertFalse(_conditions_match(c9, c10),
                         'PIB end-group variants must NOT merge')
        self.assertEqual(len(dedup_model_conditions([c9, c10])), 2)

    def test_mw_series_stays_separate(self):
        """peg 2010 vs peg 6240 (MW series) must NOT merge."""
        c1 = {'analyte_polymer': 'peg 2010', 'critical_component': 'PEG',
              'column_name': 'Symmetry 300', 'mobile_phase_ratio': '96:4',
              'temperature_celsius': '30'}
        c2 = dict(c1, analyte_polymer='peg 6240')
        self.assertFalse(_conditions_match(c1, c2), 'MW series must NOT merge')

    def test_architecture_variants_stay_separate(self):
        """Ring-PS vs Ls-PS (architecture prefixes) must NOT merge."""
        c1 = {'analyte_polymer': 'Ring-PS', 'critical_component': 'polystyrene',
              'column_name': 'Nucleosil C18', 'mobile_phase_ratio': '85:15',
              'temperature_celsius': '17.3'}
        c2 = dict(c1, analyte_polymer='Ls-PS', temperature_celsius='16.5')
        self.assertFalse(_conditions_match(c1, c2),
                         'Different architectures must NOT merge')

    def test_different_ratio(self):
        """Clearly different mobile phase ratios must NOT merge."""
        c1 = {'critical_component': 'PS', 'mobile_phase_ratio': '92/8'}
        c2 = {'critical_component': 'PS', 'mobile_phase_ratio': '57/43'}
        self.assertFalse(_conditions_match(c1, c2))

    def test_different_polymers(self):
        """PS vs PMMA critical components must NOT merge."""
        c1 = {'critical_component': 'polystyrene', 'mobile_phase_ratio': '92/8'}
        c2 = {'critical_component': 'poly(methyl methacrylate)',
              'mobile_phase_ratio': '92/8'}
        self.assertFalse(_conditions_match(c1, c2))

    def test_falkenhagen2005_named_endgroups(self):
        """PEO with hydroxyl/methoxy/vinyl/isopropenyl end groups:
        4 distinct records (END-GROUPS rule), same chromatography."""
        base = {'critical_component': 'PEO backbone',
                'column_name': 'GROM ODS 120 and Zorbax SBC18',
                'stationary_phase_chemistry': 'C18',
                'mobile_phase_ratio': '84:16', 'temperature_celsius': '45'}
        ends = ['hydroxyl', 'methoxy', 'vinyl', 'isopropenyl']
        conds = [dict(base, analyte_polymer=f'PEO with {e} end groups')
                 for e in ends]
        self.assertEqual(len(dedup_model_conditions(conds)), 4,
                         'Named end-group variants must all stay separate')

    def test_malik2009_block_architectures(self):
        """EO-PO diblock vs EO-PO-EO triblock vs PO-EO-PO triblock:
        different block architectures must NOT merge."""
        base = {'critical_component': 'EO', 'column_name': 'Chromolith RP18',
                'mobile_phase_ratio': '34 wt% ACN', 'temperature_celsius': '25.0'}
        arch = ['EO–PO diblock copolymer', 'EO–PO–EO triblock copolymer',
                'PO–EO–PO triblock copolymer']
        conds = [dict(base, analyte_polymer=a) for a in arch]
        self.assertEqual(len(dedup_model_conditions(conds)), 3,
                         'Different block architectures must stay separate')

    def test_rollet2012_homo_vs_block_copolymers(self):
        """PEO vs PEO-b-PS vs PS-b-PEO-b-PS: one critical condition,
        three properly-split analyte records — must all stay separate
        (MULTIPLE ANALYTES rule)."""
        base = {'critical_component': 'poly(ethylene oxide)',
                'column_name': 'Kromasil C18',
                'mobile_phase_ratio': '58.05% chloroform',
                'temperature_celsius': '30'}
        analytes = ['poly(ethylene oxide)',
                    'poly(ethylene oxide)-b-polystyrene',
                    'polystyrene-b-poly(ethylene oxide)-b-polystyrene']
        conds = [dict(base, analyte_polymer=a) for a in analytes]
        self.assertEqual(len(dedup_model_conditions(conds)), 3,
                         'Homo/diblock/triblock records must stay separate')

    def test_block_signature_canonicalization_still_merges(self):
        """'1,4-polyisoprene' vs 'polyisoprene (1,4-PI)' analytes must
        still be mergeable (signature tokens canonicalize identically)."""
        c1 = {'analyte_polymer': '1,4-polyisoprene',
              'critical_component': '1,4-polyisoprene',
              'mobile_phase_ratio': '92/8', 'temperature_celsius': '29'}
        c2 = {'analyte_polymer': 'polyisoprene (1,4-PI)',
              'critical_component': '1,4-polyisoprene',
              'mobile_phase_ratio': '92:8', 'temperature_celsius': '29'}
        self.assertTrue(_conditions_match(c1, c2),
                        'Canonically identical analytes must still merge')


class TestMerge(unittest.TestCase):
    def test_merge_prefers_nonnull_and_detail(self):
        c1 = {'column_name': 'Nucleosil C18', 'pore_size': None,
              'field_evidence': {'column_name': 'a Nucleosil C18 column was used',
                                 'pore_size': None}}
        c2 = {'column_name': 'Nucleosil', 'pore_size': '100 A',
              'field_evidence': {'column_name': 'Nucleosil',
                                 'pore_size': 'pore size of 100 A'}}
        m = _merge(c1, c2)
        self.assertEqual(m['column_name'], 'Nucleosil C18')  # longer wins
        self.assertEqual(m['pore_size'], '100 A')            # non-null wins
        self.assertEqual(m['field_evidence']['pore_size'], 'pore size of 100 A')
        self.assertEqual(m['field_evidence']['column_name'],
                         'a Nucleosil C18 column was used')


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for cls in [TestMustMerge, TestMustNotMerge, TestMerge]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
