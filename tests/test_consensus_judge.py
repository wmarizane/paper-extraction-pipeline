"""Focused tests for solvent and ratio matching in the consensus judge."""

import sys
import types
import unittest
from unittest.mock import patch


vllm_stub = types.ModuleType("vllm")


class _LLM:
    pass


class _SamplingParams:
    def __init__(self, *args, **kwargs):
        pass


class _StructuredOutputsParams:
    def __init__(self, *args, **kwargs):
        pass


vllm_stub.LLM = _LLM
vllm_stub.SamplingParams = _SamplingParams
sampling_stub = types.ModuleType("vllm.sampling_params")
sampling_stub.StructuredOutputsParams = _StructuredOutputsParams

config_stub = types.ModuleType("config")
settings_stub = types.ModuleType("config.settings")
settings_stub.settings = object()
registry_stub = types.ModuleType("config.model_registry")
registry_stub.get_model_config = lambda model_name: None
config_stub.settings = settings_stub

extractor_stub = types.ModuleType("pipeline.llm_extractor")
extractor_stub.EXTRACTION_SCHEMA = {"type": "object"}

with patch.dict(
    sys.modules,
    {
        "vllm": vllm_stub,
        "vllm.sampling_params": sampling_stub,
        "config": config_stub,
        "config.settings": settings_stub,
        "config.model_registry": registry_stub,
        "pipeline.llm_extractor": extractor_stub,
    },
):
    from pipeline.consensus_judge import ConsensusJudge


def _condition(solvents, ratio, units="v/v"):
    return {
        "analyte_polymer": "PEG",
        "critical_component": "PEG",
        "column_name": "C18",
        "mobile_phase_solvents": solvents,
        "mobile_phase_ratio": ratio,
        "mobile_phase_ratio_units": units,
        "temperature_celsius": "25",
    }


class ConsensusMobilePhaseTests(unittest.TestCase):
    def test_reversed_binary_order_matches_when_ratios_follow_solvents(self):
        first = _condition(["THF", "Water"], "70:30")
        second = _condition(["Water", "Tetrahydrofuran"], "30:70")

        self.assertTrue(ConsensusJudge._chromatographic_match(first, second))

    def test_reversed_binary_order_does_not_match_with_unreversed_ratio(self):
        first = _condition(["THF", "Water"], "70:30")
        second = _condition(["Water", "Tetrahydrofuran"], "70:30")

        self.assertFalse(ConsensusJudge._chromatographic_match(first, second))

    def test_same_order_does_not_match_when_composition_differs(self):
        first = _condition(["THF", "Water"], "70:30")
        second = _condition(["Tetrahydrofuran", "Water"], "60:40")

        self.assertFalse(ConsensusJudge._chromatographic_match(first, second))

    def test_ternary_permutation_matches_by_solvent_composition(self):
        first = _condition(["THF", "Water", "Methanol"], "50:30:20")
        second = _condition(["Methanol", "Tetrahydrofuran", "Water"], "20:50:30")

        self.assertTrue(ConsensusJudge._chromatographic_match(first, second))

    def test_dash_separated_ternary_relative_ratio_preserves_positions(self):
        first = _condition(["THF", "Water", "Methanol"], "1-2-3")
        second = _condition(["Methanol", "Tetrahydrofuran", "Water"], "3-1-2")

        self.assertTrue(ConsensusJudge._chromatographic_match(first, second))

    def test_relative_ratios_are_compared_as_equivalent_compositions(self):
        first = _condition(["THF", "Water"], "1:3")
        second = _condition(["Water", "Tetrahydrofuran"], "75:25")

        self.assertTrue(ConsensusJudge._chromatographic_match(first, second))

    def test_different_solvent_identity_remains_a_hard_mismatch(self):
        first = _condition(["THF", "Water"], "70:30")
        second = _condition(["Hexane", "Tetrahydrofuran"], "30:70")

        self.assertFalse(ConsensusJudge._chromatographic_match(first, second))

    def test_different_ratio_units_do_not_match(self):
        first = _condition(["THF", "Water"], "70:30", units="v/v")
        second = _condition(["Water", "Tetrahydrofuran"], "30:70", units="w/w")

        self.assertFalse(ConsensusJudge._chromatographic_match(first, second))

    def test_reordered_unparseable_ratios_do_not_match(self):
        first = _condition(["THF", "Water"], "THF rich")
        second = _condition(["Water", "Tetrahydrofuran"], "THF rich")

        self.assertFalse(ConsensusJudge._chromatographic_match(first, second))

    def test_reordered_solvents_with_one_missing_ratio_do_not_match(self):
        first = _condition(["THF", "Water"], "70:30")
        second = _condition(["Water", "Tetrahydrofuran"], None)

        self.assertFalse(ConsensusJudge._chromatographic_match(first, second))


if __name__ == "__main__":
    unittest.main()
