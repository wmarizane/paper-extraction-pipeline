import re

def test_guard():
    with open("pipeline/consensus_judge.py", "r", encoding="utf-8") as f:
        code = f.read()

    # Strip problematic imports
    code = re.sub(r'from vllm\b.*', '', code)
    code = re.sub(r'from config\b.*', '', code)
    code = re.sub(r'from pipeline\.llm_extractor import EXTRACTION_SCHEMA', 'EXTRACTION_SCHEMA = {}', code)
    
    namespace = {}
    exec(code, namespace)
    ConsensusJudge = namespace['ConsensusJudge']

    # 1. Should NOT match (the Ziebarth bug):
    c1_a = {"analyte_polymer": "Ls-PS", "critical_component": "Ls-PS", "column_name": "Nucleosil C18", "mobile_phase_solvents": ["CH2Cl2", "CH3CN"], "mobile_phase_ratio": "58/42", "temperature_celsius": "14.8"}
    c1_b = {"analyte_polymer": "Ring-PS", "critical_component": "Ring-PS", "column_name": "Nucleosil C18", "mobile_phase_solvents": ["CH2Cl2", "CH3CN"], "mobile_phase_ratio": "58/42", "temperature_celsius": "17.3"}
    r1 = ConsensusJudge._chromatographic_match(c1_a, c1_b)
    assert r1 is False, f"Test 1 failed! Expected False, got {r1}"

    # 2. Should still match (same experiment, different naming):
    c2_a = {"analyte_polymer": "Ring-PS", "critical_component": "Ring-PS", "column_name": "Nucleosil C18", "mobile_phase_solvents": ["CH2Cl2", "CH3CN"], "mobile_phase_ratio": "58/42", "temperature_celsius": "17.3"}
    c2_b = {"analyte_polymer": "polystyrene", "critical_component": "ring", "column_name": "Nucleosil C18", "mobile_phase_solvents": ["CH2Cl2", "CH3CN"], "mobile_phase_ratio": "58/42", "temperature_celsius": "17.3"}
    r2 = ConsensusJudge._chromatographic_match(c2_a, c2_b)
    assert r2 is True, f"Test 2 failed! Expected True, got {r2}"

    # 3. Should still match (same analyte name, different temp — could be text duplication):
    c3_a = {"analyte_polymer": "polystyrene", "critical_component": "linear", "column_name": "Nucleosil C18", "mobile_phase_solvents": ["CH2Cl2", "CH3CN"], "mobile_phase_ratio": "58/42", "temperature_celsius": "14.8"}
    c3_b = {"analyte_polymer": "polystyrene", "critical_component": "linear", "column_name": "Nucleosil C18", "mobile_phase_solvents": ["CH2Cl2", "CH3CN"], "mobile_phase_ratio": "58/42", "temperature_celsius": "19"}
    r3 = ConsensusJudge._chromatographic_match(c3_a, c3_b)
    assert r3 is True, f"Test 3 failed! Expected True, got {r3}"

    # 4. Should NOT match (different polymers entirely):
    c4_a = {"analyte_polymer": "PEG", "critical_component": "EO", "column_name": "Nucleosil C18", "mobile_phase_ratio": "70/30", "temperature_celsius": "25"}
    c4_b = {"analyte_polymer": "PPO", "critical_component": "PO", "column_name": "Nucleosil C18", "mobile_phase_ratio": "70/30", "temperature_celsius": "25"}
    r4 = ConsensusJudge._chromatographic_match(c4_a, c4_b)
    assert r4 is False, f"Test 4 failed! Expected False, got {r4}"

    print("ALL TESTS PASSED: Guard condition works perfectly!")

if __name__ == "__main__":
    test_guard()
