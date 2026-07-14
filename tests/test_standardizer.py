import json
import pytest
from pipeline.standardizer import (
    _normalize_ratio_units,
    _normalize_ratio,
    _normalize_flow_rate,
    _normalize_pore_size,
    _normalize_temperature,
    _normalize_column_mode,
    _normalize_architecture,
    _normalize_year,
    standardize_condition
)

def norm_units(val):
    c = {"mobile_phase_ratio_units": val}
    _normalize_ratio_units(c)
    return c.get("mobile_phase_ratio_units")

def norm_ratio(val):
    c = {"mobile_phase_ratio": val}
    _normalize_ratio(c)
    return c.get("mobile_phase_ratio_components")

def norm_flow(val):
    c = {"flow_rate": val}
    _normalize_flow_rate(c)
    return c.get("flow_rate_ml_per_min")

def norm_pore(val):
    c = {"pore_size": val}
    _normalize_pore_size(c)
    return c.get("pore_size_angstrom")

def norm_temp(val):
    c = {"temperature_celsius": val}
    _normalize_temperature(c)
    return c.get("temperature_celsius")

def norm_mode(val):
    c = {"column_mode": val}
    _normalize_column_mode(c)
    return c.get("column_mode")

def norm_arch(val):
    c = {"architecture": val}
    _normalize_architecture(c)
    return c.get("architecture")

def norm_year(val):
    c = {"publication_year": val}
    _normalize_year(c)
    return c.get("publication_year")


def test_ratio_units():
    assert norm_units("wt-% methanol") == "w/w"
    assert norm_units("v/v (THF/water)") == "v/v"
    assert norm_units("% by volume") == "v/v"
    assert norm_units("vol.%") == "v/v"
    assert norm_units("wt.-%") == "w/w"
    assert norm_units("null") is None
    assert norm_units(None) is None


def test_ratio_components():
    assert norm_ratio("85:15") == [85.0, 15.0]
    assert norm_ratio("83/17") == [83.0, 17.0]
    assert norm_ratio("84.55") == [84.55, 15.45]
    assert norm_ratio("92") == [92.0, 8.0]
    assert norm_ratio("64-35-1") == [64.0, 35.0, 1.0]
    assert norm_ratio("60/40") == [60.0, 40.0]
    assert norm_ratio("69.12 wt-% acetonitrile") == [69.12, 30.88]


def test_flow_rate():
    assert norm_flow("0.5 mL min⁻¹") == 0.5
    assert norm_flow("0.5 ml/min") == 0.5
    assert norm_flow("0.5 mL min[-][1]") == 0.5
    assert norm_flow("1.0 mL/min") == 1.0
    assert norm_flow("null") is None
    assert norm_flow(None) is None


def test_pore_size():
    assert norm_pore("100 Å") == 100.0
    assert norm_pore("100 A") == 100.0
    assert norm_pore("100-Å") == 100.0
    assert norm_pore("100Å") == 100.0
    assert norm_pore("300A") == 300.0
    assert norm_pore("30 nm") == 300.0
    assert norm_pore("300") == 300.0
    assert norm_pore("null") is None


def test_temperature():
    assert norm_temp("25") == 25.0
    assert norm_temp("25.0") == 25.0
    assert norm_temp("null") is None
    assert norm_temp(None) is None
    
    cond = standardize_condition({"temperature_celsius": "15.0–35.0"})
    assert cond["temperature_celsius"] is None
    assert cond["temperature_min_celsius"] == 15.0
    assert cond["temperature_max_celsius"] == 35.0


def test_column_mode():
    assert norm_mode("Reverse Phase") == "Reversed Phase"
    assert norm_mode("reversed phase") == "Reversed Phase"
    assert norm_mode("Reversed Phase") == "Reversed Phase"
    assert norm_mode("Normal Phase") == "Normal Phase"
    assert norm_mode(None) is None


def test_architecture():
    assert norm_arch("linear") == "linear"
    assert norm_arch("Linear") == "linear"
    assert norm_arch("Linear homopolymer") == "linear"
    assert norm_arch("ring") == "cyclic"
    assert norm_arch("triblock copolymer") == "triblock"
    assert norm_arch("AB block copolymer") == "diblock"
    assert norm_arch("difunktionell") is None


def test_publication_year():
    assert norm_year("1995") == 1995
    assert norm_year("null") is None
    assert norm_year(None) is None


def test_integration():
    raw_cond = {
        "mobile_phase_ratio_units": "wt-% methanol",
        "mobile_phase_ratio": "85:15",
        "flow_rate": "0.5 mL/min",
        "pore_size": "30 nm",
        "temperature_celsius": "25.0",
        "column_mode": "Reverse Phase",
        "architecture": "ring",
        "publication_year": "2020",
        "mobile_phase_solvents": ["water with 0.1% formic acid", "methanol"],
        "aqueous_parameters": {}
    }
    std = standardize_condition(raw_cond)
    assert std["mobile_phase_ratio_units"] == "w/w"
    assert std["mobile_phase_ratio_units_raw"] == "wt-% methanol"
    assert std["mobile_phase_ratio_components"] == [85.0, 15.0]
    assert std["flow_rate_ml_per_min"] == 0.5
    assert std["pore_size_angstrom"] == 300.0
    assert std["temperature_celsius"] == 25.0
    assert std["column_mode"] == "Reversed Phase"
    assert std["architecture"] == "cyclic"
    assert std["publication_year"] == 2020
    assert std["mobile_phase_solvents"] == ["water", "methanol"]
    assert std["aqueous_parameters"]["pH_modifier"] == "0.1% formic acid"
