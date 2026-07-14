import csv
import json
import hashlib
import tempfile
from pathlib import Path
from pipeline.standardizer import (
    _POLYMER_ALIAS_CONFLICTS,
    _POLYMER_CANONICAL_TO_ALIASES,
    _SOLVENT_CANONICAL_TO_ALIASES,
    _normalize_ratio_units,
    _normalize_ratio,
    _normalize_flow_rate,
    _normalize_pore_size,
    _normalize_temperature,
    _normalize_column_mode,
    _normalize_architecture,
    _normalize_polymer_key,
    _normalize_polymer_name,
    _normalize_solvent_name,
    _normalize_solvents,
    _normalize_year,
    canonicalize_solvent_list,
    standardize_condition,
    standardize_file,
)
from pipeline.standardized_csv_exporter import (
    POLYCRIT_FIELDNAMES,
    POLYCRIT_FIELDNAMES_REVIEW,
    POLYCRIT_REVIEW_AUDIT_FIELDNAMES,
    _derive_polymer_columns,
    _format_polycrit_solvents,
    _load_polymer_reference_dictionary,
    _pore_size_for_export,
    _ratio_columns,
    export_folder_to_csv,
)
from pipeline.ground_truth_evaluator import load_polycrit_csv

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
    assert norm_units("w/v") is None
    assert norm_units("weight/volume") is None
    assert norm_units("wt./vol.") is None
    assert norm_units("%") is None
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
    assert norm_ratio("1:1") == [50.0, 50.0]
    assert norm_ratio("3/1") == [75.0, 25.0]
    assert norm_ratio("1:1:1") == [33.333333, 33.333333, 33.333333]
    assert norm_ratio("1-1-1") == [33.333333, 33.333333, 33.333333]
    assert norm_ratio("1, 1, 1") == [33.333333, 33.333333, 33.333333]
    assert norm_ratio("64, 35, 1") == [64.0, 35.0, 1.0]


def test_missing_ratio_is_not_invented_for_pure_solvent():
    cond = standardize_condition(
        {
            "mobile_phase_solvents": ["Water"],
            "mobile_phase_ratio": None,
            "mobile_phase_ratio_units": None,
        }
    )
    assert cond["mobile_phase_ratio_components"] is None
    assert cond["mobile_phase_ratio_units"] is None
    assert _ratio_columns(
        {
            "mobile_phase_ratio_components": [10.0, 90.0],
            "mobile_phase_ratio_units_raw": "w/v",
        }
    ) == ("", "")
    assert _ratio_columns(
        {
            "mobile_phase_ratio_components": [10.0, 90.0],
            "mobile_phase_ratio_units_raw": "wt./vol.",
        }
    ) == ("", "")


def test_flow_rate():
    assert norm_flow("0.5 mL min⁻¹") == 0.5
    assert norm_flow("0.5 ml/min") == 0.5
    assert norm_flow("0.5 mL min[-][1]") == 0.5
    assert norm_flow("1.0 mL/min") == 1.0
    assert norm_flow("500 uL/min") == 0.5
    assert norm_flow("60 mL/h") == 1.0
    assert norm_flow("60 mL h\u22121") == 1.0
    assert norm_flow("60 mL hr\u22121") == 1.0
    assert norm_flow("60 mL hour\u22121") == 1.0
    assert norm_flow("1 mL sec\u22121") == 60.0
    assert norm_flow("1 mL second\u22121") == 60.0
    assert norm_flow("null") is None
    assert norm_flow(None) is None

    flow_range = standardize_condition({"flow_rate": "500-1000 uL/min"})
    assert flow_range["flow_rate_ml_per_min"] is None
    assert flow_range["flow_rate_min_ml_per_min"] == 0.5
    assert flow_range["flow_rate_max_ml_per_min"] == 1.0


def test_pore_size():
    assert norm_pore("100 Å") == 100.0
    assert norm_pore("100 A") == 100.0
    assert norm_pore("100-Å") == 100.0
    assert norm_pore("100Å") == 100.0
    assert norm_pore("300A") == 300.0
    assert norm_pore("30 nm") == 300.0
    assert norm_pore("300") == 300.0
    assert norm_pore("null") is None

    mixed = standardize_condition({"pore_size": "100 A and 30 nm"})
    assert mixed["pore_size_angstrom"] == [100.0, 300.0]

    pore_range = standardize_condition({"pore_size": "30-50 nm"})
    assert pore_range["pore_size_angstrom"] is None
    assert pore_range["pore_size_min_angstrom"] == 300.0
    assert pore_range["pore_size_max_angstrom"] == 500.0
    assert _pore_size_for_export(pore_range) == "300-500"


def test_temperature():
    assert norm_temp("25") == 25.0
    assert norm_temp("25.0") == 25.0
    assert norm_temp(0) == 0.0
    assert norm_temp("null") is None
    assert norm_temp(None) is None
    assert norm_temp("-20 C") == -20.0

    cond = standardize_condition({"temperature_celsius": "15.0\u201335.0"})
    assert cond["temperature_celsius"] is None
    assert cond["temperature_min_celsius"] == 15.0
    assert cond["temperature_max_celsius"] == 35.0
    cond_twice = standardize_condition(cond)
    assert cond_twice["temperature_min_celsius"] == 15.0
    assert cond_twice["temperature_max_celsius"] == 35.0

    negative_range = standardize_condition({"temperature_celsius": "-20 to -10 C"})
    assert negative_range["temperature_min_celsius"] == -20.0
    assert negative_range["temperature_max_celsius"] == -10.0


def test_column_mode():
    assert norm_mode("Reverse Phase") == "Reversed Phase"
    assert norm_mode("reversed phase") == "Reversed Phase"
    assert norm_mode("Reversed Phase") == "Reversed Phase"
    assert norm_mode("Normal Phase") == "Normal Phase"
    assert norm_mode("RP") == "Reversed Phase"
    assert norm_mode("Ion Exchange") == "Ion Exchange"
    assert norm_mode(None) is None


def test_architecture():
    assert norm_arch("linear") == "linear"
    assert norm_arch("Linear") == "linear"
    assert norm_arch("Linear homopolymer") == "linear"
    assert norm_arch("ring") == "cyclic"
    assert norm_arch("triblock copolymer") == "triblock"
    assert norm_arch("AB block copolymer") == "diblock"
    assert norm_arch("difunktionell") is None

    unrecognized = standardize_condition({"architecture": "telechelic"})
    assert unrecognized["architecture"] == "telechelic"
    assert unrecognized["architecture_raw"] == "telechelic"


def test_publication_year():
    assert norm_year("1995") == 1995
    assert norm_year("null") is None
    assert norm_year(None) is None


def test_polycrit_polymer_registry():
    assert len(_POLYMER_CANONICAL_TO_ALIASES) == 69
    assert sum(len(aliases) for aliases in _POLYMER_CANONICAL_TO_ALIASES.values()) == 100
    snapshot = json.dumps(
        _POLYMER_CANONICAL_TO_ALIASES,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert hashlib.sha256(snapshot.encode()).hexdigest() == (
        "b2868abe25fb5c75b2762d25adc33b4af7f806e41bf9937e04f62a3e58a275cf"
    )
    assert _POLYMER_ALIAS_CONFLICTS == {
        "dps": ("Deuterated Polystyrene", "Polystyrene"),
        "epoxyresin": ("Bisphenol A", "Polyepichlorohydrin", "Polyepoxides"),
        "pc": ("Aliphatic Polycarbonate", "Polycarbonate"),
    }

    for canonical, aliases in _POLYMER_CANONICAL_TO_ALIASES.items():
        assert _normalize_polymer_name(canonical) == canonical
        for alias in aliases:
            alias_key = _normalize_polymer_key(alias)
            if alias_key == "dps":
                expected = "Deuterated Polystyrene"
            elif alias_key == "pc":
                expected = alias
            elif alias_key == "epoxyresin":
                expected = alias
            else:
                expected = canonical
            assert _normalize_polymer_name(alias) == expected


def test_polycrit_polymer_compatibility_aliases():
    assert _normalize_polymer_name("PEO") == "Poly(ethylene glycol)"
    assert _normalize_polymer_name("Poly(ethylene oxide)") == "Poly(ethylene glycol)"
    assert _normalize_polymer_name("PPO") == "Poly(propylene glycol)"
    assert _normalize_polymer_name("Poly(propylene oxide)") == "Poly(propylene glycol)"
    assert _normalize_polymer_name("linear PMMA") == "Poly(methyl methacrylate)"
    assert _normalize_polymer_name("AA-1,2ED") == "Poly[(adipic acid)-co-(1,2-ethanediol)]"
    assert _normalize_polymer_name("AA-1,6HD") == "Poly[(adipic acid)-co-(1,6-hexanediol)]"
    assert _normalize_polymer_name("PC") == "PC"
    assert _normalize_polymer_name("Epoxy Resin") == "Epoxy Resin"

    resolved = standardize_condition({"analyte_polymer": "dPS"})
    assert resolved["analyte_polymer"] == "Deuterated Polystyrene"
    assert resolved["analyte_polymer_standardization_status"] == "resolved_ambiguous_alias"
    assert set(resolved["analyte_polymer_canonical_candidates"]) == {
        "Deuterated Polystyrene",
        "Polystyrene",
    }

    unresolved = standardize_condition({"analyte_polymer": "Epoxy Resin"})
    assert unresolved["analyte_polymer"] == "Epoxy Resin"
    assert unresolved["analyte_polymer_standardization_status"] == "ambiguous_alias"

    decorated = standardize_condition({"analyte_polymer": "linear PC"})
    assert decorated["analyte_polymer"] == "linear PC"
    assert decorated["analyte_polymer_standardization_status"] == "ambiguous_alias"
    assert set(decorated["analyte_polymer_canonical_candidates"]) == {
        "Aliphatic Polycarbonate",
        "Polycarbonate",
    }


def test_exporter_uses_shared_polymer_resolution():
    alias_map, aliases, display = _load_polymer_reference_dictionary()
    cases = {
        "dPS": ("Deuterated Polystyrene", "dPS"),
        "PC": ("PC", ""),
        "PEO": ("Poly(ethylene glycol)", ""),
        "PEG": ("Poly(ethylene glycol)", "PEG"),
        "Poly(ethylene glycol)": ("Poly(ethylene glycol)", ""),
        "AA-1,3BD": ("Poly[(adipic acid)-co-(1,3-butanediol)]", "AA-1,3BD"),
        "ACR": ("Poly(acrylic acid)", "ACR"),
        "Poly(L-lactide)": ("Poly(L-lactide)", ""),
        "Epoxy Resin": ("Epoxy Resin", ""),
    }
    for raw, expected in cases.items():
        _, polymer, alternate = _derive_polymer_columns(
            raw,
            None,
            None,
            alias_map,
            aliases,
            display,
        )
        assert (polymer, alternate) == expected


def test_polycrit_solvent_registry_and_observed_aliases():
    assert len(_SOLVENT_CANONICAL_TO_ALIASES) == 49
    snapshot = json.dumps(
        sorted(_SOLVENT_CANONICAL_TO_ALIASES),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert hashlib.sha256(snapshot.encode()).hexdigest() == (
        "66d0a1f65a4b359e8539271149d4445ab30744b59f6be2d6101f2a9341185493"
    )
    for canonical, aliases in _SOLVENT_CANONICAL_TO_ALIASES.items():
        assert _normalize_solvent_name(canonical) == canonical
        for alias in aliases:
            assert _normalize_solvent_name(alias) == canonical

    observed = {
        "ACN": "Acetonitrile",
        "CH3CN": "Acetonitrile",
        "THF": "Tetrahydrofuran",
        "CH2Cl2": "Dichloromethane",
        "CCl4": "Tetrachloromethane",
        "CHCl3": "Chloroform",
        "DMF": "Dimethylformamide",
        "MeOH": "Methanol",
        "H2O": "Water",
        "n-hexane": "Hexane",
        "butanone": "Methyl Ethyl Ketone",
        "Methylethylketon": "Methyl Ethyl Ketone",
        "Ethylacetat": "Ethyl Acetate",
        "tetrahydrofurane": "Tetrahydrofuran",
        "ODCB": "1,2-Dichlorobenzene",
        "2E1H": "2-Ethyl-1-Hexanol",
        "TEA": "Triethylamine",
    }
    for raw, expected in observed.items():
        assert _normalize_solvent_name(raw) == expected

    assert _normalize_solvent_name("TCB") == "TCB"
    assert _normalize_solvent_name("Dioxane") == "Dioxane"
    assert _normalize_solvent_name("1,4-Dioxane") == "1,4-Dioxane"

    ambiguous = standardize_condition({"mobile_phase_solvents": ["TCB", "Decane"]})
    assert ambiguous["mobile_phase_solvents"] == ["TCB", "Decane"]
    assert ambiguous["mobile_phase_solvent_ambiguities"] == [
        {
            "raw": "TCB",
            "canonical_candidates": [
                "1,2,4-Trichlorobenzene",
                "Trichlorobenzene",
            ],
        }
    ]


def test_exact_polycrit_solvent_token_snapshot():
    exact_tokens = set(_SOLVENT_CANONICAL_TO_ALIASES)
    exact_tokens.remove("1,2-Dichloroethane")
    exact_tokens.remove("1-Propanol")
    exact_tokens.update(
        {
            "1,2-Dichloroethane (near crit)",
            "1-Propanol (near crit)",
            "2-Propanol (near crit)",
            "Acetonitrile (near crit)",
            "Ethanol (near crit)",
            "Heptane (near crit)",
            "Methanol (near crit)",
            "Tetrahydrofuran (near crit)",
            "Dimethyl Formamide",
        }
    )
    assert len(exact_tokens) == 56
    for token in exact_tokens:
        condition = {"mobile_phase_solvents": [token]}
        _normalize_solvents(condition)
        rendered = _format_polycrit_solvents(
            condition["mobile_phase_solvents"],
            condition["mobile_phase_solvent_qualifiers"],
        )
        expected = "Dimethylformamide" if token == "Dimethyl Formamide" else token
        assert rendered == expected


def test_solvent_list_parsing_preserves_order_and_locants():
    assert canonicalize_solvent_list("1,4-Dioxane, Hexane") == [
        "1,4-Dioxane",
        "Hexane",
    ]
    assert canonicalize_solvent_list("1,4-Dioxane,Hexane") == [
        "1,4-Dioxane",
        "Hexane",
    ]
    assert canonicalize_solvent_list("CO2,1,4-Dioxane") == [
        "Carbon Dioxide",
        "1,4-Dioxane",
    ]
    assert canonicalize_solvent_list("CCl4,1,4-Dioxane") == [
        "Tetrachloromethane",
        "1,4-Dioxane",
    ]
    assert canonicalize_solvent_list("N,N-Dimethylformamide,Acetonitrile") == [
        "Dimethylformamide",
        "Acetonitrile",
    ]
    assert canonicalize_solvent_list(["CH2Cl2/CH3CN", "n-hexane"]) == [
        "Dichloromethane",
        "Acetonitrile",
        "Hexane",
    ]
    assert canonicalize_solvent_list("water with methanol") == ["Water", "Methanol"]


def test_solvent_raw_values_qualifiers_and_idempotence():
    raw = {
        "mobile_phase_solvents": ["ACN (near critical)", "THF", "tetrahydrofuran"],
    }
    standardized = standardize_condition(raw)
    assert standardized["mobile_phase_solvents"] == ["Acetonitrile", "Tetrahydrofuran"]
    assert standardized["mobile_phase_solvent_qualifiers"] == ["near crit", None]
    assert standardized["mobile_phase_solvents_raw"] == raw["mobile_phase_solvents"]
    assert _format_polycrit_solvents(
        standardized["mobile_phase_solvents"],
        standardized["mobile_phase_solvent_qualifiers"],
    ) == "Acetonitrile (near crit), Tetrahydrofuran"

    standardized_twice = standardize_condition(standardized)
    assert standardized_twice["mobile_phase_solvents_raw"] == raw["mobile_phase_solvents"]
    assert standardized_twice["mobile_phase_solvent_qualifiers"] == ["near crit", None]


def test_aqueous_modifier_classification():
    acid = standardize_condition(
        {"mobile_phase_solvents": ["water with 0.1% formic acid", "MeOH"]}
    )
    assert acid["mobile_phase_solvents"] == ["Water", "Methanol"]
    assert acid["aqueous_parameters"]["pH_modifier"] == "0.1% formic acid"

    original_aqueous = {
        "pH": 7,
        "salt_added": False,
        "salt_type": None,
        "salt_concentration": None,
    }
    salt = standardize_condition(
        {
            "mobile_phase_solvents": ["water with 0.1 M NaCl", "ACN"],
            "aqueous_parameters": original_aqueous,
        }
    )
    assert salt["mobile_phase_solvents"] == ["Water", "Acetonitrile"]
    assert salt["aqueous_parameters_raw"] == original_aqueous
    assert salt["aqueous_parameters"]["salt_added"] is True
    assert salt["aqueous_parameters"]["salt_type"] == "Sodium chloride"
    assert salt["aqueous_parameters"]["salt_concentration"] == "0.1 M"


def test_standardize_file_preserves_source_metadata():
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        input_path = root / "sample_consensus.json"
        output_path = root / "sample_standardized.json"
        payload = {
            "metadata": {
                "source_pdf": "[1] Sample.pdf",
                "model": "local-programmatic-consensus",
                "inputs": ["model-a", "model-b"],
            },
            "custom_top_level": {"keep": True},
            "extracted_data": {
                "conditions": [
                    {
                        "analyte_polymer": "PEO",
                        "mobile_phase_solvents": ["THF", "H2O"],
                    }
                ]
            },
        }
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        assert standardize_file(input_path, output_path) == 1
        output = json.loads(output_path.read_text(encoding="utf-8"))
        assert output["metadata"]["source_pdf"] == "[1] Sample.pdf"
        assert output["metadata"]["model"] == "local-programmatic-consensus"
        assert output["metadata"]["inputs"] == ["model-a", "model-b"]
        assert output["metadata"]["standardized_by"] == "pipeline/standardizer.py"
        assert output["custom_top_level"] == {"keep": True}
        assert output["extracted_data"]["conditions"][0]["analyte_polymer"] == "Poly(ethylene glycol)"


def test_polycrit_review_export_preserves_complete_raw_audit_values():
    raw_condition = {
        "publication_year": "Published 1998",
        "paper_doi": "10.1000/example",
        "corresponding_author_name": "Ada Example",
        "corresponding_email_address": "ada@example.org",
        "physical_address": "Example Laboratory, Chicago, IL",
        "critical_condition_basis": "Explicit LCCC statement",
        "critical_condition_confidence": "explicit",
        "model_confidences": {"qwen": 0.91, "mistral": 0.87},
        "analyte_polymer": "PEO",
        "critical_component": "PEG",
        "architecture": "ring",
        "mobile_phase_solvents": ["THF", "THF", "H2O"],
        "mobile_phase_ratio": "1:1",
        "mobile_phase_ratio_units": "vol.%",
        "stationary_phase_chemistry": "C18 silica",
        "column_name": "Example C18 column",
        "column_mode": "RP",
        "column_dimensions": "250 x 4.6 mm, 5 µm",
        "pore_size": "30 nm",
        "temperature_celsius": "25 C",
        "flow_rate": "500 uL/min",
        "detector": "RI",
        "aqueous_parameters": {
            "pH": 7,
            "salt_added": False,
            "salt_type": None,
            "salt_concentration": None,
        },
        "evidence_text": "Raw evidence for review.",
        "notes": "Raw notes for review.",
    }
    standardized = standardize_condition(raw_condition)

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        input_dir = root / "standardized"
        input_dir.mkdir()
        payload = {
            "metadata": {"source_pdf": "[42] Example.pdf"},
            "extracted_data": {"conditions": [standardized]},
        }
        (input_dir / "example_standardized.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        review_csv = root / "review.csv"
        export_folder_to_csv(
            str(input_dir),
            str(review_csv),
            mode="polycrit",
            include_review_columns=True,
        )
        with review_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == POLYCRIT_FIELDNAMES_REVIEW
            assert reader.fieldnames[: len(POLYCRIT_FIELDNAMES)] == POLYCRIT_FIELDNAMES
            assert reader.fieldnames[len(POLYCRIT_FIELDNAMES) :] == (
                POLYCRIT_REVIEW_AUDIT_FIELDNAMES
            )
            row = next(reader)

        assert row["Reference"] == "42"
        assert row["Reference (Raw)"] == "[42] Example.pdf"
        assert row["Paper DOI"] == "10.1000/example"
        assert row["Corresponding Author Name"] == "Ada Example"
        assert row["Corresponding Email Address"] == "ada@example.org"
        assert row["Physical Address"] == "Example Laboratory, Chicago, IL"
        assert row["Publication Year (Raw)"] == "Published 1998"
        assert row["Author Year"] == "1998"
        assert row["Critical Condition Basis"] == "Explicit LCCC statement"
        assert row["Critical Condition Confidence"] == "explicit"
        assert json.loads(row["Model Confidences"]) == {"qwen": 0.91, "mistral": 0.87}
        assert row["Qwen Confidence"] == "0.91"
        assert row["Mistral Confidence"] == "0.87"
        assert row["Analyte Polymer (Raw)"] == "PEO"
        assert row["Analyte Polymer (Standardized)"] == "Poly(ethylene glycol)"
        assert row["Critical Component (Raw)"] == "PEG"
        assert row["Critical Component (Standardized)"] == "Poly(ethylene glycol)"
        assert row["Architecture (Raw)"] == "ring"
        assert row["Architecture (Standardized)"] == "cyclic"
        assert row["Solvents (Raw)"] == '["THF", "THF", "H2O"]'
        assert row["Solvents (Standardized)"] == '["Tetrahydrofuran", "Water"]'
        assert row["Solvent Ratio (Raw)"] == "1:1"
        assert row["Solvent Ratio Units (Raw)"] == "vol.%"
        assert row["Solvent Ratio Components (Standardized)"] == "[50.0, 50.0]"
        assert row["Solvent Ratio Units (Standardized)"] == "v/v"
        assert row["Pore Size (Raw)"] == "30 nm"
        assert row["Pore Size (Standardized)"] == "300.0"
        assert row["Temperature (Raw)"] == "25 C"
        assert row["Temperature (Standardized)"] == "25"
        assert row["Flow Rate (Raw)"] == "500 uL/min"
        assert row["Flow Rate (Standardized)"] == "0.5"
        assert row["Stationary Phase (Raw)"] == "C18 silica"
        assert row["Base Material Source (Raw)"] == "C18 silica"
        assert row["Base Material"] == "Silica"
        assert row["Column Mode (Raw)"] == "RP"
        assert row["Column Mode (Standardized)"] == "Reversed Phase"
        assert row["Column Dimensions (Raw)"] == "250 x 4.6 mm, 5 µm"
        assert row["Particle diameter (μm)"] == "5"
        assert row["Manufacturer (Raw)"] == ""
        assert row["Injected Polymer Concentration (Raw)"] == ""
        assert row["Detector (Raw)"] == "RI"
        assert row["Detector"] == "RI"
        assert row["Evidence Text"] == "Raw evidence for review."
        assert row["Notes"] == "Raw notes for review."
        assert json.loads(row["Aqueous Parameters (Raw)"]) == raw_condition[
            "aqueous_parameters"
        ]
        assert json.loads(row["Aqueous Parameters (Standardized)"]) == raw_condition[
            "aqueous_parameters"
        ]
        assert len(load_polycrit_csv(review_csv)) == 1

        base_csv = root / "base.csv"
        export_folder_to_csv(str(input_dir), str(base_csv), mode="polycrit")
        with base_csv.open(newline="", encoding="utf-8") as handle:
            assert next(csv.reader(handle)) == POLYCRIT_FIELDNAMES


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
    assert std["mobile_phase_solvents"] == ["Water", "Methanol"]
    assert std["mobile_phase_solvents_raw"] == [
        "water with 0.1% formic acid",
        "methanol",
    ]
    assert std["aqueous_parameters"]["pH_modifier"] == "0.1% formic acid"
