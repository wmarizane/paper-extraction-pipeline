"""Property-style robustness tests for mobile-phase standardization.

These do NOT test fixed strings from papers we've seen. Instead they take a
handful of *semantic truths* (which solvent is at which fraction, in which
unit) and generate many surface forms — different spacing, unit spelling/
placement, separators, and solvent/fraction orderings — asserting they all
standardize to the same correct structure. That's what demonstrates the
standardizer generalizes to layouts it hasn't encountered.

Run: python3 -m pytest tests/test_mobile_phase_robustness.py -q
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.standardizer import standardize_condition, _canonical_ratio_unit


# ── 1. Unit parsing is invariant to spelling / spacing / ordering ───────

WEIGHT_FORMS = [
    "w/w", "% w/w", "%w/w", "w / w", "wt%", "wt.%", "wt %", "% wt",
    "weight", "% by weight", "Gew.-%", "gew%",
]
VOLUME_FORMS = [
    "v/v", "% v/v", "%v/v", "v / v", "vol%", "vol.-%", "vol %", "% vol",
    "volume", "% by volume",
]
AMBIGUOUS_OR_NONE = ["w/v", "v/w", "%", "", None, "percent"]


def test_weight_unit_forms_all_collapse():
    assert {_canonical_ratio_unit(f) for f in WEIGHT_FORMS} == {"w/w"}


def test_volume_unit_forms_all_collapse():
    assert {_canonical_ratio_unit(f) for f in VOLUME_FORMS} == {"v/v"}


def test_ambiguous_or_empty_units_are_none():
    assert {_canonical_ratio_unit(f) for f in AMBIGUOUS_OR_NONE} == {None}


# ── 2. Solvent↔fraction pairing survives every input ordering ───────────

def _render_labeled(components, sep):
    """Render '<frac>% <solvent>' components joined by a separator."""
    return sep.join(f"{frac}% {name}" for frac, name in components)


def _pairing(out):
    return dict(zip(out["mobile_phase_solvents"], out["mobile_phase_ratio_components"]))


# semantic truth: canonical solvent -> fraction (canonical display names)
TRUTHS = [
    {"Tetrahydrofuran": 43.4, "Hexane": 56.6},
    {"Chloroform": 58.05, "Methanol": 6.45, "Heptane": 35.5},
    {"Acetonitrile": 70.0, "Water": 30.0},
]

# solvent surface aliases the extractor might emit (resolve to the canonical)
ALIASES = {
    "Tetrahydrofuran": ["THF", "tetrahydrofuran"],
    "Hexane": ["n-hexane", "Hexane"],
    "Chloroform": ["chloroform", "CHCl3"],
    "Methanol": ["methanol", "MeOH"],
    "Heptane": ["n-heptane", "heptane"],
    "Acetonitrile": ["ACN", "acetonitrile"],
    "Water": ["water", "H2O"],
}


def test_pairing_is_correct_regardless_of_input_order():
    for truth in TRUTHS:
        canon_solvents = list(truth)
        # try the ratio string components in several orders, and the solvent
        # LIST in several orders — the output pairing must always match truth.
        comp_orders = list(itertools.permutations(canon_solvents))
        list_orders = list(itertools.permutations(canon_solvents))
        # cap the combinatorics for the 3-solvent case
        for comp_order in comp_orders[:4]:
            for list_order in list_orders[:4]:
                for sep in (" : ", ", ", "/"):
                    components = [(truth[s], ALIASES[s][0]) for s in comp_order]
                    ratio = _render_labeled(components, sep)
                    solvents_list = [ALIASES[s][1] for s in list_order]
                    out = standardize_condition({
                        "mobile_phase_solvents": solvents_list,
                        "mobile_phase_ratio": ratio,
                    })
                    got = _pairing(out)
                    # every canonical solvent maps to its true fraction
                    for s in canon_solvents:
                        assert s in got, f"{s} missing in {got} (ratio={ratio!r}, list={solvents_list})"
                        assert abs(got[s] - truth[s]) < 1e-6, (
                            f"{s}: {got[s]} != {truth[s]} (ratio={ratio!r}, list={solvents_list})"
                        )


def test_flip_between_two_solvents_is_fixed():
    # the canonical failure: list order is the reverse of the ratio order.
    out = standardize_condition({
        "mobile_phase_solvents": ["n-hexane", "THF"],          # hexane first
        "mobile_phase_ratio": "43.4% THF : 56.6% n-hexane",     # THF first
    })
    assert out["mobile_phase_solvents"] == ["Tetrahydrofuran", "Hexane"]
    assert out["mobile_phase_ratio_components"] == [43.4, 56.6]
    assert out.get("mobile_phase_order_reconciled") is True


# ── 3. Embedded unit backfill (unit lives in the ratio string) ──────────

def test_embedded_unit_backfilled_across_forms():
    for form, expected in [("wt.%", "w/w"), ("wt %", "w/w"), ("vol%", "v/v"),
                           ("% by volume", "v/v")]:
        out = standardize_condition({
            "mobile_phase_solvents": ["Methanol"],
            "mobile_phase_ratio": f"85.8 {form} methanol",
        })
        assert out.get("mobile_phase_ratio_units") == expected, form
        assert out.get("mobile_phase_ratio_units_source") == "embedded_in_ratio"


# ── 4. Safety: never invent or corrupt an ordering ──────────────────────

def test_bare_numeric_ratio_is_not_reordered():
    # no inline solvent names -> nothing to reconcile, order untouched
    out = standardize_condition({
        "mobile_phase_solvents": ["Water", "Methanol"],
        "mobile_phase_ratio": "70:30",
    })
    assert out["mobile_phase_solvents"] == ["Water", "Methanol"]
    assert out.get("mobile_phase_order_reconciled") is None


def test_inline_solvent_mismatch_flags_but_does_not_corrupt():
    # ratio names a solvent not in the list -> flag, leave data unchanged
    before = ["Water", "Methanol"]
    out = standardize_condition({
        "mobile_phase_solvents": list(before),
        "mobile_phase_ratio": "70% acetone : 30% water",
    })
    assert out["mobile_phase_solvents"] == before
    assert out.get("mobile_phase_order_reconciled") is None
    assert out.get("mobile_phase_order_note")


def test_locant_solvent_not_split_by_its_comma():
    # "1,4-dioxane" contains a comma but must stay one solvent, paired right
    out = standardize_condition({
        "mobile_phase_solvents": ["Hexane", "1,4-Dioxane"],
        "mobile_phase_ratio": "60% 1,4-dioxane : 40% hexane",
    })
    assert out["mobile_phase_solvents"] == ["1,4-Dioxane", "Hexane"]
    assert out["mobile_phase_ratio_components"] == [60.0, 40.0]


# ── 5. Idempotence: standardizing twice changes nothing ─────────────────

def test_idempotent_on_reconciled_fields():
    once = standardize_condition({
        "mobile_phase_solvents": ["n-hexane", "THF"],
        "mobile_phase_ratio": "43.4% THF : 56.6% n-hexane",
    })
    twice = standardize_condition(once)
    assert twice["mobile_phase_solvents"] == once["mobile_phase_solvents"]
    assert twice["mobile_phase_ratio_components"] == once["mobile_phase_ratio_components"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
