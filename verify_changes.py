import re

CANONICAL_POLYMERS = {
    "ps": "polystyrene",
    "polystyrene": "polystyrene",
    "peo": "poly(ethylene glycol)",
    "peg": "poly(ethylene glycol)",
    "poly(ethylene oxide)": "poly(ethylene glycol)",
    "poly(ethylene glycol)": "poly(ethylene glycol)",
    "eo": "poly(ethylene glycol)",
    "pmma": "poly(methyl methacrylate)",
    "poly(methyl methacrylate)": "poly(methyl methacrylate)",
    "ppo": "poly(propylene glycol)",
    "poly(propylene oxide)": "poly(propylene glycol)",
    "poly(propylene glycol)": "poly(propylene glycol)",
    "po": "poly(propylene glycol)",
    "bo": "poly(butene oxide)",
    "poly(butene oxide)": "poly(butene oxide)",
    "poly(butylene oxide)": "poly(butene oxide)",
    "ho": "poly(hexene oxide)",
    "poly(hexene oxide)": "poly(hexene oxide)",
    "pp": "polypropylene",
    "polypropylene": "polypropylene",
    "poly(l-lactic acid)": "poly(lactic acid)",
    "poly(lactide)": "poly(lactic acid)",
    "polylactide": "poly(lactic acid)",
    "pi": "polyisoprene",
    "polyisoprene": "polyisoprene",
    "1,4-pi": "polyisoprene",
    "polyisoprene (1,4-pi)": "polyisoprene",
    "polyisoprene (1,4-isoprene)": "polyisoprene",
}

ARCH_PREFIX_RE = re.compile(
    r'^(ring|cyclic|linear|star|comb|ls|lu|it|at|st|dendri|hyper|branched)[_\-\s]+',
    re.IGNORECASE
)

def _canonicalize_polymer(val: str) -> str:
    if not val:
        return ""
    val_norm = val.lower().strip()

    val_norm = re.sub(
        r'\s+(block|repeat\s*unit|repeating\s*unit|unit|segment|chain|backbone)$',
        '', val_norm, flags=re.IGNORECASE
    ).strip()

    val_clean = re.sub(r'[^a-z0-9]', '', val_norm)
    for key, canonical in CANONICAL_POLYMERS.items():
        key_clean = re.sub(r'[^a-z0-9]', '', key)
        if val_clean == key_clean or val_clean == key or val_norm == key:
            return canonical

    val_stripped = ARCH_PREFIX_RE.sub('', val_norm).strip()
    if val_stripped and val_stripped != val_norm:
        val_stripped_clean = re.sub(r'[^a-z0-9]', '', val_stripped)
        for key, canonical in CANONICAL_POLYMERS.items():
            key_clean = re.sub(r'[^a-z0-9]', '', key)
            if val_stripped_clean == key_clean or val_stripped_clean == key or val_stripped == key:
                return canonical

    return val_norm

import sys

def test():
    tests = [
        ("Ring-PS", "polystyrene"),
        ("Ls-PS", "polystyrene"),
        ("Lu-PS", "polystyrene"),
        ("it-PP", "polypropylene"),
        ("BO block", "poly(butene oxide)"),
        ("EO repeat unit", "poly(ethylene glycol)"),
        ("PO block", "poly(propylene glycol)"),
        ("HO block", "poly(hexene oxide)"),
        ("polystyrene", "polystyrene"),
        ("PEG", "poly(ethylene glycol)"),
        ("Star-PMMA", "poly(methyl methacrylate)")
    ]

    for val, expected in tests:
        result = _canonicalize_polymer(val)
        if result != expected:
            print(f"FAILED: '{val}' -> '{result}' (expected '{expected}')")
            sys.exit(1)
        else:
            print(f"PASSED: '{val}' -> '{result}'")

    # Read files directly to check for strings
    with open("pipeline/llm_extractor.py", "r") as f:
        llm_content = f.read()
    if '"column_mode"' not in llm_content:
        print("FAILED: column_mode not in EXTRACTION_SCHEMA")
        sys.exit(1)
    else:
        print("PASSED: column_mode in EXTRACTION_SCHEMA")
        
    with open("pipeline/csv_exporter.py", "r") as f:
        csv_content = f.read()
    if '"Column Mode"' not in csv_content:
        print("FAILED: 'Column Mode' not in csv_exporter.FIELDNAMES")
        sys.exit(1)
    else:
        print("PASSED: 'Column Mode' in csv_exporter.FIELDNAMES")
        
    print("\nALL CUSTOM TESTS PASSED!")

if __name__ == "__main__":
    test()
