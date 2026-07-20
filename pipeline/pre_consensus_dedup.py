"""Pre-consensus per-model deduplication.

Removes duplicate conditions within a single model's extraction before
the extraction enters the consensus pipeline. Implements Dr. Wang's
7-7 feedback (items 1-4): a 6-field chromatographic fingerprint
("stationary phase chemistry", "column name", "mobile phase solvents",
"mobile phase ratio", "temperature", "critical component") with
case-insensitive, polymer-chemistry-aware matching.

An additional analyte guard prevents over-merging of genuinely distinct
records that share the same critical condition fingerprint:
  - different end-group functionality (PIB-diol vs PIB-diallyl)
  - molecular-weight series (peg 2010 vs peg 6240)
  - different architecture prefixes (Ring-PS vs Ls-PS)

Reuses matching helpers from pipeline.consensus_judge — no logic is
duplicated. NOTE: importing this module transitively imports vllm (via
consensus_judge). For local/laptop use, mock vllm first, e.g.:

    import sys
    from unittest.mock import MagicMock
    sys.modules.setdefault('vllm', MagicMock())
    sys.modules.setdefault('vllm.sampling_params', MagicMock())
"""

import re
from typing import Dict, List

from pipeline.consensus_judge import (
    ConsensusJudge,
    CANONICAL_POLYMERS,
    ARCH_PREFIX_RE,
    _CONSENSUS_FUNCTIONAL_SUFFIX_RE,
    _CONSENSUS_MW_TOKEN_RE,
)

# ── Tolerances (single source of truth — tighten here if dry-run shows over-merging) ──
RATIO_TOLERANCE = 2.0   # leading-number tolerance for mobile phase ratio
TEMP_TOLERANCE = 2.0    # °C tolerance for column temperature

# ── Solvent synonym table ──────────────────────────────────────────────
# Layered ON TOP of ConsensusJudge._norm_solvents / CANONICAL_SOLVENTS.
# Only entries missing from CANONICAL_SOLVENTS are needed here; targets
# are aligned with CANONICAL_SOLVENTS canonical names (e.g. IPA maps to
# "isopropyl alcohol", the canonical used by consensus_judge — NOT
# "isopropanol").
SOLVENT_SYNONYMS = {
    "h2o": "water",
    "meoh": "methanol",
    "etoh": "ethanol",
    "ipa": "isopropyl alcohol",
    "isopropanol": "isopropyl alcohol",
    "2-propanol": "isopropyl alcohol",
    "isopropyl alcohol": "isopropyl alcohol",
}

# ── Suffix patterns stripped from critical component before comparing ──
_CC_SUFFIX_RE = re.compile(
    r'\s+(block|repeat(?:ing)?\s*units?|units?|segments?|chains?|backbone|moiety|moieties)$',
    re.IGNORECASE,
)

# Generic words ignored when checking column-name token overlap
_COLUMN_STOPWORDS = {
    "column", "columns", "of", "the", "a", "an", "and", "with", "x", "by",
    "two", "three", "four", "five", "rp", "np", "hplc", "lc", "phase",
}

_CANONICAL_VALUES = set(CANONICAL_POLYMERS.values())


# ── Basic normalizers ──────────────────────────────────────────────────

def _norm(val) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not val:
        return ""
    return re.sub(r'\s+', ' ', str(val).lower().strip())


def _alnum(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s)


def _deplural(w: str) -> str:
    """Strip a single trailing plural 's' (conservative: keeps 'ps' intact)."""
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    return w


def _extract_number(val) -> float | None:
    """Extract leading numeric value from a string."""
    if val is None or val == "":
        return None
    m = re.search(r'(-?\d+(?:\.\d+)?)', str(val))
    return float(m.group(1)) if m else None


_RATIO_UNIT_RE = re.compile(r'(vol\s*\.?\s*%|wt\s*\.?\s*%|v\s*/\s*v|w\s*/\s*w|%)', re.IGNORECASE)


def _ratio_number(val) -> float | None:
    """Leading numeric value of a ratio after stripping separators and unit labels."""
    if val is None or val == "":
        return None
    s = _RATIO_UNIT_RE.sub(' ', str(val).lower())
    s = re.sub(r'[/\\:–—-]', ' ', s)
    return _extract_number(s)


def _norm_solvents_ext(solv) -> frozenset:
    """ConsensusJudge._norm_solvents + the extra SOLVENT_SYNONYMS layer."""
    base = ConsensusJudge._norm_solvents(solv)
    out = set()
    for s in base:
        out.add(SOLVENT_SYNONYMS.get(s, SOLVENT_SYNONYMS.get(_alnum(s), s)))
    return frozenset(out)


# ── Signal matchers (null in either condition ⇒ signal auto-matches) ───

def _spc_match(a, b) -> bool:
    """Signal 1: stationary phase chemistry. Exact or containment, case-insensitive."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return True
    return na == nb or na in nb or nb in na


def _col_match(a, b) -> bool:
    """Signal 2: column name. Substring, Jaccard >= 0.4, or a shared
    non-numeric distinctive token (e.g. 'Nucleosil C18' vs
    'three RP columns of C18 (100-5, 300-5, 1000-7)' share 'c18')."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return True
    if na in nb or nb in na:
        return True
    if ConsensusJudge._word_jaccard(na, nb) >= 0.4:
        return True
    ta = {w for w in re.findall(r'[a-z0-9]+', na) if w not in _COLUMN_STOPWORDS}
    tb = {w for w in re.findall(r'[a-z0-9]+', nb) if w not in _COLUMN_STOPWORDS}
    shared = {w for w in (ta & tb) if not w.isdigit()}
    return bool(shared)


def _solv_match(a, b) -> bool:
    """Signal 3: mobile phase solvents. Set equality or Jaccard >= 0.8."""
    sa, sb = _norm_solvents_ext(a), _norm_solvents_ext(b)
    if not sa or not sb:
        return True
    if sa == sb:
        return True
    union = sa | sb
    return len(sa & sb) / len(union) >= 0.8 if union else True


def _ratio_match(a, b) -> bool:
    """Signal 4: mobile phase ratio. Leading numbers within RATIO_TOLERANCE."""
    na, nb = _ratio_number(a), _ratio_number(b)
    if na is None or nb is None:
        return True
    return abs(na - nb) <= RATIO_TOLERANCE


def _temp_match(a, b) -> bool:
    """Signal 5: temperature. Numeric values within TEMP_TOLERANCE °C."""
    na, nb = _extract_number(a), _extract_number(b)
    if na is None or nb is None:
        return True
    return abs(na - nb) <= TEMP_TOLERANCE


# ── Signal 6: critical component (polymer-chemistry-aware) ─────────────

def _cc_variants(s: str) -> set:
    """Normalized variants: raw, suffix-stripped, arch-prefix-stripped."""
    out = {s}
    t = _CC_SUFFIX_RE.sub('', s).strip()
    if t:
        out.add(t)
    u = ARCH_PREFIX_RE.sub('', t or s).strip()
    if u:
        out.add(u)
    return out


def _canon_or_none(v: str):
    """Canonical polymer family, or None when unknown. Plural-tolerant."""
    c = ConsensusJudge._canonicalize_polymer(v)
    if c in _CANONICAL_VALUES:
        return c
    v_al = _deplural(_alnum(v))
    for key, canon in CANONICAL_POLYMERS.items():
        if v_al == _alnum(key):
            return canon
    return None


def _cc_word_jaccard(a: str, b: str) -> float:
    """Word Jaccard with abbreviation expansion (PEG -> poly ethylene glycol)
    and plural tolerance."""
    def words(s):
        ws = set()
        for w in re.findall(r'[a-z0-9]+', s):
            w = _deplural(w)
            expanded = None
            if len(w) >= 2:
                for key, canon in CANONICAL_POLYMERS.items():
                    if w == _alnum(key):
                        expanded = canon
                        break
            if expanded:
                ws.update(re.findall(r'[a-z0-9]+', expanded))
            else:
                ws.add(w)
        return ws
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _cc_match(a, b) -> bool:
    """Signal 6: critical component. Same polymer chemistry ⇒ match,
    regardless of phrasing (Dr. Wang item 3: 'linear poly(lactides)',
    'poly(lactide) repeating units', 'Linear poly(lactide)' are the same)."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return True
    if na == nb:
        return True

    va, vb = _cc_variants(na), _cc_variants(nb)

    # (b) canonical polymer family match (plural/architecture tolerant)
    ca = {c for c in (_canon_or_none(v) for v in va) if c}
    cb = {c for c in (_canon_or_none(v) for v in vb) if c}
    if ca & cb:
        return True

    for x in va:
        for y in vb:
            ax, ay = _alnum(x), _alnum(y)
            # (a) exact after normalization (plural tolerant)
            if ax and ay and (ax == ay or _deplural(ax) == _deplural(ay)):
                return True
            # monomer/polymer prefix tolerance: '1,4-isoprene' vs '1,4-polyisoprene'
            if ax and ay and len(ax) >= 4 and len(ay) >= 4:
                if _deplural(ax).replace('poly', '', 1) == _deplural(ay).replace('poly', '', 1):
                    return True
            # (c) abbreviation (PS <-> polystyrene)
            if ConsensusJudge._is_abbreviation(x, y):
                return True
            # (e) substring containment ('PS' in 'polystyrene (ps)')
            if x in y or y in x:
                return True
    # (d) expanded word-level Jaccard
    if _cc_word_jaccard(na, nb) >= 0.6:
        return True
    return False


# ── Analyte guard (prevents over-merging of distinct records) ──────────

_ENDGROUP_PHRASE_RE = re.compile(
    r'with\s+([\w-]+)\s+end[\s-]*groups?|(\w[\w-]*)[-\s]terminated\b',
    re.IGNORECASE,
)
_BLOCK_COUNT_RE = re.compile(r'\b(di|tri|tetra|penta|multi)[\s-]*block\b', re.IGNORECASE)
_BLOCK_CONNECTORS = {'b', 'block', 'co', 'stat', 'ran', 'alt', 'g', 'graft'}


def _endgroup_token(s: str):
    m = _ENDGROUP_PHRASE_RE.search(s)
    if not m:
        return None
    return (m.group(1) or m.group(2) or '').lower()


def _block_signature(s: str):
    """Ordered tuple of block tokens from dash-connected runs, canonicalized
    via CANONICAL_POLYMERS ('eo–po–eo' -> (PEG-family, PPG-family, PEG-family)).
    Returns None when the analyte has no dash-connected block structure."""
    t = re.sub(r'[–—]', '-', s)
    m = re.search(r'\b([a-z0-9]{1,14}(?:-[a-z0-9]{1,14})+)\b', t)
    if not m:
        return None
    tokens = []
    for tok in m.group(1).split('-'):
        if tok in _BLOCK_CONNECTORS:
            continue
        canon = None
        for key, c in CANONICAL_POLYMERS.items():
            if tok == _alnum(key):
                canon = c
                break
        tokens.append(canon or tok)
    return tuple(tokens) if len(tokens) >= 2 else None


def _seg_canon(s: str) -> str:
    """Canonicalize one block segment of a -b- copolymer name."""
    s = re.sub(r'\bblock\s+copolymers?\b', '', s).strip(' -')
    return _canon_or_none(s) or _alnum(s)


def _analyte_guard_ok(ca: Dict, cb: Dict) -> bool:
    """Return False when the two analyte polymers are provably DIFFERENT
    records even though the chromatographic fingerprint matches:
      1. different functional-group suffixes (PIB-diol vs PIB-diallyl)
      2. molecular-weight series (peg 2010 vs peg 6240, c10- vs c12-)
      3. different architecture prefixes (Ring-PS vs Ls-PS)
      4. different named end groups ('with hydroxyl end groups' vs
         'with methoxy end groups') — END-GROUPS rule
      5. different block architectures (EO-PO diblock vs EO-PO-EO
         triblock vs PO-EO-PO triblock)
      6. homopolymer vs block copolymer, or different -b- sequences
         (PEO vs PEO-b-PS vs PS-b-PEO-b-PS)
    Reuses the same regexes as ConsensusJudge._analyte_base_family_match;
    the full method is not used directly because its conservative fallback
    (unknown ⇒ different) would block legitimate same-analyte merges like
    'poly(ethylene glycol) methyl ether' vs 'PEG monomethyl ethers'."""
    a = _norm(ca.get("analyte_polymer"))
    b = _norm(cb.get("analyte_polymer"))
    if not a or not b or a == b:
        return True

    # 1. Functional-group suffix difference (diol vs diallyl, etc.)
    fa = _CONSENSUS_FUNCTIONAL_SUFFIX_RE.search(a)
    fb = _CONSENSUS_FUNCTIONAL_SUFFIX_RE.search(b)
    if fa and fb and fa.group().lower() != fb.group().lower():
        return False

    # 2. Molecular-weight series (both carry MW tokens that differ)
    mwa = set(_CONSENSUS_MW_TOKEN_RE.findall(a))
    mwb = set(_CONSENSUS_MW_TOKEN_RE.findall(b))
    if mwa and mwb and mwa != mwb:
        return False

    # 3. Architecture prefix difference (ring- vs linear-/ls-, etc.)
    ma = ARCH_PREFIX_RE.match(a)
    mb = ARCH_PREFIX_RE.match(b)
    if ma and mb:
        if ma.group(1).lower() != mb.group(1).lower():
            return False

    # 4. Named end-group difference (END-GROUPS rule):
    #    'PEO with hydroxyl end groups' vs 'PEO with methoxy end groups'
    ea, eb = _endgroup_token(a), _endgroup_token(b)
    if ea and eb and ea != eb:
        return False

    # 5. Block-architecture difference:
    #    keyword (diblock vs triblock) or block sequence (eo-po-eo vs po-eo-po)
    ka, kb = _BLOCK_COUNT_RE.search(a), _BLOCK_COUNT_RE.search(b)
    if ka and kb and ka.group(1).lower() != kb.group(1).lower():
        return False
    sa, sb = _block_signature(a), _block_signature(b)
    if sa and sb and sa != sb:
        return False

    # 6. Homopolymer vs block copolymer / different -b- sequences
    #    (mirrors ConsensusJudge._analyte_base_family_match HARD PRE-BLOCK 2).
    #    Only applied to single analytes: comma-separated multi-analyte lists
    #    (', ') are messy pre-split records — the judge's MULTIPLE ANALYTES
    #    rule handles those, so other signals decide.
    if ', ' not in a and ', ' not in b:
        a_blk = bool(re.search(r'-b-|block\s+copolymer', a))
        b_blk = bool(re.search(r'-b-|block\s+copolymer', b))
        if a_blk != b_blk:
            return False  # PEO homopolymer vs PEO-b-PS
        if a_blk and b_blk and '-b-' in a and '-b-' in b:
            seg_a = tuple(_seg_canon(x) for x in a.split('-b-'))
            seg_b = tuple(_seg_canon(x) for x in b.split('-b-'))
            if len(seg_a) > 1 and len(seg_b) > 1 and seg_a != seg_b:
                return False  # PEO-b-PS vs PS-b-PEO-b-PS

    return True


# ── Fingerprint match ──────────────────────────────────────────────────

def _conditions_match(ca: Dict, cb: Dict) -> bool:
    """True if two conditions describe the same experiment: the analyte
    guard passes AND all 6 fingerprint signals match (nulls tolerated)."""
    if not _analyte_guard_ok(ca, cb):
        return False
    return (
        _spc_match(ca.get("stationary_phase_chemistry"), cb.get("stationary_phase_chemistry"))
        and _col_match(ca.get("column_name"), cb.get("column_name"))
        and _solv_match(ca.get("mobile_phase_solvents"), cb.get("mobile_phase_solvents"))
        and _ratio_match(ca.get("mobile_phase_ratio"), cb.get("mobile_phase_ratio"))
        and _temp_match(ca.get("temperature_celsius"), cb.get("temperature_celsius"))
        and _cc_match(ca.get("critical_component"), cb.get("critical_component"))
    )


# ── Merge ──────────────────────────────────────────────────────────────

def _prefer(va, vb):
    """Prefer non-null; when both set and different, prefer the longer
    (usually the more detailed extraction)."""
    if va is None or va == "" or va == []:
        return vb
    if vb is None or vb == "" or vb == []:
        return va
    if va == vb:
        return va
    return va if len(str(va)) >= len(str(vb)) else vb


def _merge(ca: Dict, cb: Dict) -> Dict:
    """Merge two matching conditions field-by-field; field_evidence
    sub-fields are merged individually."""
    merged = {}
    for k in set(ca.keys()) | set(cb.keys()):
        va, vb = ca.get(k), cb.get(k)
        if k == "field_evidence":
            fa, fb = va or {}, vb or {}
            fe = {fk: _prefer(fa.get(fk), fb.get(fk)) for fk in set(fa) | set(fb)}
            merged[k] = fe if fe else (va if va is not None else vb)
        else:
            merged[k] = _prefer(va, vb)
    return merged


# ── Public API ─────────────────────────────────────────────────────────

def dedup_model_conditions(conditions: List[Dict]) -> List[Dict]:
    """Remove duplicate conditions from a single model's extraction.

    Args:
        conditions: List of condition dicts from one model's extraction
            of one paper.

    Returns:
        Deduplicated list; matching pairs are merged (non-null values
        preferred, longer values win conflicts).
    """
    if not conditions:
        return []
    deduped: List[Dict] = []
    for c in conditions:
        if not isinstance(c, dict):
            deduped.append(c)
            continue
        matched = False
        for i, existing in enumerate(deduped):
            if isinstance(existing, dict) and _conditions_match(c, existing):
                deduped[i] = _merge(existing, c)
                matched = True
                break
        if not matched:
            deduped.append(c)
    return deduped


# ── Vague-row absorber ─────────────────────────────────────────────────
# Handles the residual dupes that survive the fingerprint dedup: a row
# phrased more vaguely (generic analyte, missing/less-precise fields) that
# is fully covered by a strictly-more-specific row for the SAME polymer in
# the same paper. Example: a summary "PEG / XB-phenyl / 45:55 / 30°C" row
# shadowing the per-MW "PEG 2k", "PEG 4k", "PEG 6k" rows on that column.
#
# Deliberately stricter than _conditions_match: it keys on analyte
# compatibility (same or strictly-more-generic) rather than critical
# component, so two DIFFERENT analytes sharing one critical condition
# (MULTIPLE ANALYTES rule) are never collapsed.

_FINGERPRINT_KEYS = (
    "stationary_phase_chemistry", "column_name", "mobile_phase_solvents",
    "mobile_phase_ratio", "temperature_celsius", "critical_component",
)

_FIELD_MATCHERS = {
    "stationary_phase_chemistry": _spc_match,
    "column_name": _col_match,
    "mobile_phase_solvents": _solv_match,
    "mobile_phase_ratio": _ratio_match,
    "temperature_celsius": _temp_match,
    "critical_component": _cc_match,
}


def _nonempty_signals(c: Dict) -> int:
    """Count populated fingerprint fields."""
    return sum(1 for k in _FINGERPRINT_KEYS if _norm(c.get(k)))


def _vague_fields_covered(v: Dict, s: Dict) -> bool:
    """Every populated fingerprint field of v matches s (empty v field is a
    wildcard). Uses the same tolerant matchers as the fingerprint."""
    for k in _FINGERPRINT_KEYS:
        if _norm(v.get(k)) and not _FIELD_MATCHERS[k](v.get(k), s.get(k)):
            return False
    return True


def _analyte_generic_or_equal(v: Dict, s: Dict) -> bool:
    """True when v's analyte is the same as, or a strictly-more-generic form
    of, s's analyte. Requires the analyte guard to pass first, so it never
    bridges MW / end-group / architecture / block-distinct records."""
    if not _analyte_guard_ok(v, s):
        return False
    a, b = _norm(v.get("analyte_polymer")), _norm(s.get("analyte_polymer"))
    if not a or a == b:
        return True
    if _cc_match(a, b):  # same polymer family
        av, bv = _alnum(a), _alnum(b)
        if av and bv and av in bv and av != bv:
            return True  # 'peg' ⊂ 'peg2k'
        if ConsensusJudge._is_abbreviation(a, b):
            return True
    return False


def _is_strictly_more_specific(s: Dict, v: Dict) -> bool:
    """s is strictly more informative than v: it populates every fingerprint
    field v does (never drops v's unique info) AND either has more populated
    fields, or an equal field set with a strictly-more-specific analyte."""
    if not all(_norm(s.get(k)) for k in _FINGERPRINT_KEYS if _norm(v.get(k))):
        return False
    ns, nv = _nonempty_signals(s), _nonempty_signals(v)
    if ns > nv:
        return True
    if ns == nv:
        a, b = _alnum(_norm(v.get("analyte_polymer"))), _alnum(_norm(s.get("analyte_polymer")))
        if a and b and a != b and a in b:
            return True
    return False


def absorb_vague_conditions(conditions: List[Dict]) -> List[Dict]:
    """Drop a condition when a strictly-more-specific condition in the same
    list covers it on every populated fingerprint field and refers to the
    same (or a more specific) analyte.

    Conservative by design — respects the analyte guard, requires the
    absorbing row to retain all of the dropped row's information, and never
    collapses two rows that carry mutually-unique fields. Intended to run
    AFTER dedup_model_conditions / the judge, on the final condition set.
    """
    if not conditions:
        return []
    keep = [True] * len(conditions)
    for i, v in enumerate(conditions):
        if not isinstance(v, dict) or not keep[i]:
            continue
        for j, s in enumerate(conditions):
            if i == j or not keep[j] or not isinstance(s, dict):
                continue
            if (_vague_fields_covered(v, s)
                    and _analyte_generic_or_equal(v, s)
                    and _is_strictly_more_specific(s, v)):
                keep[i] = False
                break
    return [c for c, k in zip(conditions, keep) if k]
