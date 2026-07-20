#!/usr/bin/env python3
"""Audit v5 consensus results quality."""
import json, re
from pathlib import Path
from collections import Counter

SUBFOLDERS = ['PEG', 'PLA', 'PPO', 'pdf_files_1stbatch', 'pdf_files_2ndbatch', 'pdf_files_3rdbatch']
consensus_dir = Path('results/consensus')

all_conds = []
for sf in SUBFOLDERS:
    sf_dir = consensus_dir / sf
    if not sf_dir.exists():
        continue
    for jf in sf_dir.glob('*_consensus.json'):
        paper = jf.stem.replace('_consensus', '')
        with open(jf) as f:
            data = json.load(f)
        for c in data.get('extracted_data', {}).get('conditions', []):
            c['_paper'] = paper
            c['_subfolder'] = sf
            all_conds.append(c)

N = len(all_conds)
print(f'Total conditions: {N}')

# 1. NULL RATES
key_fields = ['analyte_polymer','critical_component','column_name','mobile_phase_solvents',
              'mobile_phase_ratio','temperature_celsius','flow_rate','detector',
              'paper_doi','corresponding_author_name','corresponding_email_address',
              'stationary_phase_chemistry','pore_size','column_dimensions','column_mode']
null_counts = Counter()
for c in all_conds:
    for f in key_fields:
        v = c.get(f)
        if v is None or v == '' or (isinstance(v, str) and v.lower() == 'null'):
            null_counts[f] += 1
print('\n== NULL/EMPTY FIELD RATES ==')
for f in sorted(null_counts, key=null_counts.get, reverse=True):
    print(f'  {f:<32} {null_counts[f]:>3}/{N} ({100*null_counts[f]/N:.0f}%)')

# 2. String "null"
ns = 0
ns_fields = Counter()
for c in all_conds:
    for k, v in c.items():
        if k.startswith('_'):
            continue
        if isinstance(v, str) and v.lower() == 'null':
            ns += 1
            ns_fields[k] += 1
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, str) and sv.lower() == 'null':
                    ns += 1
                    ns_fields[f'{k}.{sk}'] += 1
print(f'\n== STRING "null" ({ns} total) ==')
for k, cnt in ns_fields.most_common():
    print(f'  {k}: {cnt}')

# 3. Comma analytes
print('\n== COMMA-SEPARATED ANALYTES ==')
ca = 0
for c in all_conds:
    a = str(c.get('analyte_polymer', '') or '')
    if ', ' in a:
        stripped = re.sub(r'\([^)]*\)', '', a)
        parts = [p for p in stripped.split(', ') if p.strip()]
        if len(parts) >= 2:
            ca += 1
            print(f'  [{c["_subfolder"]}] {c["_paper"]}: "{a}"')
print(f'  Total: {ca}')

# 4. Fractionation
print('\n== FRACTIONATION/PREP KEYWORDS ==')
fr = 0
for c in all_conds:
    ev = str(c.get('evidence_text', '')).lower()
    kws = ['fractionat', 'preparative', 'semipreparative', 'semi-preparative']
    if any(kw in ev for kw in kws):
        fr += 1
        mc = c.get('model_confidences', {})
        print(f'  [{c["_subfolder"]}] {c["_paper"]}: {c.get("analyte_polymer")} | T={c.get("temperature_celsius")} | Q={mc.get("qwen","?")} M={mc.get("mistral","?")}')
print(f'  Total: {fr}')

# 5. High temp
print('\n== HIGH TEMP (>100C) ==')
for c in all_conds:
    t = c.get('temperature_celsius')
    if t is None:
        continue
    try:
        tv = float(str(t).replace('°C', '').strip())
        if tv > 100:
            print(f'  {c["_paper"]}: T={t} | {c.get("analyte_polymer")}')
    except:
        pass

# 6. Null-heavy
print('\n== NULL-HEAVY ROWS (>=3 of 6 critical fields null) ==')
crit = ['column_name', 'mobile_phase_ratio', 'temperature_celsius', 'flow_rate', 'detector', 'pore_size']
nh = 0
for c in all_conds:
    nulls = sum(1 for f in crit if not c.get(f) or (isinstance(c.get(f), str) and c.get(f).lower() == 'null'))
    if nulls >= 3:
        nh += 1
        print(f'  [{c["_subfolder"]}] {c["_paper"]}: {c.get("analyte_polymer")} | nulls={nulls}')
print(f'  Total: {nh}')

# 7. column_mode
print('\n== COLUMN_MODE FIELD ==')
cm = Counter()
for c in all_conds:
    v = c.get('column_mode')
    if v and not (isinstance(v, str) and v.lower() == 'null'):
        cm[v] += 1
    else:
        cm['EMPTY/NULL'] += 1
for val, cnt in cm.most_common(10):
    print(f'  {val:<40} {cnt:>4}')

# 8. Model agreement
print('\n== MODEL AGREEMENT ==')
both = qonly = monly = neither = 0
for c in all_conds:
    mc = c.get('model_confidences', {})
    q = mc.get('qwen', 'missed')
    m = mc.get('mistral', 'missed')
    if q != 'missed' and m != 'missed': both += 1
    elif q != 'missed': qonly += 1
    elif m != 'missed': monly += 1
    else: neither += 1
print(f'  Both:    {both:>4}/{N} ({100*both/N:.1f}%)')
print(f'  Qwen:    {qonly:>4}/{N} ({100*qonly/N:.1f}%)')
print(f'  Mistral: {monly:>4}/{N} ({100*monly/N:.1f}%)')
print(f'  Neither: {neither:>4}/{N} ({100*neither/N:.1f}%)')

# 9. Near-duplicates
print('\n== NEAR-DUPLICATES (same col/ratio/temp, diff analyte, same comp) ==')
from collections import defaultdict
def norm(v):
    if not v: return ''
    return re.sub(r'\s+', ' ', str(v).lower().strip())
by_paper = defaultdict(list)
for c in all_conds:
    by_paper[(c['_subfolder'], c['_paper'])].append(c)
nd = 0
for (sf, paper), conds in by_paper.items():
    for i in range(len(conds)):
        for j in range(i+1, len(conds)):
            ci, cj = conds[i], conds[j]
            if (norm(ci.get('column_name')) and norm(ci.get('column_name')) == norm(cj.get('column_name')) and
                norm(ci.get('mobile_phase_ratio')) and norm(ci.get('mobile_phase_ratio')) == norm(cj.get('mobile_phase_ratio')) and
                norm(ci.get('temperature_celsius')) and norm(ci.get('temperature_celsius')) == norm(cj.get('temperature_celsius')) and
                norm(ci.get('analyte_polymer')) != norm(cj.get('analyte_polymer')) and
                norm(ci.get('critical_component')) == norm(cj.get('critical_component'))):
                nd += 1
print(f'  Total: {nd}')

# 10. Exact duplicates
print('\n== EXACT DUPLICATES ==')
ed = 0
for (sf, paper), conds in by_paper.items():
    seen = set()
    for c in conds:
        sig = (norm(c.get('analyte_polymer')), norm(c.get('critical_component')),
               norm(c.get('column_name')), norm(c.get('mobile_phase_ratio')),
               norm(c.get('temperature_celsius')))
        if sig in seen: ed += 1
        else: seen.add(sig)
print(f'  Total: {ed}')

# 11. Papers with changed counts vs v4 
print('\n== SUBFOLDER TOTALS ==')
for sf in SUBFOLDERS:
    cnt = sum(1 for c in all_conds if c['_subfolder'] == sf)
    print(f'  {sf:<25} {cnt:>4}')
