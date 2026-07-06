#!/usr/bin/env python3
"""
Post-process all existing consensus JSONs to remove surviving duplicates.
No LLM calls. Pure dedup using updated _chromatographic_match logic.

Run from project root ONLY after test_dedup_fix.py passes completely:
  python3 scripts/dedup_existing_consensus.py
"""
import json, logging, sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault('vllm', MagicMock())
sys.modules.setdefault('vllm.sampling_params', MagicMock())
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.consensus_judge import ConsensusJudge

log_path = Path('logs') / f'dedup_postprocess_{datetime.now():%Y%m%d_%H%M%S}.log'
log_path.parent.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_path)])
log = logging.getLogger()

judge = ConsensusJudge(init_llm=False)
total_files = total_changed = total_removed = 0

for f in sorted(Path('results/consensus').rglob('*_consensus.json')):
    try:
        data = json.load(open(f))
    except Exception as e:
        log.warning(f"SKIP {f.name}: {e}"); continue

    conds = data.get('extracted_data', {}).get('conditions', [])
    before = len(conds)
    total_files += 1
    if before < 2:
        continue

    deduped = judge._dedup_conditions(conds)
    after = len(deduped)

    if after < before:
        removed = before - after
        total_changed += 1
        total_removed += removed
        log.info(f"[CHANGED] {f.stem}: {before} → {after} ({removed} removed)")
        removed_conds = [c for c in conds if not any(
            judge._chromatographic_match(c, d) for d in deduped)]
        for rc in removed_conds:
            log.info(f"  - {rc.get('analyte_polymer')!r}  col={rc.get('column_name')!r}  "
                     f"ratio={rc.get('mobile_phase_ratio')!r}  temp={rc.get('temperature_celsius')!r}")
        data['extracted_data']['conditions'] = deduped
        data['summary']['total_conditions'] = after
        with open(f, 'w') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

log.info(f"\n{'='*60}")
log.info(f"Files processed: {total_files} | Changed: {total_changed} | Conditions removed: {total_removed}")
log.info(f"Log: {log_path}")
