#!/usr/bin/env python3
"""Apply the vague-row absorber to existing consensus JSONs, then re-export
the affected subfolder CSVs. No LLM calls — pure post-processing, safe to run
locally/on the cluster where the real results/consensus/ tree lives.

  python3 scripts/apply_vague_absorber.py            # apply + re-export
  python3 scripts/apply_vague_absorber.py --dry-run  # report only, write nothing

The absorber drops a condition only when a strictly-more-specific same-paper
condition covers it on every populated field and refers to the same (or a more
specific) analyte. See pipeline/pre_consensus_dedup.absorb_vague_conditions.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.sampling_params", MagicMock())
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pre_consensus_dedup import absorb_vague_conditions
from pipeline.csv_exporter import export_folder_to_csv

CONSENSUS_DIR = Path("results/consensus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    if not CONSENSUS_DIR.exists():
        print(f"Not found: {CONSENSUS_DIR} (run from project root).")
        return

    total_before = total_after = files_changed = 0
    affected_subfolders = set()

    for f in sorted(CONSENSUS_DIR.rglob("*_consensus.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"SKIP {f}: {e}")
            continue
        conds = d.get("extracted_data", {}).get("conditions", [])
        total_before += len(conds)
        if len(conds) < 2:
            total_after += len(conds)
            continue
        kept = absorb_vague_conditions(conds)
        total_after += len(kept)
        if len(kept) < len(conds):
            files_changed += 1
            rel = f.relative_to(CONSENSUS_DIR)
            print(f"{rel}: {len(conds)} -> {len(kept)}")
            if f.parent != CONSENSUS_DIR:
                affected_subfolders.add(f.parent.name)
            if not args.dry_run:
                d["extracted_data"]["conditions"] = kept
                if isinstance(d.get("summary"), dict):
                    d["summary"]["total_conditions"] = len(kept)
                with open(f, "w", encoding="utf-8") as fh:
                    json.dump(d, fh, indent=2, ensure_ascii=False)

    print(f"\n{files_changed} files, {total_before} -> {total_after} conditions "
          f"({total_before - total_after} absorbed)"
          + (" [DRY RUN — nothing written]" if args.dry_run else ""))

    if not args.dry_run and affected_subfolders:
        print("\nRe-exporting affected subfolder CSVs:")
        for sub in sorted(affected_subfolders):
            src = CONSENSUS_DIR / sub
            out = Path("results") / f"{sub}_consensus_summary.csv"
            export_folder_to_csv(str(src), str(out))


if __name__ == "__main__":
    main()
