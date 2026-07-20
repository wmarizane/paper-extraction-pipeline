#!/usr/bin/env python3
"""Compare old consensus summary CSVs against new consensus JSON files."""

import csv
import json
from pathlib import Path
from collections import defaultdict

results_dir = Path("results")
consensus_dir = results_dir / "consensus"

SUBFOLDERS = ["PEG", "PLA", "PPO", "pdf_files_1stbatch", "pdf_files_2ndbatch", "pdf_files_3rdbatch"]

# --- Load OLD counts from summary CSVs ---
old_counts = {}  # subfolder -> {paper: count}
old_totals = {}  # subfolder -> total

for sf in SUBFOLDERS:
    csv_path = results_dir / f"{sf}_consensus_summary.csv"
    if not csv_path.exists():
        print(f"WARNING: Old CSV not found: {csv_path}")
        continue
    
    paper_counts = defaultdict(int)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paper = row.get("Paper", "unknown")
            paper_counts[paper] += 1
    
    old_counts[sf] = dict(paper_counts)
    old_totals[sf] = sum(paper_counts.values())

# --- Load NEW counts from consensus JSONs ---
new_counts = {}  # subfolder -> {paper: count}
new_totals = {}  # subfolder -> total
new_details = {}  # subfolder -> {paper: [conditions]}

for sf in SUBFOLDERS:
    sf_dir = consensus_dir / sf
    if not sf_dir.exists():
        print(f"WARNING: New consensus dir not found: {sf_dir}")
        continue
    
    paper_counts = {}
    paper_details = {}
    
    for jf in sorted(sf_dir.glob("*_consensus.json")):
        paper = jf.stem.replace("_consensus", "")
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            conds = data.get("extracted_data", {}).get("conditions", [])
            paper_counts[paper] = len(conds)
            paper_details[paper] = conds
        except Exception as e:
            print(f"ERROR reading {jf}: {e}")
    
    new_counts[sf] = paper_counts
    new_totals[sf] = sum(paper_counts.values())
    new_details[sf] = paper_details

# --- Print comparison ---
print("=" * 80)
print("CONSENSUS QUALITY COMPARISON: OLD vs NEW")
print("=" * 80)

grand_old = 0
grand_new = 0

for sf in SUBFOLDERS:
    ot = old_totals.get(sf, 0)
    nt = new_totals.get(sf, 0)
    grand_old += ot
    grand_new += nt
    
    delta = nt - ot
    sign = "+" if delta > 0 else ""
    
    print(f"\n{'─' * 60}")
    print(f"  {sf}: OLD={ot}  NEW={nt}  ({sign}{delta})")
    print(f"{'─' * 60}")
    
    # Get all papers from both old and new
    all_papers = sorted(set(list(old_counts.get(sf, {}).keys()) + list(new_counts.get(sf, {}).keys())))
    
    for paper in all_papers:
        oc = old_counts.get(sf, {}).get(paper, 0)
        nc = new_counts.get(sf, {}).get(paper, 0)
        d = nc - oc
        
        if d == 0:
            marker = "  "
        elif d > 0:
            marker = " ▲"
        else:
            marker = " ▼"
        
        ds = f"+{d}" if d > 0 else str(d)
        print(f"    {paper:<45} OLD={oc:<3} NEW={nc:<3} ({ds}){marker}")

print(f"\n{'=' * 80}")
print(f"  GRAND TOTAL: OLD={grand_old}  NEW={grand_new}  ({'+' if grand_new - grand_old > 0 else ''}{grand_new - grand_old})")
print(f"{'=' * 80}")

# --- Detailed analysis: Papers with changes ---
print(f"\n\n{'=' * 80}")
print("DETAILED CHANGES: Papers where condition count changed")
print(f"{'=' * 80}")

for sf in SUBFOLDERS:
    all_papers = sorted(set(list(old_counts.get(sf, {}).keys()) + list(new_counts.get(sf, {}).keys())))
    
    for paper in all_papers:
        oc = old_counts.get(sf, {}).get(paper, 0)
        nc = new_counts.get(sf, {}).get(paper, 0)
        
        if oc != nc:
            print(f"\n  📄 [{sf}] {paper}: {oc} → {nc} conditions")
            
            # Show new conditions details
            conds = new_details.get(sf, {}).get(paper, [])
            for i, c in enumerate(conds):
                analyte = c.get("analyte_polymer", "?")
                comp = c.get("critical_component", "?")
                col = c.get("column_name", "?")
                ratio = c.get("mobile_phase_ratio", "?")
                temp = c.get("temperature_celsius", "?")
                conf = c.get("critical_condition_confidence", "?")
                mc = c.get("model_confidences", {})
                qc = mc.get("qwen", "?")
                msc = mc.get("mistral", "?")
                print(f"      [{i+1}] analyte={analyte} | comp={comp} | col={col} | ratio={ratio} | temp={temp}°C | conf={conf} | qwen={qc} mistral={msc}")

# --- Model confidence analysis ---
print(f"\n\n{'=' * 80}")
print("MODEL AGREEMENT ANALYSIS")
print(f"{'=' * 80}")

both_found = 0
qwen_only = 0
mistral_only = 0
both_missed = 0
total = 0

for sf in SUBFOLDERS:
    for paper, conds in new_details.get(sf, {}).items():
        for c in conds:
            mc = c.get("model_confidences", {})
            qc = mc.get("qwen", "missed")
            msc = mc.get("mistral", "missed")
            total += 1
            
            if qc != "missed" and msc != "missed":
                both_found += 1
            elif qc != "missed" and msc == "missed":
                qwen_only += 1
            elif qc == "missed" and msc != "missed":
                mistral_only += 1
            else:
                both_missed += 1

print(f"  Both models found:     {both_found}/{total} ({100*both_found/total:.1f}%)" if total else "  No conditions found")
print(f"  Qwen only:             {qwen_only}/{total} ({100*qwen_only/total:.1f}%)" if total else "")
print(f"  Mistral only:          {mistral_only}/{total} ({100*mistral_only/total:.1f}%)" if total else "")
print(f"  Neither (judge-added): {both_missed}/{total} ({100*both_missed/total:.1f}%)" if total else "")
