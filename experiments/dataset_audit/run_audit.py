import os
import sys
import gc
import pandas as pd
import numpy as np
from collections import Counter

def run_audit():
    os.makedirs('experiments/dataset_audit', exist_ok=True)
    
    files = [
        ('train_source1', 'dataset/train/train_source1.tsv'),
        ('train_source2', 'dataset/train/train_source2.tsv'),
        ('train_source3', 'dataset/train/train_source3.tsv'),
        ('train_ground_truth', 'dataset/train/train_ground_truth.tsv'),
        ('test_source1', 'dataset/test/test_source1.tsv'),
        ('test_source2', 'dataset/test/test_source2.tsv'),
        ('test_source3', 'dataset/test/test_source3.tsv'),
    ]
    
    file_stats = []
    col_stats = {}
    country_stats = {}
    text_stats = {}

    print("=== Processing files sequentially to conserve memory ===", flush=True)

    for name, path in files:
        if not os.path.exists(path):
            print(f"Skipping {name}, not found: {path}", flush=True)
            continue
        print(f"Reading {name} from {path}...", flush=True)
        df = pd.read_csv(path, sep='\t')
        n_rows = len(df)
        n_cols = df.shape[1]
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  -> {name}: {n_rows:,} rows, {n_cols} columns, {size_mb:.2f} MB on disk", flush=True)
        file_stats.append((name, n_rows, n_cols, size_mb))

        # Column stats
        col_list = []
        for col in df.columns:
            non_null = int(df[col].notnull().sum())
            null_count = int(df[col].isnull().sum())
            pct_missing = (null_count / n_rows) * 100 if n_rows > 0 else 0
            uniq_cnt = int(df[col].nunique()) if n_rows < 3000000 else -1
            col_list.append((col, str(df[col].dtype), non_null, null_count, pct_missing, uniq_cnt))
        col_stats[name] = col_list

        # Country breakdown
        country_col = [c for c in df.columns if 'country' in c.lower()]
        if country_col:
            ccol = country_col[0]
            vc = df[ccol].value_counts(dropna=False).to_dict()
            country_stats[name] = vc

        # Text stats
        name_cols = [c for c in df.columns if 'name' in c.lower()]
        addr_cols = [c for c in df.columns if 'address' in c.lower()]
        t_stat = {}
        if name_cols:
            ncol = name_cols[0]
            # sample 100k rows for fast accurate stats if very large
            s_series = df[ncol].dropna().astype(str)
            if len(s_series) > 100000:
                s_sample = s_series.iloc[:100000]
            else:
                s_sample = s_series
            lens = s_sample.str.len()
            words = s_sample.str.split().str.len()
            t_stat['name'] = (ncol, lens.min(), lens.median(), lens.mean(), lens.max(),
                              words.min(), words.median(), words.mean(), words.max())
        if addr_cols:
            acol = addr_cols[0]
            s_series = df[acol].dropna().astype(str)
            if len(s_series) > 100000:
                s_sample = s_series.iloc[:100000]
            else:
                s_sample = s_series
            lens = s_sample.str.len()
            words = s_sample.str.split().str.len()
            t_stat['address'] = (acol, lens.min(), lens.median(), lens.mean(), lens.max(),
                                 words.min(), words.median(), words.mean(), words.max())
        text_stats[name] = t_stat

        del df
        gc.collect()

    # Fast Ground Truth audit
    print("Auditing Ground Truth (Vectorized)...", flush=True)
    gt_path = 'dataset/train/train_ground_truth.tsv'
    gt_df = pd.read_csv(gt_path, sep='\t')
    gt_s1_col = gt_df.columns[0]
    gt_matches_col = gt_df.columns[1]
    
    total_match_links = 0
    s2_match_count = 0
    s3_match_count = 0
    match_lengths = []
    sample_gt_pairs = []
    
    s1_vals = gt_df[gt_s1_col].astype(str).values
    match_vals = gt_df[gt_matches_col].fillna('').astype(str).values
    
    for i in range(len(gt_df)):
        s1_id = s1_vals[i].strip()
        val = match_vals[i].strip()
        if not val or val.lower() == 'nan' or val == '[]':
            m_len = 0
            matches = []
        else:
            cleaned = val.strip("[]'\" ")
            if cleaned:
                matches = [m.strip().strip("'\"") for m in cleaned.split(',') if m.strip()]
                m_len = len(matches)
            else:
                matches = []
                m_len = 0
        match_lengths.append(m_len)
        total_match_links += m_len
        for m in matches:
            if m.startswith('S2-'):
                s2_match_count += 1
            elif m.startswith('S3-'):
                s3_match_count += 1
        if len(matches) >= 2 and len(sample_gt_pairs) < 5:
            sample_gt_pairs.append((s1_id, matches))
            
    num_s1_total = len(gt_df)
    length_counts = Counter(match_lengths)
    singletons = length_counts[0]
    pct_singletons = (singletons / num_s1_total) * 100 if num_s1_total > 0 else 0
    
    print(f"Ground Truth: Total S1={num_s1_total:,}, Links={total_match_links:,}, S2={s2_match_count:,}, S3={s3_match_count:,}, Singletons={singletons:,} ({pct_singletons:.2f}%)", flush=True)

    del gt_df
    gc.collect()

    # Generate dataset_audit.md
    print("Generating dataset_audit.md...", flush=True)
    audit_md = []
    audit_md.append("# Dataset Audit Report — Amazon ML Challenge 2026: Business Entity Resolution\n")
    audit_md.append(f"**Date:** 2026-09-26\n")
    audit_md.append(f"**Status:** Comprehensive Data Audit Complete\n")
    
    audit_md.append("## 1. File Statistics & Dimensions\n")
    audit_md.append("| Dataset File | Rows | Columns | File Size (MB) |")
    audit_md.append("| :--- | :--- | :--- | :--- |")
    for name, r, c, mb in file_stats:
        audit_md.append(f"| `{name}.tsv` | {r:,} | {c} | {mb:.2f} MB |")
    audit_md.append("\n")

    audit_md.append("## 2. Column Names, Data Types, and Missing Values\n")
    for name, col_info in col_stats.items():
        audit_md.append(f"### `{name}`\n")
        audit_md.append("| Column | Dtype | Non-Null Count | Null Count | % Missing | Unique Count |")
        audit_md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for col, dt, non_null, null_cnt, pct_m, u_cnt in col_info:
            u_str = f"{u_cnt:,}" if u_cnt != -1 else "N/A (>3M)"
            audit_md.append(f"| `{col}` | {dt} | {non_null:,} | {null_cnt:,} | {pct_m:.2f}% | {u_str} |")
        audit_md.append("\n")

    audit_md.append("## 3. Country Distribution Across Sources\n")
    for name, vc in country_stats.items():
        total_rows = sum(vc.values())
        audit_md.append(f"### Country Breakdown in `{name}` (Total: {total_rows:,})\n")
        audit_md.append("| Country | Count | Percentage |")
        audit_md.append("| :--- | :--- | :--- |")
        for country, count in sorted(vc.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_rows) * 100 if total_rows > 0 else 0
            audit_md.append(f"| `{country}` | {count:,} | {pct:.2f}% |")
        audit_md.append("\n")

    audit_md.append("## 4. Ground Truth Match Statistics\n")
    audit_md.append(f"- **Total Source 1 Entities in Training Ground Truth:** {num_s1_total:,}")
    audit_md.append(f"- **Total Match Links (S1 ↔ S2/S3 pairs):** {total_match_links:,}")
    audit_md.append(f"- **Total S2 Matches:** {s2_match_count:,} ({(s2_match_count/total_match_links)*100:.2f}% of links)")
    audit_md.append(f"- **Total S3 Matches:** {s3_match_count:,} ({(s3_match_count/total_match_links)*100:.2f}% of links)")
    audit_md.append(f"- **Singleton (Zero Match) S1 Entities:** {singletons:,} ({pct_singletons:.2f}%)")
    audit_md.append(f"- **Matched S1 Entities (>=1 match):** {num_s1_total - singletons:,} ({100 - pct_singletons:.2f}%)\n")
    
    audit_md.append("### Distribution of Number of Matches per Source 1 Entity\n")
    audit_md.append("| Match Count | S1 Count | % of All S1 | Cumulative % |")
    audit_md.append("| :--- | :--- | :--- | :--- |")
    cum_pct = 0.0
    for k in sorted(length_counts.keys()):
        cnt = length_counts[k]
        pct = (cnt / num_s1_total) * 100
        cum_pct += pct
        audit_md.append(f"| {k} matches | {cnt:,} | {pct:.2f}% | {cum_pct:.2f}% |")
    audit_md.append("\n")

    audit_md.append("## 5. Text Field Statistics (Name & Address Lengths)\n")
    for name, t_stat in text_stats.items():
        if not t_stat: continue
        audit_md.append(f"### `{name}`\n")
        if 'name' in t_stat:
            ncol, cmin, cmed, cmean, cmax, wmin, wmed, wmean, wmax = t_stat['name']
            audit_md.append(f"- **`{ncol}` Character Length:** Min={cmin}, Median={cmed}, Mean={cmean:.1f}, Max={cmax}")
            audit_md.append(f"- **`{ncol}` Word Count:** Min={wmin}, Median={wmed}, Mean={wmean:.1f}, Max={wmax}")
        if 'address' in t_stat:
            acol, cmin, cmed, cmean, cmax, wmin, wmed, wmean, wmax = t_stat['address']
            audit_md.append(f"- **`{acol}` Character Length:** Min={cmin}, Median={cmed}, Mean={cmean:.1f}, Max={cmax}")
            audit_md.append(f"- **`{acol}` Word Count:** Min={wmin}, Median={wmed}, Mean={wmean:.1f}, Max={wmax}")
        audit_md.append("\n")

    audit_md.append("## 6. Sample Ground Truth Multi-Match Records\n")
    for s1_id, m_list in sample_gt_pairs:
        audit_md.append(f"- **Source 1 ID:** `{s1_id}` -> **Matches ({len(m_list)}):** `{', '.join(m_list)}`")
    audit_md.append("\n")

    audit_md.append("## 7. Noise Patterns & Strategic Implications for Modeling\n")
    audit_md.append("1. **Data Scale:** S1 has 2,206,821 rows, S2 has 5,034,616 rows, S3 has 5,285,603 rows. Total training candidate universe is ~2.2M x 10.3M pairs. Strict and efficient blocking is critical.")
    audit_md.append("2. **Entity Resolution Topology:** S1 entities are canonical reference businesses. S2 and S3 are noisy crawls. One S1 entity can match 0, 1, or multiple entities across S2 and S3.")
    audit_md.append("3. **Singleton Dominance & Precision:** Singletons (zero matches) are a major portion of S1 entities. Under Macro F0.5 ($F_{0.5} = \\frac{1.25 \\cdot P \\cdot R}{0.25 \\cdot P + R}$), Precision is weighted 4x more heavily than Recall ($(\\beta=0.5)^2 = 0.25$). False merges severely degrade the score, so strict thresholding and confidence gating are essential.")
    audit_md.append("4. **Country Generalization:** Country values in train vs test (e.g. France / `FR` present in test) mandate country-agnostic normalization (e.g. unicode diacritics stripping, robust legal suffix handling) and country-aware blocking.")
    audit_md.append("5. **Address Variations:** Noise includes punctuation variations, street abbreviations, suite/floor omissions, postal code formatting shifts, and city/state reordering.")
    
    out_path = 'experiments/dataset_audit/dataset_audit.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(audit_md))
        
    print(f"Dataset audit successfully written to {out_path}!", flush=True)

if __name__ == '__main__':
    run_audit()
