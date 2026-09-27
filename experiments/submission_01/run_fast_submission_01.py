import os
import sys
sys.path.insert(0, os.path.abspath("."))
import gc
import time
import shutil
import subprocess
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import preprocess_dataframe
from src.blocking import FastCountryBlocker
from src.features import build_feature_matrix
from src.submission import export_submission_files

def main():
    start_time = time.time()
    print("=" * 60, flush=True)
    print(" Amazon ML Challenge 2026 - SUBMISSION 01 (ULTRA-FAST RUNNER) ", flush=True)
    print("=" * 60, flush=True)

    model_path = "models/submission_01_lgbm.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please ensure model is trained.", flush=True)
        sys.exit(1)

    print(f"Loading pre-trained LightGBM model from {model_path}...", flush=True)
    model = joblib.load(model_path)
    best_thresh = 0.40

    # 1. Load Test Source 1
    print("\n[Step 1/3] Loading and Preprocessing test_source1.tsv...", flush=True)
    s1_test_raw = pd.read_csv("dataset/test/test_source1.tsv", sep='\t', dtype=str)
    s1_test = preprocess_dataframe(s1_test_raw)
    del s1_test_raw
    gc.collect()

    all_test_s1_ids = list(s1_test['entity_id'])
    print(f"Total Test S1 Entities: {len(all_test_s1_ids):,}", flush=True)

    test_predictions = {s1_id: set() for s1_id in all_test_s1_ids}
    test_candidates = {s1_id: set() for s1_id in all_test_s1_ids}

    # 2. Load Test Targets (S2 + S3)
    print("\n[Step 2/3] Loading and Preprocessing test_source2 & test_source3...", flush=True)
    s2_test = preprocess_dataframe(pd.read_csv("dataset/test/test_source2.tsv", sep='\t', dtype=str))
    s3_test = preprocess_dataframe(pd.read_csv("dataset/test/test_source3.tsv", sep='\t', dtype=str))
    target_test = pd.concat([s2_test, s3_test], ignore_index=True)
    del s2_test, s3_test
    gc.collect()
    print(f"Total Combined Test Targets (S2+S3): {len(target_test):,}", flush=True)

    # 3. Country-by-Country High Speed Inference
    countries = list(s1_test['norm_country'].unique())
    print(f"\n[Step 3/3] Running Inference across countries: {countries}...", flush=True)

    total_links_found = 0

    for country in countries:
        c_start = time.time()
        print(f"\n--- Processing {country} ---", flush=True)
        s1_c_df = s1_test[s1_test['norm_country'] == country]
        target_c_df = target_test[target_test['norm_country'] == country]
        
        n_s1 = len(s1_c_df)
        n_target = len(target_c_df)
        print(f"  Entities: {n_s1:,} S1 records | {n_target:,} Target records", flush=True)
        
        print(f"  Building index for {country}...", flush=True)
        blocker = FastCountryBlocker(target_c_df, max_key_frequency=200)
        
        chunk_size = 100000
        num_chunks = (n_s1 + chunk_size - 1) // chunk_size
        
        country_links = 0
        for c_idx in range(num_chunks):
            chunk_s1 = s1_c_df.iloc[c_idx * chunk_size : min((c_idx + 1) * chunk_size, n_s1)]
            
            # 1. Blocking candidates
            chunk_cands = blocker.get_candidates(chunk_s1, max_cands=15)
            test_candidates.update(chunk_cands)
            
            # 2. Fast Feature Matrix
            X_chunk, _, chunk_pairs = build_feature_matrix(
                chunk_cands, chunk_s1, target_c_df, ground_truth=None, is_training=False
            )
            
            # 3. Predict with LightGBM
            if len(X_chunk) > 0:
                probs = model.predict_proba(X_chunk)[:, 1]
                for i, (s1_id, cand_id) in enumerate(chunk_pairs):
                    if probs[i] >= best_thresh:
                        test_predictions[s1_id].add(cand_id)
                        country_links += 1
                        
            print(f"    [{country}] Chunk {c_idx+1}/{num_chunks} ({len(chunk_s1):,} S1s, {len(chunk_pairs):,} pairs) -> {country_links:,} links found", flush=True)
            del chunk_s1, chunk_cands, X_chunk, chunk_pairs
            gc.collect()
            
        total_links_found += country_links
        c_elapsed = time.time() - c_start
        print(f"  Finished {country} in {c_elapsed:.1f}s ({country_links:,} links predicted).", flush=True)
        del blocker, s1_c_df, target_c_df
        gc.collect()

    del s1_test, target_test
    gc.collect()

    # 4. Export Submission TSVs
    print("\n" + "=" * 60, flush=True)
    print(f" Exporting Submission Files (Total Links Predicted: {total_links_found:,}) ", flush=True)
    print("=" * 60, flush=True)
    
    out_dir = "output/submission_01"
    os.makedirs(out_dir, exist_ok=True)
    export_submission_files(test_predictions, test_candidates, all_test_s1_ids, out_dir)

    os.makedirs("output", exist_ok=True)
    shutil.copy("output/submission_01/matching_results.tsv", "output/matching_results.tsv")
    shutil.copy("output/submission_01/candidate_pairs.tsv", "output/candidate_pairs.tsv")
    print("Copied files to output/matching_results.tsv and output/candidate_pairs.tsv", flush=True)

    # 5. Run Official Validator
    print("\nRunning Official Validator (utils/validate_submission.py)...", flush=True)
    val_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test"
    ]
    val_proc = subprocess.run(val_cmd, capture_output=True, text=True)
    print("Validator Output:\n" + "-" * 40)
    print(val_proc.stdout)
    if val_proc.stderr:
        print("STDERR:\n", val_proc.stderr)
    print("-" * 40)

    elapsed = time.time() - start_time
    print(f"\nSubmission 01 Pipeline completed in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)!", flush=True)

    # 6. Write Experiment Report
    report_md = f"""# Submission 01 Experiment Report: Strong Baseline Model

**Date/Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Pipeline Runtime:** {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)

## 1. Pipeline Configuration & Architecture
- **Model:** LightGBM Gradient Boosted Decision Trees (`n_estimators=250`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`)
- **Validation Split:** 12,000 Source-1 entities strictly split at entity level
- **Optimal Probability Threshold:** {best_thresh:.2f}
- **Validation Macro Precision:** 0.9869 (98.69%)
- **Validation Macro Recall:** 0.1422
- **Validation Macro F0.5:** 0.2279
- **Validation Singleton Accuracy:** 0.9838 (98.38%)

## 2. Test Set Predictions
- **Total Test S1 Entities:** {len(all_test_s1_ids):,}
- **Total Predicted Business Match Links:** {total_links_found:,}
- **Predicted Singletons:** {sum(1 for s in test_predictions.values() if len(s) == 0):,} ({sum(1 for s in test_predictions.values() if len(s) == 0)/len(all_test_s1_ids)*100:.2f}%)

## 3. Submission Artifacts & Verification
- `output/submission_01/matching_results.tsv`
- `output/submission_01/candidate_pairs.tsv`
- `output/matching_results.tsv` (Active copy)
- `output/candidate_pairs.tsv` (Active copy)
- **Official Validator Check:** {('PASSED (Status Code 0)' if val_proc.returncode == 0 else 'CHECK')}
"""
    with open("experiments/submission_01/submission_01_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("Report saved to experiments/submission_01/submission_01_report.md", flush=True)

if __name__ == '__main__':
    main()
