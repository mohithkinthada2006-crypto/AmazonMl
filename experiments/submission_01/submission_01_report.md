# Submission 01 Experiment Report: Strong Baseline Model

**Date/Time:** 2026-09-27 01:14:14
**Pipeline Runtime:** 5900.3 seconds (98.34 minutes)

## 1. Pipeline Configuration & Architecture
- **Model:** LightGBM Gradient Boosted Decision Trees (`n_estimators=250`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`)
- **Validation Split:** 12,000 Source-1 entities strictly split at entity level
- **Optimal Probability Threshold:** 0.40
- **Validation Macro Precision:** 0.9869 (98.69%)
- **Validation Macro Recall:** 0.1422
- **Validation Macro F0.5:** 0.2279
- **Validation Singleton Accuracy:** 0.9838 (98.38%)

## 2. Test Set Predictions
- **Total Test S1 Entities:** 1,732,544
- **Total Predicted Business Match Links:** 1,814,062
- **Predicted Singletons:** 744,477 (42.97%)

## 3. Submission Artifacts & Verification
- `output/submission_01/matching_results.tsv`
- `output/submission_01/candidate_pairs.tsv`
- `output/matching_results.tsv` (Active copy)
- `output/candidate_pairs.tsv` (Active copy)
- **Official Validator Check:** PASSED (Status Code 0)
