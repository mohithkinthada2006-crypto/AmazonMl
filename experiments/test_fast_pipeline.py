import os, sys, time
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
from src.preprocessing import preprocess_dataframe
from src.blocking import FastCountryBlocker
from src.features import build_feature_matrix

print("1. Loading raw sample...", flush=True)
s1_raw = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=1000, dtype=str)
s1 = preprocess_dataframe(s1_raw)
target_raw = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=10000, dtype=str)
target = preprocess_dataframe(target_raw)
print(f"Loaded: S1={len(s1)}, Target={len(target)}", flush=True)

print("2. Building blocker...", flush=True)
t0 = time.time()
blocker = FastCountryBlocker(target)
print(f"Blocker built in {time.time()-t0:.2f}s", flush=True)

print("3. Querying candidates...", flush=True)
t0 = time.time()
cands = blocker.get_candidates(s1, max_cands=30)
n_cands = sum(len(v) for v in cands.values())
print(f"Candidates queried in {time.time()-t0:.2f}s (Total {n_cands} pairs)", flush=True)

print("4. Building feature matrix...", flush=True)
t0 = time.time()
X, y, pairs = build_feature_matrix(cands, s1, target, ground_truth=None, is_training=False)
print(f"Feature matrix built in {time.time()-t0:.2f}s, Shape: {X.shape}", flush=True)
print("ALL TESTS PASSED SUCCESSFULLY!", flush=True)

