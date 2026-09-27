import time
import pandas as pd
from src.preprocessing import preprocess_dataframe

def test_groupby_index():
    t0 = time.time()
    print("Loading 1M rows S2...", flush=True)
    df = pd.read_csv("dataset/train/train_source2.tsv", sep='\t', nrows=1000000, dtype=str)
    df = preprocess_dataframe(df)
    print(f"Loaded in {time.time()-t0:.2f}s", flush=True)
    
    t0 = time.time()
    # Vectorized groupby index
    valid_mask = df['primary_name_token'].str.len() >= 3
    grp = df[valid_mask].groupby('primary_name_token')['entity_id'].apply(lambda s: s.values[:250].tolist())
    tok_dict = grp.to_dict()
    print(f"Pandas groupby index built ({len(tok_dict):,} keys) in {time.time()-t0:.2f}s", flush=True)

if __name__ == '__main__':
    test_groupby_index()
