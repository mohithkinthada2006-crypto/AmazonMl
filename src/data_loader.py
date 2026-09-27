import os
import pandas as pd
import numpy as np

def load_source_data(data_dir, source_type="train", sources=(1, 2, 3)):
    """
    Loads source datasets from TSV files.
    """
    dfs = {}
    base_folder = os.path.join(data_dir, source_type) if os.path.isdir(os.path.join(data_dir, source_type)) else data_dir
    
    for s_num in sources:
        fname = f"{source_type}_source{s_num}.tsv"
        fpath = os.path.join(base_folder, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, sep='\t', dtype=str)
            dfs[f"source{s_num}"] = df
        else:
            raise FileNotFoundError(f"Source file not found: {fpath}")
            
    return dfs

def load_ground_truth(gt_path, s1_ids_filter: set = None):
    """
    Loads ground truth mapping with ultra-fast streaming parser and early exit.
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
    gt_map = {}
    target_count = len(s1_ids_filter) if s1_ids_filter is not None else None
    
    with open(gt_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline()
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            s1_id = parts[0].strip()
            if s1_ids_filter is not None and s1_id not in s1_ids_filter:
                continue
            if len(parts) < 2 or not parts[1].strip() or parts[1].strip() in ('[]', 'nan', 'NaN'):
                gt_map[s1_id] = set()
            else:
                m_str = parts[1].strip("[]'\" ")
                if m_str:
                    gt_map[s1_id] = {x.strip().strip("'\"") for x in m_str.split(',') if x.strip()}
                else:
                    gt_map[s1_id] = set()
                    
            if target_count is not None and len(gt_map) >= target_count:
                break
                
    return gt_map

def split_s1_entities(s1_df, val_ratio=0.2, random_state=42):
    """
    Performs leakage-free train/validation split strictly at the Source 1 entity level.
    """
    np.random.seed(random_state)
    s1_ids = list(s1_df['entity_id'])
    np.random.shuffle(s1_ids)
    
    n_val = int(len(s1_ids) * val_ratio)
    val_ids = set(s1_ids[:n_val])
    train_ids = set(s1_ids[n_val:])
    
    train_s1 = s1_df[s1_df['entity_id'].isin(train_ids)].copy()
    val_s1 = s1_df[s1_df['entity_id'].isin(val_ids)].copy()
    
    return train_s1, val_s1
