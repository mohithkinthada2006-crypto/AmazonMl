import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import rapidfuzz
from rapidfuzz import fuzz
from src.similarity import compute_string_similarities, jaccard_similarity, char_ngram_set

def extract_digits_set(text: str) -> set:
    """Extracts all continuous digit tokens from string."""
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d+\b', text))

FEATURE_NAMES = [
    'is_s2', 'is_s3', 'country_match',
    'name_exact', 'name_ratio', 'name_tok_set', 'name_tok_jaccard', 'name_ngram_jaccard', 'name_len_diff', 'name_len_ratio', 'name_pfx_match',
    'addr_exact', 'addr_ratio', 'addr_tok_jaccard', 'addr_ngram_jaccard', 'addr_len_diff', 'addr_len_ratio',
    'addr_digit_jaccard', 'addr_has_common_digit',
    'name_x_addr_ratio', 'name_x_addr_tok_jaccard', 'name_x_addr_ngram_jaccard',
    'min_ratio', 'max_ratio', 'mean_ratio'
]

def make_entity_cache(row: dict) -> dict:
    """Precomputes and caches token/ngram/digit sets once per entity."""
    name = row.get('norm_name', '')
    addr = row.get('norm_address', '')
    return {
        'norm_name': name,
        'norm_address': addr,
        'norm_country': row.get('norm_country', ''),
        'name_toks': set(name.split()) if name else set(),
        'name_ngrams': char_ngram_set(name, 3),
        'addr_toks': set(addr.split()) if addr else set(),
        'addr_ngrams': char_ngram_set(addr, 3),
        'addr_digits': extract_digits_set(addr)
    }

def compute_pair_features_fast(
    s1_entry: dict,
    cand_entry: dict,
    cand_id: str
) -> list:
    """
    Computes fast feature list using precomputed entity caches.
    """
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0
    
    s1_c = s1_entry['norm_country']
    c_c = cand_entry['norm_country']
    country_match = 1.0 if s1_c == c_c and s1_c != '' else 0.0
    
    # Names
    s1_name = s1_entry['norm_name']
    cand_name = cand_entry['norm_name']
    name_exact = 1.0 if s1_name == cand_name and s1_name != '' else 0.0
    
    # Fast short-circuit for completely disjoint names
    toks_disjoint = s1_entry['name_toks'].isdisjoint(cand_entry['name_toks'])
    pfx3_match = (s1_name[:3] == cand_name[:3]) if (len(s1_name) >= 3 and len(cand_name) >= 3) else False
    
    if name_exact == 0.0 and toks_disjoint and not pfx3_match:
        return [
            is_s2, is_s3, country_match,
            0.0, 0.0, 0.0, 0.0, 0.0, float(abs(len(s1_name) - len(cand_name))), 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, float(abs(len(s1_entry['norm_address']) - len(cand_entry['norm_address']))), 0.0,
            0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ]
        
    name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    name_tok_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_tok_jaccard = jaccard_similarity(s1_entry['name_toks'], cand_entry['name_toks'])
    name_ngram_jaccard = jaccard_similarity(s1_entry['name_ngrams'], cand_entry['name_ngrams'])
    
    l1, l2 = len(s1_name), len(cand_name)
    name_len_diff = float(abs(l1 - l2))
    name_len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0
    name_pfx_match = 1.0 if s1_name and cand_name and s1_name[:4] == cand_name[:4] else 0.0
    
    # Addresses
    s1_addr = s1_entry['norm_address']
    cand_addr = cand_entry['norm_address']
    addr_exact = 1.0 if s1_addr == cand_addr and s1_addr != '' else 0.0
    addr_ratio = fuzz.ratio(s1_addr, cand_addr) / 100.0
    addr_tok_jaccard = jaccard_similarity(s1_entry['addr_toks'], cand_entry['addr_toks'])
    addr_ngram_jaccard = jaccard_similarity(s1_entry['addr_ngrams'], cand_entry['addr_ngrams'])
    
    al1, al2 = len(s1_addr), len(cand_addr)
    addr_len_diff = float(abs(al1 - al2))
    addr_len_ratio = (min(al1, al2) / max(al1, al2)) if max(al1, al2) > 0 else 1.0
    
    # Digits
    s1_dig = s1_entry['addr_digits']
    c_dig = cand_entry['addr_digits']
    addr_digit_jaccard = jaccard_similarity(s1_dig, c_dig)
    addr_has_common_digit = 1.0 if len(s1_dig.intersection(c_dig)) > 0 else 0.0
    
    # Interactions
    name_x_addr_ratio = name_ratio * addr_ratio
    name_x_addr_tok_jaccard = name_tok_jaccard * addr_tok_jaccard
    name_x_addr_ngram_jaccard = name_ngram_jaccard * addr_ngram_jaccard
    min_ratio = min(name_ratio, addr_ratio)
    max_ratio = max(name_ratio, addr_ratio)
    mean_ratio = 0.5 * (name_ratio + addr_ratio)
    
    return [
        is_s2, is_s3, country_match,
        name_exact, name_ratio, name_tok_set, name_tok_jaccard, name_ngram_jaccard, name_len_diff, name_len_ratio, name_pfx_match,
        addr_exact, addr_ratio, addr_tok_jaccard, addr_ngram_jaccard, addr_len_diff, addr_len_ratio,
        addr_digit_jaccard, addr_has_common_digit,
        name_x_addr_ratio, name_x_addr_tok_jaccard, name_x_addr_ngram_jaccard,
        min_ratio, max_ratio, mean_ratio
    ]

def build_feature_matrix(
    candidate_map: Dict[str, set],
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth: Dict[str, set] = None,
    is_training: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Constructs feature matrix X as a contiguous numpy array with pre-cached entity representations.
    """
    s1_cache = {}
    cols_s1 = ['entity_id', 'norm_name', 'norm_address', 'norm_country']
    for row in s1_df[cols_s1].to_dict(orient='records'):
        s1_cache[row['entity_id']] = make_entity_cache(row)
        
    all_needed_cands = set()
    for cands in candidate_map.values():
        all_needed_cands.update(cands)
        
    cols_t = ['entity_id', 'norm_name', 'norm_address', 'norm_country']
    target_sub = target_df[target_df['entity_id'].isin(all_needed_cands)][cols_t]
    target_cache = {}
    for row in target_sub.to_dict(orient='records'):
        target_cache[row['entity_id']] = make_entity_cache(row)
        
    feature_rows = []
    labels = []
    pairs = []
    
    for s1_id, cands in candidate_map.items():
        s1_entry = s1_cache.get(s1_id)
        if not s1_entry:
            continue
            
        true_matches = ground_truth.get(s1_id, set()) if ground_truth is not None else set()
        
        for cand_id in cands:
            cand_entry = target_cache.get(cand_id)
            if not cand_entry:
                continue
                
            feat_list = compute_pair_features_fast(s1_entry, cand_entry, cand_id)
            feature_rows.append(feat_list)
            pairs.append((s1_id, cand_id))
            
            if is_training:
                is_match = 1 if cand_id in true_matches else 0
                labels.append(is_match)
                
    X = np.array(feature_rows, dtype=np.float32) if feature_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(labels, dtype=np.int32) if is_training else None
    
    return X, y, pairs
