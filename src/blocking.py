import collections
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple

STOPWORDS = {
    'inc', 'llc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'the', 'and', 'co', 'company', 'services', 'enterprises', 'store', 'shop',
    'center', 'centre', 'group', 'india', 'us', 'usa', 'france'
}

class FastCountryBlocker:
    """
    High-speed, high-recall inverted-index blocker.
    Indexes exact normalized name, salient tokens, and 4-character prefix.
    """
    def __init__(self, target_df: pd.DataFrame, max_key_frequency: int = 300):
        self.max_key_freq = max_key_frequency
        
        t_ids = target_df['entity_id'].values
        t_names = target_df['norm_name'].values
        t_pfx4 = target_df['name_prefix_4'].values
        
        self.idx_exact_name = collections.defaultdict(list)
        self.idx_token = collections.defaultdict(list)
        self.idx_pfx4 = collections.defaultdict(list)
        
        for i in range(len(target_df)):
            eid = t_ids[i]
            nm = t_names[i]
            if not nm:
                continue
                
            # 1. Exact Name Index
            if len(self.idx_exact_name[nm]) < max_key_frequency:
                self.idx_exact_name[nm].append(eid)
                
            # 2. Salient Name Tokens (up to 3 tokens)
            tokens = [w for w in nm.split() if len(w) >= 3 and w not in STOPWORDS]
            for tk in tokens[:3]:
                lst = self.idx_token[tk]
                if len(lst) < max_key_frequency:
                    lst.append(eid)
                    
            # 3. 4-char Prefix Index
            p4 = t_pfx4[i]
            if p4 and len(p4) >= 4:
                lst_p = self.idx_pfx4[p4]
                if len(lst_p) < max_key_frequency:
                    lst_p.append(eid)

    def get_candidates(self, s1_df: pd.DataFrame, max_cands: int = 30) -> Dict[str, Set[str]]:
        """
        Retrieves candidate entity ID sets for S1 entities using multi-pass retrieval.
        """
        s1_ids = s1_df['entity_id'].values
        s1_names = s1_df['norm_name'].values
        s1_pfx4 = s1_df['name_prefix_4'].values
        
        candidates = {}
        
        for i in range(len(s1_df)):
            s1_id = s1_ids[i]
            cands = set()
            nm = s1_names[i]
            
            if nm:
                # Pass 1: Exact Name Match
                if nm in self.idx_exact_name:
                    cands.update(self.idx_exact_name[nm][:max_cands])
                    
                # Pass 2: Salient Tokens Match
                tokens = [w for w in nm.split() if len(w) >= 3 and w not in STOPWORDS]
                for tk in tokens[:3]:
                    if len(cands) >= max_cands:
                        break
                    if tk in self.idx_token:
                        cands.update(self.idx_token[tk][:(max_cands - len(cands))])
                        
                # Pass 3: 4-char Prefix Fallback if 0 candidates
                p4 = s1_pfx4[i]
                if len(cands) == 0 and p4 and p4 in self.idx_pfx4:
                    cands.update(self.idx_pfx4[p4][:5])
                    
            candidates[s1_id] = cands
            
        return candidates

def evaluate_candidate_recall(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    s1_ids: List[str]
) -> Dict[str, float]:
    """
    Measures candidate recall and statistics.
    """
    total_true_matches = 0
    captured_true_matches = 0
    total_candidates = 0
    s1_with_cands = 0
    
    for s1_id in s1_ids:
        true_set = ground_truth.get(s1_id, set())
        cand_set = candidates.get(s1_id, set())
        
        total_true_matches += len(true_set)
        captured_true_matches += len(true_set.intersection(cand_set))
        total_candidates += len(cand_set)
        if len(cand_set) > 0:
            s1_with_cands += 1
            
    recall = (captured_true_matches / total_true_matches) if total_true_matches > 0 else 1.0
    avg_cands = (total_candidates / len(s1_ids)) if len(s1_ids) > 0 else 0.0
    
    return {
        "candidate_recall": recall,
        "total_true_matches": total_true_matches,
        "captured_true_matches": captured_true_matches,
        "avg_candidates_per_s1": avg_cands,
        "total_candidate_pairs": total_candidates,
        "s1_coverage_pct": (s1_with_cands / len(s1_ids)) * 100 if len(s1_ids) > 0 else 0.0
    }
