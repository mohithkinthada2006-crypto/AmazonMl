import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from lightgbm import LGBMClassifier

def predict_matches(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    test_pairs: List[Tuple[str, str]],
    all_s1_ids: List[str],
    threshold: float = 0.50
) -> Dict[str, Set[str]]:
    """
    Scores all candidate pairs with the trained model and applies the optimized threshold
    to produce entity-level match predictions.
    Guarantees every S1 entity has an entry (empty set if no candidate exceeds threshold).
    """
    predictions = {s1_id: set() for s1_id in all_s1_ids}
    
    if len(X_test) == 0:
        return predictions
        
    print(f"Scoring {len(X_test):,} candidate pairs...", flush=True)
    probs = model.predict_proba(X_test)[:, 1]
    
    for i, (s1_id, cand_id) in enumerate(test_pairs):
        if probs[i] >= threshold:
            if s1_id in predictions:
                predictions[s1_id].add(cand_id)
            else:
                predictions[s1_id] = {cand_id}
                
    return predictions
