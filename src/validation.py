import numpy as np
from typing import Dict, List, Set, Tuple

def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes precision, recall, and F0.5 for a single Source 1 entity.
    Returns (precision, recall, f05).
    """
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0, 1.0, 1.0
    if len(true_set) == 0 and len(pred_set) > 0:
        return 0.0, 1.0, 0.0
    if len(true_set) > 0 and len(pred_set) == 0:
        return 1.0, 0.0, 0.0
        
    tp = len(true_set.intersection(pred_set))
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    
    denom = 0.25 * precision + recall
    if denom == 0 or (precision == 0 and recall == 0):
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom
        
    return precision, recall, f05

def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_ids: List[str]
) -> Dict[str, float]:
    """
    Computes macro-averaged evaluation metrics across all Source 1 entities:
      - macro_precision
      - macro_recall
      - macro_f05
      - singleton_accuracy
      - total_s1
      - total_predicted_links
      - singleton_predictions_count
    """
    precisions = []
    recalls = []
    f05_scores = []
    
    singleton_correct = 0
    total_singletons_true = 0
    singleton_preds_count = 0
    total_predicted_links = 0
    
    for s1_id in s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        
        p, r, f = compute_entity_f05(true_set, pred_set)
        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f)
        
        total_predicted_links += len(pred_set)
        if len(pred_set) == 0:
            singleton_preds_count += 1
            
        if len(true_set) == 0:
            total_singletons_true += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                
    singleton_acc = (singleton_correct / total_singletons_true) if total_singletons_true > 0 else 1.0
    
    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f05": float(np.mean(f05_scores)),
        "singleton_accuracy": float(singleton_acc),
        "total_s1": len(s1_ids),
        "total_predicted_links": total_predicted_links,
        "singleton_predictions_count": singleton_preds_count
    }

def find_best_threshold(
    val_probs: np.ndarray,
    val_pairs: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    val_s1_ids: List[str],
    thresholds: List[float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Searches across candidate probability thresholds to maximize validation Macro F0.5.
    """
    if thresholds is None:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        
    best_thresh = 0.50
    best_metrics = None
    best_f05 = -1.0
    
    # Pre-organize probabilities by s1_id for speed
    s1_cand_probs = collections_defaultdict = {}
    for i, (s1_id, cand_id) in enumerate(val_pairs):
        prob = val_probs[i]
        if s1_id not in s1_cand_probs:
            s1_cand_probs[s1_id] = []
        s1_cand_probs[s1_id].append((cand_id, prob))
        
    print("\n--- Threshold Optimization Sweep ---", flush=True)
    print(f"{'Threshold':<10} | {'Macro Precision':<16} | {'Macro Recall':<14} | {'Macro F0.5':<12} | {'Links':<10} | {'Singletons':<10}", flush=True)
    print("-" * 80, flush=True)
    
    for thresh in thresholds:
        preds = {}
        for s1_id in val_s1_ids:
            cand_list = s1_cand_probs.get(s1_id, [])
            matched = {cand_id for cand_id, prob in cand_list if prob >= thresh}
            preds[s1_id] = matched
            
        metrics = evaluate_predictions(ground_truth, preds, val_s1_ids)
        print(f"{thresh:<10.2f} | {metrics['macro_precision']:<16.4f} | {metrics['macro_recall']:<14.4f} | {metrics['macro_f05']:<12.4f} | {metrics['total_predicted_links']:<10} | {metrics['singleton_predictions_count']:<10}", flush=True)
        
        if metrics['macro_f05'] > best_f05:
            best_f05 = metrics['macro_f05']
            best_thresh = thresh
            best_metrics = metrics
            
    print("-" * 80, flush=True)
    print(f"Optimal Threshold: {best_thresh:.2f} with Macro F0.5 = {best_f05:.4f}\n", flush=True)
    return best_thresh, best_metrics
