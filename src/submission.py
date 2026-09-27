import os
import pandas as pd
from typing import Dict, List, Set

def export_submission_files(
    matching_dict: Dict[str, Set[str]],
    candidate_dict: Dict[str, Set[str]],
    all_test_s1_ids: List[str],
    output_dir: str
):
    """
    Exports matching_results.tsv and candidate_pairs.tsv strictly formatted for submission.
    Ensures every test S1 entity is present, empty matches are handled cleanly,
    and final matches are a subset of candidate pairs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    # 1. Write matching_results.tsv
    print(f"Writing matching results to {matching_path}...", flush=True)
    with open(matching_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_test_s1_ids:
            matches = sorted(list(matching_dict.get(s1_id, set())))
            # Clean: filter out any invalid S1 IDs or self-matches
            valid_matches = [m for m in matches if m.startswith(('S2-', 'S3-')) and m != s1_id]
            match_str = ",".join(valid_matches)
            f.write(f"{s1_id}\t{match_str}\n")
            
    # 2. Write candidate_pairs.tsv
    print(f"Writing candidate pairs to {candidate_path}...", flush=True)
    with open(candidate_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_test_s1_ids:
            cands = set(candidate_dict.get(s1_id, set()))
            # Ensure all final matches are included in candidates (FINAL_MATCHES ⊆ CANDIDATES)
            final_matches = matching_dict.get(s1_id, set())
            cands.update(final_matches)
            
            valid_cands = sorted([c for c in cands if c.startswith(('S2-', 'S3-')) and c != s1_id])
            cand_str = ",".join(valid_cands)
            f.write(f"{s1_id}\t{cand_str}\n")
            
    print(f"Submission files successfully generated in {output_dir}!", flush=True)
