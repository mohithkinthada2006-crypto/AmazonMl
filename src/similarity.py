import math
import rapidfuzz
from rapidfuzz import fuzz, distance

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculates Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union > 0 else 0.0

def char_ngram_set(text: str, n: int = 3) -> set:
    """Extracts character n-grams from text."""
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def token_set(text: str) -> set:
    """Splits string into word token set."""
    return set(text.split()) if text else set()

def compute_string_similarities(s1: str, s2: str, prefix: str = "name") -> dict:
    """
    Computes a comprehensive suite of lexical, token, and edit-distance similarities
    between two strings.
    """
    s1 = s1 if isinstance(s1, str) else ""
    s2 = s2 if isinstance(s2, str) else ""
    
    # Exact matches
    exact_match = 1.0 if s1 == s2 and s1 != "" else 0.0
    
    # RapidFuzz metrics (scaled to 0.0 - 1.0)
    ratio = fuzz.ratio(s1, s2) / 100.0
    partial_ratio = fuzz.partial_ratio(s1, s2) / 100.0
    token_sort_ratio = fuzz.token_sort_ratio(s1, s2) / 100.0
    token_set_ratio = fuzz.token_set_ratio(s1, s2) / 100.0
    
    # Token Jaccard
    tokens_1 = token_set(s1)
    tokens_2 = token_set(s2)
    tok_jaccard = jaccard_similarity(tokens_1, tokens_2)
    
    # Char 3-gram Jaccard
    ngrams_1 = char_ngram_set(s1, 3)
    ngrams_2 = char_ngram_set(s2, 3)
    ngram_jaccard = jaccard_similarity(ngrams_1, ngrams_2)
    
    # Length features
    len1 = len(s1)
    len2 = len(s2)
    len_diff = abs(len1 - len2)
    len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0
    
    # Prefix match
    prefix_match = 1.0 if s1 and s2 and s1[:4] == s2[:4] else 0.0
    
    return {
        f"{prefix}_exact": exact_match,
        f"{prefix}_ratio": ratio,
        f"{prefix}_partial_ratio": partial_ratio,
        f"{prefix}_token_sort_ratio": token_sort_ratio,
        f"{prefix}_token_set_ratio": token_set_ratio,
        f"{prefix}_tok_jaccard": tok_jaccard,
        f"{prefix}_ngram_jaccard": ngram_jaccard,
        f"{prefix}_len_diff": len_diff,
        f"{prefix}_len_ratio": len_ratio,
        f"{prefix}_prefix_match": prefix_match
    }
