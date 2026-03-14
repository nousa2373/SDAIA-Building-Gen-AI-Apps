import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def jaccard_similarity(s1: str, s2: str) -> float:
    """Calculate the Jaccard similarity between two strings."""
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)
