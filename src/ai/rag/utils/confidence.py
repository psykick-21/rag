from typing import List, Literal

def compute_confidence(scores: List[float]) -> Literal["low", "medium", "high"]:
    """Computes the confidence level based on the scores."""

    sorted_scores = sorted(scores, reverse=False)

    d1 = sorted_scores[0]
    d2 = sorted_scores[1] if len(sorted_scores) > 1 else None
    avg_d = sum(sorted_scores) / len(sorted_scores)

    if d1 > 0.45:
        return "low"

    if d2 and (d2 - d1) > 0.08 and d1 < 0.30:
        return "high"

    return "medium"