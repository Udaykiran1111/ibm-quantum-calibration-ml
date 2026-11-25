def summarize_counts(counts):
    total = sum(counts.values())
    probabilities = {k: v / total for k, v in counts.items()}
    p00 = probabilities.get("00", 0.0)
    p11 = probabilities.get("11", 0.0)
    forbidden = probabilities.get("01", 0.0) + probabilities.get("10", 0.0)
    return {
        "counts": counts,
        "probabilities": probabilities,
        "p_00": p00,
        "p_11": p11,
        "forbidden_probability": forbidden,
        "bell_success_probability": p00 + p11,
    }
