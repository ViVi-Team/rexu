from __future__ import annotations

from math import comb, erf, sqrt
from typing import Any


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> dict[str, Any]:
    """Wilson score interval for a binomial proportion.

    No scipy/statsmodels dependency: the normal quantile for the two standard
    confidence levels used in practice (0.95, 0.99) is hardcoded; other levels
    raise rather than silently approximate.
    """
    z_by_confidence = {0.90: 1.6448536269514722, 0.95: 1.9599639845400545, 0.99: 2.5758293035489004}
    if confidence not in z_by_confidence:
        raise ValueError(f"Unsupported confidence level {confidence!r}; supported: {sorted(z_by_confidence)}")
    if n <= 0:
        return {"point": None, "low": None, "high": None, "n": 0, "successes": successes, "confidence": confidence}
    z = z_by_confidence[confidence]
    p = successes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    low = (center - spread) / denom
    high = (center + spread) / denom
    return {
        "point": p,
        "low": max(0.0, low),
        "high": min(1.0, high),
        "n": n,
        "successes": successes,
        "confidence": confidence,
    }


def _binom_two_sided_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (sum of tail probabilities <= P(k))."""
    if n == 0:
        return 1.0
    point_probs = [comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    threshold = point_probs[k] * (1 + 1e-9)
    return min(1.0, sum(prob for prob in point_probs if prob <= threshold))


def _normal_sf(x: float) -> float:
    """Survival function (1 - CDF) of the standard normal distribution."""
    return 0.5 * (1.0 - erf(x / sqrt(2.0)))


def mcnemar(b: int, c: int, exact_threshold: int = 25) -> dict[str, Any]:
    """McNemar's test on a 2x2 paired contingency table's discordant cells.

    ``b`` = count of (condition1=1, condition2=0) pairs, ``c`` = count of
    (condition1=0, condition2=1) pairs. Concordant pairs don't affect the
    statistic and aren't passed in. Uses the exact binomial test when
    ``b + c < exact_threshold`` (standard McNemar convention for small
    samples), otherwise the chi-square approximation with continuity
    correction. No scipy dependency.
    """
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "statistic": 0.0, "p_value": 1.0, "method": "degenerate"}
    if n_discordant < exact_threshold:
        p_value = _binom_two_sided_pvalue(min(b, c), n_discordant, 0.5)
        return {"b": b, "c": c, "n_discordant": n_discordant, "statistic": float(min(b, c)), "p_value": p_value, "method": "exact_binomial"}
    chi2 = ((abs(b - c) - 1) ** 2) / n_discordant
    p_value = 2.0 * _normal_sf(sqrt(chi2))
    return {"b": b, "c": c, "n_discordant": n_discordant, "statistic": chi2, "p_value": min(1.0, p_value), "method": "chi2_continuity_corrected"}
