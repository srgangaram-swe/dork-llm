"""Small, dependency-light statistical estimators for model evaluation.

The functions in this module intentionally return estimates and confidence
intervals rather than binary "significance" decisions.  They are deterministic
for a fixed seed and preserve pairing when resampling benchmark observations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

PairedStatistic = Literal["mean_delta", "ratio_of_means"]

_BOOTSTRAP_METHOD = "paired_percentile_bootstrap"
_MAX_INDEX_ELEMENTS = 1_000_000


@dataclass(frozen=True, slots=True)
class PairedBootstrapEstimate:
    """A point estimate and percentile interval from paired resampling.

    ``seed`` and ``n_resamples`` are retained in the result so reports contain
    enough provenance to reproduce the Monte Carlo estimate.
    """

    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_pairs: int
    n_resamples: int
    seed: int
    statistic: PairedStatistic
    method: str = _BOOTSTRAP_METHOD

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BinomialEstimate:
    """A binomial proportion and Wilson score confidence interval."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    successes: int
    trials: int
    method: str = "wilson_score"

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _validate_confidence_level(confidence_level: float) -> float:
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not math.isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence_level must be a finite number strictly between 0 and 1")
    return float(confidence_level)


def _validate_positive_integer(value: int, *, name: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _as_finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f"{name} must contain only real numeric values")
    vector = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def paired_bootstrap(
    first: ArrayLike,
    second: ArrayLike,
    *,
    statistic: PairedStatistic = "mean_delta",
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapEstimate:
    """Estimate a paired mean delta or ratio of means with a percentile CI.

    Pair ``i`` in ``first`` must correspond to pair ``i`` in ``second``.  For
    ``mean_delta``, the estimand is ``mean(first - second)``.  For
    ``ratio_of_means``, it is ``mean(first) / mean(second)`` and every value in
    ``second`` must be strictly positive.  Bootstrap indices resample whole
    pairs, preserving within-pair dependence.

    The percentile interval measures sampling uncertainty under the empirical
    paired distribution.  It is not, by itself, a hypothesis test.
    """
    first_values = _as_finite_vector(first, name="first")
    second_values = _as_finite_vector(second, name="second")
    if first_values.size != second_values.size:
        raise ValueError("first and second must contain the same number of paired observations")
    if statistic not in ("mean_delta", "ratio_of_means"):
        raise ValueError("statistic must be 'mean_delta' or 'ratio_of_means'")

    confidence = _validate_confidence_level(confidence_level)
    resamples = _validate_positive_integer(n_resamples, name="n_resamples")
    bootstrap_seed = _validate_positive_integer(seed, name="seed", allow_zero=True)
    n_pairs = int(first_values.size)

    if statistic == "mean_delta":
        paired_values = first_values - second_values
        estimate = float(np.mean(paired_values))
    else:
        if np.any(second_values <= 0.0):
            raise ValueError("second must be strictly positive for ratio_of_means")
        estimate = float(np.mean(first_values) / np.mean(second_values))

    # Generate indices in bounded batches.  This keeps peak memory predictable
    # for larger evaluation sets without changing the seeded random stream.
    batch_size = max(1, min(resamples, _MAX_INDEX_ELEMENTS // n_pairs))
    bootstrap_statistics = np.empty(resamples, dtype=np.float64)
    generator = np.random.default_rng(bootstrap_seed)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = generator.integers(0, n_pairs, size=(stop - start, n_pairs))
        if statistic == "mean_delta":
            bootstrap_statistics[start:stop] = np.mean(paired_values[indices], axis=1)
        else:
            numerator = np.mean(first_values[indices], axis=1)
            denominator = np.mean(second_values[indices], axis=1)
            bootstrap_statistics[start:stop] = numerator / denominator

    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(bootstrap_statistics, [alpha, 1.0 - alpha])
    return PairedBootstrapEstimate(
        estimate=estimate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=confidence,
        n_pairs=n_pairs,
        n_resamples=resamples,
        seed=bootstrap_seed,
        statistic=statistic,
    )


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> BinomialEstimate:
    """Return a Wilson score interval for a binomial proportion.

    Wilson intervals remain informative at zero or ``trials`` successes, unlike
    the normal/Wald interval.  ``trials`` must be positive because an observed
    proportion is undefined for an empty experiment.
    """
    n = _validate_positive_integer(trials, name="trials")
    k = _validate_positive_integer(successes, name="successes", allow_zero=True)
    if k > n:
        raise ValueError("successes must not exceed trials")
    confidence = _validate_confidence_level(confidence_level)

    proportion = k / n
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (proportion + z_squared / (2.0 * n)) / denominator
    half_width = (
        z * math.sqrt(proportion * (1.0 - proportion) / n + z_squared / (4.0 * n * n)) / denominator
    )
    return BinomialEstimate(
        estimate=proportion,
        ci_low=0.0 if k == 0 else max(0.0, center - half_width),
        ci_high=1.0 if k == n else min(1.0, center + half_width),
        confidence_level=confidence,
        successes=k,
        trials=n,
    )
