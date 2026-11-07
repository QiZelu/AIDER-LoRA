"""Dependency-light regression and rank-correlation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _tie_pairs(values: np.ndarray) -> int:
    if len(values) < 2:
        return 0
    counts = np.unique(values, return_counts=True)[1].astype(np.int64)
    return int(np.sum(counts * (counts - 1) // 2))


def _kendall_tau_b(first: np.ndarray, second: np.ndarray) -> float:
    """Compute Kendall tau-b in O(N log N), including tie correction."""
    size = len(first)
    if size < 2:
        return float("nan")

    total_pairs = size * (size - 1) // 2
    first_ties = _tie_pairs(first)
    second_ties = _tie_pairs(second)
    joint_ties = _tie_pairs(
        np.rec.fromarrays([first, second], names=("first", "second"))
    )
    denominator = np.sqrt(
        float(total_pairs - first_ties) * float(total_pairs - second_ties)
    )
    if denominator == 0.0:
        return float("nan")

    order = np.lexsort((second, first))
    ordered_first = first[order]
    _, second_ranks = np.unique(second, return_inverse=True)
    ordered_second_ranks = second_ranks[order]

    # Fenwick tree over second-value ranks. Values from an equal-first group
    # are inserted only after the whole group is queried, so first ties are
    # never counted as discordant.
    rank_count = int(second_ranks.max()) + 1
    tree = np.zeros(rank_count + 1, dtype=np.int64)

    def prefix_count(index: int) -> int:
        count = 0
        while index > 0:
            count += int(tree[index])
            index -= index & -index
        return count

    def insert(index: int) -> None:
        while index <= rank_count:
            tree[index] += 1
            index += index & -index

    discordant = 0
    processed = 0
    start = 0
    while start < size:
        end = start + 1
        while end < size and ordered_first[end] == ordered_first[start]:
            end += 1
        for rank in ordered_second_ranks[start:end]:
            discordant += processed - prefix_count(int(rank) + 1)
        for rank in ordered_second_ranks[start:end]:
            insert(int(rank) + 1)
        processed += end - start
        start = end

    comparable_pairs = total_pairs - first_ties - second_ties + joint_ties
    concordant = comparable_pairs - discordant
    return float((concordant - discordant) / denominator)


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    dimensions: list[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    aggregate: dict[str, list[float]] = {
        "plcc": [],
        "srocc": [],
        "krocc": [],
        "rmse": [],
    }
    for index, dimension in enumerate(dimensions):
        valid = np.isfinite(targets[:, index]) & np.isfinite(
            predictions[:, index]
        )
        predicted = predictions[valid, index]
        expected = targets[valid, index]
        plcc = _correlation(predicted, expected)
        srocc = _correlation(_rankdata(predicted), _rankdata(expected))
        krocc = _kendall_tau_b(predicted, expected)
        rmse = (
            float(np.sqrt(np.mean(np.square(predicted - expected))))
            if len(predicted)
            else float("nan")
        )
        results[dimension] = {
            "count": int(valid.sum()),
            "plcc": plcc,
            "srocc": srocc,
            "krocc": krocc,
            "rmse": rmse,
            # Backward-compatible spelling used by earlier checkpoints/logs.
            "srcc": srocc,
        }
        for name, value in (
            ("plcc", plcc),
            ("srocc", srocc),
            ("krocc", krocc),
            ("rmse", rmse),
        ):
            if np.isfinite(value):
                aggregate[name].append(value)
    for name, values in aggregate.items():
        results[f"mean_{name}"] = (
            float(np.mean(values)) if values else float("nan")
        )
    results["mean_srcc"] = results["mean_srocc"]
    return results
