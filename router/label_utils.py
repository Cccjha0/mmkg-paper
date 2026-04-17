from __future__ import annotations

from collections import Counter, defaultdict


def compute_delta_rr(rr_fusion: float, rr_struct: float) -> float:
    return float(rr_fusion) - float(rr_struct)


def build_binary_gain_label(delta_rr: float, delta: float) -> int:
    return int(float(delta_rr) > float(delta))


def summarize_gain_distribution(rows: list[dict]) -> dict:
    n_total = len(rows)
    n_positive = sum(int(row["label_gain"]) for row in rows)
    by_regime_counter: dict[str, Counter] = defaultdict(Counter)
    by_seed_counter: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        regime = str(row["target_regime"])
        seed = str(row["seed"])
        label = int(row["label_gain"])
        by_regime_counter[regime]["total"] += 1
        by_regime_counter[regime]["positive"] += label
        by_seed_counter[seed]["total"] += 1
        by_seed_counter[seed]["positive"] += label

    def to_rate(counter: Counter) -> float:
        total = int(counter["total"])
        return float(counter["positive"] / total) if total else 0.0

    return {
        "n_total": n_total,
        "n_positive": n_positive,
        "positive_rate": float(n_positive / n_total) if n_total else 0.0,
        "by_regime": {
            regime: {
                "n_total": int(counter["total"]),
                "n_positive": int(counter["positive"]),
                "positive_rate": to_rate(counter),
            }
            for regime, counter in sorted(by_regime_counter.items())
        },
        "by_seed": {
            seed: {
                "n_total": int(counter["total"]),
                "n_positive": int(counter["positive"]),
                "positive_rate": to_rate(counter),
            }
            for seed, counter in sorted(by_seed_counter.items(), key=lambda item: int(item[0]))
        },
    }

