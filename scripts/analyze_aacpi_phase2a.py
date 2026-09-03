from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, TextIO


ZERO_TOLERANCE = 1e-15
PAIR_QUERY_FIELDS = (
    "dataset",
    "pair_id",
    "expert_a",
    "expert_b",
    "query_id",
    "original_triple_id",
    "seed",
    "direction",
    "rr_a",
    "rr_b",
    "winner_label",
    "endpoint_preference",
    "endpoint_preference_exists",
    "endpoint_rr_gap_abs",
    "alpha0",
    "best_alpha",
    "best_delta_alpha",
    "best_direction",
    "best_advantage",
    "beneficial_deviation",
    "winner_direction",
    "winner_direction_matches_best",
    "winner_label_matches_positive_correction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AACPI V2 Phase 2A DEV-only diagnostics over existing utility tables. "
            "No model is trained and TEST inputs are rejected."
        )
    )
    parser.add_argument("--utility-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument("--output-dir", default="outputs/aacpi/phase2a_diagnostics")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_csv_index(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key}={value!r} in {path}")
        result[value] = row
    return result


def iter_query_actions(path: Path) -> Iterator[list[dict[str, str]]]:
    seen: set[str] = set()
    current_id = None
    current_rows: list[dict[str, str]] = []
    with open_text(path, "r") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "dev":
                raise RuntimeError(
                    f"AACPI Phase 2A is DEV-only; found split={row.get('split')!r} in {path}"
                )
            query_id = str(row["query_id"])
            if current_id is None:
                current_id = query_id
            if query_id != current_id:
                if current_id in seen:
                    raise ValueError(f"Non-contiguous duplicate query_id={current_id!r} in {path}")
                seen.add(current_id)
                yield current_rows
                current_id = query_id
                current_rows = []
            current_rows.append(row)
    if current_rows:
        if current_id in seen:
            raise ValueError(f"Non-contiguous duplicate query_id={current_id!r} in {path}")
        yield current_rows


def sign(value: float) -> str:
    if value > ZERO_TOLERANCE:
        return "positive"
    if value < -ZERO_TOLERANCE:
        return "negative"
    return "zero"


def direction_from_delta(delta: float) -> str:
    if delta > ZERO_TOLERANCE:
        return "toward_a"
    if delta < -ZERO_TOLERANCE:
        return "toward_b"
    return "stay"


def fmt_delta(value: float) -> str:
    return f"{value:+.2f}"


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def csv_value(value) -> object:
    return "" if value is None else value


def best_action(rows: list[dict[str, str]]) -> dict[str, float | str]:
    candidates = []
    anchor_values = set()
    alphas = set()
    for row in rows:
        alpha = float(row["alpha"])
        delta = float(row["delta_alpha"])
        rr = float(row["rr_action"])
        advantage = float(row["advantage"])
        anchor_values.add(float(row["rr_anchor"]))
        alphas.add(alpha)
        candidates.append((rr, -abs(delta), -alpha, alpha, delta, advantage))
    if len(anchor_values) != 1:
        raise ValueError(f"rr_anchor is inconsistent for query {rows[0]['query_id']}")
    winner = max(candidates)
    return {
        "alpha": winner[3],
        "delta": winner[4],
        "advantage": winner[5],
        "direction": direction_from_delta(winner[4]),
        "grid_min": min(alphas),
        "grid_max": max(alphas),
    }


def write_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: csv_value(row.get(key)) for key in fields} for row in rows)


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def signed_float(value: float | None) -> str:
    return "" if value is None else f"{value:+.6f}"


def pair_display(pair: dict) -> str:
    return f"{pair['dataset']} / {pair['expert_a']} + {pair['expert_b']}"


def action_landscape_svg(pair_summaries: list[dict], path: Path) -> None:
    columns, rows_n = 2, 3
    panel_w, panel_h = 520, 260
    width, height = columns * panel_w, rows_n * panel_h + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.panel{font-size:14px;font-weight:700}.axis{font-size:10px}.legend{font-size:12px}</style>',
        '<text x="24" y="30" class="title">AACPI V2 Phase 2A — DEV action utility signs</text>',
        '<rect x="24" y="46" width="12" height="12" fill="#2f855a"/><text x="42" y="57" class="legend">Positive U</text>',
        '<rect x="132" y="46" width="12" height="12" fill="#a0aec0"/><text x="150" y="57" class="legend">Zero U</text>',
        '<rect x="224" y="46" width="12" height="12" fill="#c53030"/><text x="242" y="57" class="legend">Negative U</text>',
    ]
    for index, pair in enumerate(pair_summaries):
        col, row_index = index % columns, index // columns
        ox, oy = col * panel_w + 28, row_index * panel_h + 88
        chart_x, chart_y, chart_w, chart_h = ox + 46, oy + 34, 430, 160
        label = html.escape(pair_display(pair))
        parts.append(f'<text x="{ox}" y="{oy + 14}" class="panel">{label}</text>')
        parts.append(f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" stroke="#5f6368"/>')
        parts.append(f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="#5f6368"/>')
        actions = pair["action_landscape"]
        gap = 5
        bar_w = (chart_w - gap * (len(actions) - 1)) / len(actions)
        for action_index, action in enumerate(actions):
            x = chart_x + action_index * (bar_w + gap)
            y = chart_y
            for field, color in (("positive_rate", "#2f855a"), ("zero_rate", "#a0aec0"), ("negative_rate", "#c53030")):
                h = chart_h * float(action[field])
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
                y += h
            parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{chart_y + chart_h + 14}" text-anchor="middle" class="axis">{html.escape(action["delta_alpha"])}</text>')
        for fraction in (0.0, 0.5, 1.0):
            y = chart_y + chart_h * (1.0 - fraction)
            parts.append(f'<text x="{chart_x - 6}" y="{y + 3}" text-anchor="end" class="axis">{int(100*fraction)}%</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def mean_utility_svg(pair_summaries: list[dict], path: Path) -> None:
    columns, rows_n = 2, 3
    panel_w, panel_h = 520, 250
    width, height = columns * panel_w, rows_n * panel_h + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.panel{font-size:14px;font-weight:700}.axis{font-size:10px}</style>',
        '<text x="24" y="30" class="title">AACPI V2 Phase 2A — mean DEV advantage by action</text>',
    ]
    for index, pair in enumerate(pair_summaries):
        col, row_index = index % columns, index // columns
        ox, oy = col * panel_w + 28, row_index * panel_h + 56
        chart_x, chart_y, chart_w, chart_h = ox + 58, oy + 34, 415, 150
        actions = pair["action_landscape"]
        values = [float(item["mean_u"]) for item in actions]
        scale = max(max(abs(value) for value in values), 1e-6)
        zero_y = chart_y + chart_h / 2
        parts.append(f'<text x="{ox}" y="{oy + 14}" class="panel">{html.escape(pair_display(pair))}</text>')
        parts.append(f'<line x1="{chart_x}" y1="{zero_y}" x2="{chart_x + chart_w}" y2="{zero_y}" stroke="#a0aec0" stroke-dasharray="4 3"/>')
        points = []
        for action_index, action in enumerate(actions):
            x = chart_x + (action_index + 0.5) * chart_w / len(actions)
            y = zero_y - float(action["mean_u"]) / scale * (chart_h * 0.44)
            points.append(f"{x:.2f},{y:.2f}")
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="#2b6cb0"/>')
            parts.append(f'<text x="{x:.2f}" y="{chart_y + chart_h + 14}" text-anchor="middle" class="axis">{html.escape(action["delta_alpha"])}</text>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#2b6cb0" stroke-width="2"/>')
        parts.append(f'<text x="{chart_x - 6}" y="{chart_y + 4}" text-anchor="end" class="axis">+{scale:.4f}</text>')
        parts.append(f'<text x="{chart_x - 6}" y="{zero_y + 4}" text-anchor="end" class="axis">0</text>')
        parts.append(f'<text x="{chart_x - 6}" y="{chart_y + chart_h}" text-anchor="end" class="axis">-{scale:.4f}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def confusion_svg(pair_summaries: list[dict], path: Path) -> None:
    columns, rows_n = 2, 3
    panel_w, panel_h = 520, 235
    width, height = columns * panel_w, rows_n * panel_h + 70
    winner_classes = ("toward_b_or_tie", "toward_a")
    best_classes = ("toward_b", "stay", "toward_a")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.panel{font-size:14px;font-weight:700}.cell{font-size:12px}.axis{font-size:11px}</style>',
        '<text x="24" y="30" class="title">Winner label vs best local correction direction (DEV)</text>',
    ]
    for index, pair in enumerate(pair_summaries):
        col, row_index = index % columns, index // columns
        ox, oy = col * panel_w + 30, row_index * panel_h + 60
        parts.append(f'<text x="{ox}" y="{oy + 14}" class="panel">{html.escape(pair_display(pair))}</text>')
        matrix = pair["winner_vs_best_direction_confusion"]
        cell_w, cell_h = 105, 55
        x0, y0 = ox + 145, oy + 50
        for c, best_class in enumerate(best_classes):
            parts.append(f'<text x="{x0 + c*cell_w + cell_w/2}" y="{y0 - 10}" text-anchor="middle" class="axis">{best_class.replace("toward_", "")}</text>')
        for r, winner_class in enumerate(winner_classes):
            total = sum(matrix[winner_class].values())
            parts.append(f'<text x="{x0 - 10}" y="{y0 + r*cell_h + cell_h/2 + 4}" text-anchor="end" class="axis">{winner_class.replace("toward_", "")}</text>')
            for c, best_class in enumerate(best_classes):
                count = matrix[winner_class][best_class]
                rate = count / total if total else 0.0
                shade = int(248 - 150 * rate)
                color = f"rgb({shade},{shade},{255})"
                x, y = x0 + c*cell_w, y0 + r*cell_h
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="#ffffff"/>')
                parts.append(f'<text x="{x + cell_w/2}" y="{y + 23}" text-anchor="middle" class="cell">{100*rate:.1f}%</text>')
                parts.append(f'<text x="{x + cell_w/2}" y="{y + 40}" text-anchor="middle" class="cell">n={count}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def analyze_pair(manifest_path: Path, output_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = read_json(manifest_path)
    if manifest.get("split") != "dev":
        raise RuntimeError(f"AACPI Phase 2A is DEV-only: {manifest_path}")
    table_path = Path(manifest["output_table"]["path"])
    source_path = Path(manifest["source_query_rows"]["path"])
    source = read_csv_index(source_path, "query_id")
    pair_id = str(manifest["pair_id"])
    per_delta: dict[float, dict[str, float | int]] = defaultdict(
        lambda: {
            "count": 0,
            "positive": 0,
            "zero": 0,
            "negative": 0,
            "sum": 0.0,
            "positive_sum": 0.0,
            "negative_sum": 0.0,
            "best_count": 0,
        }
    )
    counts: Counter[str] = Counter()
    direction_matrix: dict[str, Counter[str]] = {
        "toward_b_or_tie": Counter(),
        "toward_a": Counter(),
    }
    preference_opportunity_matrix: dict[str, Counter[str]] = {
        "no_endpoint_preference": Counter(),
        "endpoint_preference": Counter(),
    }
    exact_action_matrix: dict[str, Counter[str]] = {
        "winner_0": Counter(),
        "winner_1": Counter(),
    }
    query_output_path = output_dir / "query_diagnostics" / f"{pair_id}_dev_winner_action_diagnostics.csv.gz"
    query_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_text(query_output_path, "w") as query_handle:
        writer = csv.DictWriter(query_handle, fieldnames=PAIR_QUERY_FIELDS)
        writer.writeheader()
        seen_source = set()
        grid_min = grid_max = alpha0 = None
        for actions in iter_query_actions(table_path):
            first = actions[0]
            query_id = first["query_id"]
            if query_id not in source:
                raise ValueError(f"Utility query_id not found in source exact rows: {query_id}")
            exact = source[query_id]
            seen_source.add(query_id)
            rr_a, rr_b = float(exact["rr_a"]), float(exact["rr_b"])
            winner_label = int(rr_a > rr_b)
            endpoint_preference = "a" if rr_a > rr_b else "b" if rr_b > rr_a else "tie"
            preference_exists = endpoint_preference != "tie"
            winner_direction = "toward_a" if winner_label else "toward_b_or_tie"
            best = best_action(actions)
            beneficial = float(best["advantage"]) > ZERO_TOLERANCE
            best_direction = str(best["direction"])
            best_delta = float(best["delta"])
            alpha0 = float(first["alpha0"])
            grid_min, grid_max = float(best["grid_min"]), float(best["grid_max"])

            counts["queries"] += 1
            counts["endpoint_ties"] += int(not preference_exists)
            counts["beneficial_deviation"] += int(beneficial)
            direction_match = (
                (winner_label == 1 and best_direction == "toward_a")
                or (winner_label == 0 and best_direction == "toward_b")
            )
            counts["winner_direction_match_all"] += int(direction_match)
            counts["winner_label_matches_positive_correction"] += int(
                winner_label == int(best_delta > ZERO_TOLERANCE)
            )
            counts["preference_opportunity_match"] += int(preference_exists == beneficial)
            if beneficial:
                counts["opportunity_queries"] += 1
                counts["winner_direction_match_opportunity"] += int(direction_match)
                if preference_exists:
                    counts["untied_opportunity_queries"] += 1
                    counts["winner_direction_match_untied_opportunity"] += int(direction_match)
                    opposite = (
                        (winner_label == 1 and best_direction == "toward_b")
                        or (winner_label == 0 and best_direction == "toward_a")
                    )
                    counts["opposite_direction_untied_opportunity"] += int(opposite)
            if preference_exists:
                counts["endpoint_preference_queries"] += 1
                counts["stay_despite_endpoint_preference"] += int(best_direction == "stay")
            else:
                counts["beneficial_despite_endpoint_tie"] += int(beneficial)

            direction_matrix[winner_direction][best_direction] += 1
            preference_class = "endpoint_preference" if preference_exists else "no_endpoint_preference"
            opportunity_class = "worth_deviating" if beneficial else "stay_anchor"
            preference_opportunity_matrix[preference_class][opportunity_class] += 1
            exact_action_matrix[f"winner_{winner_label}"][fmt_delta(best_delta)] += 1

            for action in actions:
                delta = float(action["delta_alpha"])
                advantage = float(action["advantage"])
                category = sign(advantage)
                stats = per_delta[delta]
                stats["count"] = int(stats["count"]) + 1
                stats[category] = int(stats[category]) + 1
                stats["sum"] = float(stats["sum"]) + advantage
                if category == "positive":
                    stats["positive_sum"] = float(stats["positive_sum"]) + advantage
                elif category == "negative":
                    stats["negative_sum"] = float(stats["negative_sum"]) + advantage
            per_delta[best_delta]["best_count"] = int(per_delta[best_delta]["best_count"]) + 1

            writer.writerow(
                {
                    "dataset": manifest["dataset"],
                    "pair_id": pair_id,
                    "expert_a": exact["expert_a_name"],
                    "expert_b": exact["expert_b_name"],
                    "query_id": query_id,
                    "original_triple_id": first["original_triple_id"],
                    "seed": first["seed"],
                    "direction": first["direction"],
                    "rr_a": rr_a,
                    "rr_b": rr_b,
                    "winner_label": winner_label,
                    "endpoint_preference": endpoint_preference,
                    "endpoint_preference_exists": int(preference_exists),
                    "endpoint_rr_gap_abs": abs(rr_a - rr_b),
                    "alpha0": alpha0,
                    "best_alpha": best["alpha"],
                    "best_delta_alpha": best_delta,
                    "best_direction": best_direction,
                    "best_advantage": best["advantage"],
                    "beneficial_deviation": int(beneficial),
                    "winner_direction": winner_direction,
                    "winner_direction_matches_best": int(direction_match),
                    "winner_label_matches_positive_correction": int(
                        winner_label == int(best_delta > ZERO_TOLERANCE)
                    ),
                }
            )
    if seen_source != set(source):
        raise ValueError(
            f"Utility/source query mismatch for {pair_id}: utility={len(seen_source)}, source={len(source)}"
        )
    if alpha0 is None or grid_min is None or grid_max is None:
        raise RuntimeError(f"No utility queries found for {pair_id}")

    n_queries = counts["queries"]
    max_abs_delta = max(abs(delta) for delta in per_delta)
    landscape = []
    for delta, stats in sorted(per_delta.items()):
        count = int(stats["count"])
        positive = int(stats["positive"])
        negative = int(stats["negative"])
        alpha = round(alpha0 + delta, 10)
        landscape.append(
            {
                "dataset": manifest["dataset"],
                "pair_id": pair_id,
                "expert_a": source[next(iter(source))]["expert_a_name"],
                "expert_b": source[next(iter(source))]["expert_b_name"],
                "alpha0": alpha0,
                "alpha": alpha,
                "delta_alpha": fmt_delta(delta),
                "n": count,
                "positive_rate": positive / count,
                "zero_rate": int(stats["zero"]) / count,
                "negative_rate": negative / count,
                "mean_u": float(stats["sum"]) / count,
                "mean_positive_u": float(stats["positive_sum"]) / positive if positive else None,
                "mean_negative_u": float(stats["negative_sum"]) / negative if negative else None,
                "best_action_count": int(stats["best_count"]),
                "best_action_rate": int(stats["best_count"]) / n_queries,
                "is_grid_endpoint": int(
                    math.isclose(alpha, grid_min, abs_tol=1e-12)
                    or math.isclose(alpha, grid_max, abs_tol=1e-12)
                ),
                "is_max_radius_boundary": int(
                    math.isclose(abs(delta), max_abs_delta, rel_tol=0.0, abs_tol=1e-12)
                ),
            }
        )

    direction_confusion = {
        winner: {best: direction_matrix[winner][best] for best in ("toward_b", "stay", "toward_a")}
        for winner in ("toward_b_or_tie", "toward_a")
    }
    preference_confusion = {
        preference: {
            outcome: preference_opportunity_matrix[preference][outcome]
            for outcome in ("stay_anchor", "worth_deviating")
        }
        for preference in ("no_endpoint_preference", "endpoint_preference")
    }
    best_delta_distribution = {
        item["delta_alpha"]: {
            "count": item["best_action_count"],
            "rate": item["best_action_rate"],
        }
        for item in landscape
        if item["best_action_count"]
    }
    lower_boundary = landscape[0]
    upper_boundary = landscape[-1]
    pair_summary = {
        "dataset": manifest["dataset"],
        "pair_id": pair_id,
        "expert_a": landscape[0]["expert_a"],
        "expert_b": landscape[0]["expert_b"],
        "alpha0": alpha0,
        "n_queries": n_queries,
        "winner_label_definition": "1[RR_A > RR_B]; endpoint ties map to 0",
        "endpoint_tie_rate": counts["endpoint_ties"] / n_queries,
        "beneficial_deviation_rate": counts["beneficial_deviation"] / n_queries,
        "winner_direction_agreement_all_queries": counts["winner_direction_match_all"] / n_queries,
        "winner_direction_agreement_on_beneficial_deviations": safe_rate(
            counts["winner_direction_match_opportunity"], counts["opportunity_queries"]
        ),
        "winner_direction_agreement_on_untied_beneficial_deviations": safe_rate(
            counts["winner_direction_match_untied_opportunity"],
            counts["untied_opportunity_queries"],
        ),
        "opposite_direction_rate_on_untied_beneficial_deviations": safe_rate(
            counts["opposite_direction_untied_opportunity"],
            counts["untied_opportunity_queries"],
        ),
        "winner_label_vs_positive_correction_binary_agreement": (
            counts["winner_label_matches_positive_correction"] / n_queries
        ),
        "endpoint_preference_exists_vs_worth_deviating_agreement": (
            counts["preference_opportunity_match"] / n_queries
        ),
        "stay_despite_endpoint_preference_rate": safe_rate(
            counts["stay_despite_endpoint_preference"], counts["endpoint_preference_queries"]
        ),
        "beneficial_deviation_despite_endpoint_tie_rate": safe_rate(
            counts["beneficial_despite_endpoint_tie"], counts["endpoint_ties"]
        ),
        "winner_vs_best_direction_confusion": direction_confusion,
        "endpoint_preference_vs_opportunity_confusion": preference_confusion,
        "winner_vs_best_action_class_confusion": {
            winner: dict(sorted(values.items(), key=lambda item: float(item[0])))
            for winner, values in exact_action_matrix.items()
        },
        "best_delta_alpha_distribution": best_delta_distribution,
        "max_abs_delta": max_abs_delta,
        "max_radius_boundary_best_rate": sum(
            item["best_action_rate"] for item in landscape if item["is_max_radius_boundary"]
        ),
        "lower_grid_boundary": {
            "alpha": lower_boundary["alpha"],
            "delta_alpha": lower_boundary["delta_alpha"],
            "best_action_rate": lower_boundary["best_action_rate"],
        },
        "upper_grid_boundary": {
            "alpha": upper_boundary["alpha"],
            "delta_alpha": upper_boundary["delta_alpha"],
            "best_action_rate": upper_boundary["best_action_rate"],
        },
        "action_landscape": landscape,
        "query_diagnostics": portable_path(query_output_path),
    }

    confusion_rows = []
    for winner, values in direction_confusion.items():
        total = sum(values.values())
        for best, count in values.items():
            confusion_rows.append(
                {
                    "dataset": manifest["dataset"],
                    "pair_id": pair_id,
                    "matrix": "winner_label_vs_best_direction",
                    "row_class": winner,
                    "column_class": best,
                    "count": count,
                    "row_rate": count / total if total else None,
                }
            )
    for preference, values in preference_confusion.items():
        total = sum(values.values())
        for outcome, count in values.items():
            confusion_rows.append(
                {
                    "dataset": manifest["dataset"],
                    "pair_id": pair_id,
                    "matrix": "endpoint_preference_exists_vs_worth_deviating",
                    "row_class": preference,
                    "column_class": outcome,
                    "count": count,
                    "row_rate": count / total if total else None,
                }
            )
    for winner, values in exact_action_matrix.items():
        total = sum(values.values())
        for delta, count in sorted(values.items(), key=lambda item: float(item[0])):
            confusion_rows.append(
                {
                    "dataset": manifest["dataset"],
                    "pair_id": pair_id,
                    "matrix": "winner_label_vs_best_action_class",
                    "row_class": winner,
                    "column_class": delta,
                    "count": count,
                    "row_rate": count / total if total else None,
                }
            )
    return pair_summary, landscape, confusion_rows


def write_report(path: Path, summaries: list[dict], landscape: list[dict]) -> None:
    lines = [
        "# AACPI V2 Phase 2A DEV Diagnostics",
        "",
        "This report is descriptive and DEV-only. No predictor or policy was trained.",
        "",
        "## Definitions",
        "",
        "- `winner_label = 1[RR_A > RR_B]`; endpoint ties map to zero exactly as specified.",
        "- Winner direction is toward A for label 1 and toward B-or-tie for label 0.",
        "- Best action maximizes actual local-action RR, then prefers smaller absolute deviation and then smaller alpha.",
        "- A query is worth deviating when at least one frozen local action has strictly positive advantage.",
        "- Because the binary winner label has no stay class, the deviation comparison uses `RR_A != RR_B` as its implied endpoint-preference signal. The exact 2x3 and action-class confusion matrices are exported separately.",
        "",
        "## Winner label versus mixture action",
        "",
        "| Dataset / pair | Endpoint ties | Worth deviating | Direction agreement on beneficial actions | Untied direction agreement | Opposite direction on untied beneficial actions | Endpoint preference vs deviation agreement | Stay despite endpoint preference |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {pair_display(item)} | {percent(item['endpoint_tie_rate'])} | "
            f"{percent(item['beneficial_deviation_rate'])} | "
            f"{percent(item['winner_direction_agreement_on_beneficial_deviations'])} | "
            f"{percent(item['winner_direction_agreement_on_untied_beneficial_deviations'])} | "
            f"{percent(item['opposite_direction_rate_on_untied_beneficial_deviations'])} | "
            f"{percent(item['endpoint_preference_exists_vs_worth_deviating_agreement'])} | "
            f"{percent(item['stay_despite_endpoint_preference_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Action-wise utility landscape",
            "",
            "| Dataset / pair | Delta alpha | Positive U | Zero U | Negative U | Mean U | Mean positive U | Mean negative U | Best action | Max-radius boundary |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in landscape:
        lines.append(
            f"| {item['dataset']} / {item['expert_a']} + {item['expert_b']} | {item['delta_alpha']} | "
            f"{percent(item['positive_rate'])} | {percent(item['zero_rate'])} | "
            f"{percent(item['negative_rate'])} | {item['mean_u']:+.6f} | "
            f"{signed_float(item['mean_positive_u'])} | "
            f"{signed_float(item['mean_negative_u'])} | "
            f"{percent(item['best_action_rate'])} | {'yes' if item['is_max_radius_boundary'] else ''} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "High zero rates confirm a stepped RR surface, so later regression MAE cannot be the main scientific criterion. Boundary-best rates are descriptive evidence that some query utilities continue improving at the frozen radius; the Phase 2 action grid remains unchanged.",
            "",
            "The NativE + AdaMF-MAT rows are retained as the primary falsification diagnostic. These results assess target alignment only and do not establish advantage learnability.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    utility_dir = Path(args.utility_dir)
    output_dir = Path(args.output_dir)
    manifests = sorted(utility_dir.glob("*_dev_source_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No DEV source manifests found under {utility_dir}")
    output_paths = (
        output_dir / "phase2a_summary.json",
        output_dir / "winner_action_alignment.csv",
        output_dir / "winner_action_confusion.csv",
        output_dir / "action_utility_landscape.csv",
        output_dir / "phase2a_diagnostics.md",
        output_dir / "action_utility_sign_rates.svg",
        output_dir / "action_mean_advantage.svg",
        output_dir / "winner_best_direction_confusion.svg",
    )
    if not args.overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, landscape_rows, confusion_rows = [], [], []
    for manifest in manifests:
        summary, landscape, confusion = analyze_pair(manifest, output_dir)
        summaries.append(summary)
        landscape_rows.extend(landscape)
        confusion_rows.extend(confusion)
    summaries.sort(key=lambda item: (item["dataset"], item["pair_id"]))
    landscape_rows.sort(key=lambda item: (item["dataset"], item["pair_id"], float(item["delta_alpha"])))

    alignment_fields = [
        key
        for key in summaries[0]
        if key
        not in {
            "winner_vs_best_direction_confusion",
            "endpoint_preference_vs_opportunity_confusion",
            "winner_vs_best_action_class_confusion",
            "best_delta_alpha_distribution",
            "lower_grid_boundary",
            "upper_grid_boundary",
            "action_landscape",
        }
    ]
    write_csv(output_dir / "winner_action_alignment.csv", alignment_fields, summaries)
    write_csv(
        output_dir / "winner_action_confusion.csv",
        ["dataset", "pair_id", "matrix", "row_class", "column_class", "count", "row_rate"],
        confusion_rows,
    )
    write_csv(
        output_dir / "action_utility_landscape.csv",
        list(landscape_rows[0]),
        landscape_rows,
    )
    payload = {
        "schema_version": 1,
        "phase": "AACPI V2 Phase 2A",
        "split": "dev",
        "model_training_performed": False,
        "test_accessed": False,
        "definitions": {
            "winner_label": "1[RR_A > RR_B]; endpoint ties map to 0",
            "best_action": "max actual RR; tie: min abs(delta_alpha), then smaller alpha",
            "worth_deviating": "max local advantage > 0",
            "deviation_proxy_for_winner_target": "endpoint preference exists iff RR_A != RR_B",
        },
        "pairs": summaries,
    }
    (output_dir / "phase2a_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "phase2a_diagnostics.md", summaries, landscape_rows)
    action_landscape_svg(summaries, output_dir / "action_utility_sign_rates.svg")
    mean_utility_svg(summaries, output_dir / "action_mean_advantage.svg")
    confusion_svg(summaries, output_dir / "winner_best_direction_confusion.svg")
    print(f"[OK] analyzed {len(summaries)} DEV pairs -> {output_dir}")
    for item in summaries:
        print(
            f"[2A] {item['pair_id']} opportunity={item['beneficial_deviation_rate']:.6f} "
            f"direction_agreement={item['winner_direction_agreement_on_beneficial_deviations']:.6f} "
            f"opposite_untied={item['opposite_direction_rate_on_untied_beneficial_deviations']:.6f}"
        )


if __name__ == "__main__":
    main()
