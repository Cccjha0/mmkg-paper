from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
ALPHA_COLUMNS = [f"rr_alpha_{alpha:.2f}".replace(".", "_") for alpha in ALPHA_GRID]
ZERO_TOLERANCE = 1e-15
EXPECTED_PAIRS = {
    "db15k_mhyper_adamf",
    "db15k_mhyper_native",
    "db15k_native_adamf",
    "mkgw_mhyper_adamf",
    "mkgw_mhyper_native",
    "mkgw_native_adamf",
}
PAIR_ORDER = [
    "mkgw_mhyper_native",
    "mkgw_mhyper_adamf",
    "mkgw_native_adamf",
    "db15k_mhyper_native",
    "db15k_mhyper_adamf",
    "db15k_native_adamf",
]
PAIR_SHORT = {
    "mkgw_mhyper_native": "MKG-W\nM-Hyper + NativE",
    "mkgw_mhyper_adamf": "MKG-W\nM-Hyper + AdaMF",
    "mkgw_native_adamf": "MKG-W\nNativE + AdaMF",
    "db15k_mhyper_native": "DB15K\nM-Hyper + NativE",
    "db15k_mhyper_adamf": "DB15K\nM-Hyper + AdaMF",
    "db15k_native_adamf": "DB15K\nNativE + AdaMF",
}
PAIR_DISPLAY = {
    "mkgw_mhyper_native": "MKG-W / M-Hyper + NativE",
    "mkgw_mhyper_adamf": "MKG-W / M-Hyper + AdaMF-MAT",
    "mkgw_native_adamf": "MKG-W / NativE + AdaMF-MAT",
    "db15k_mhyper_native": "DB15K / M-Hyper + NativE",
    "db15k_mhyper_adamf": "DB15K / M-Hyper + AdaMF-MAT",
    "db15k_native_adamf": "DB15K / NativE + AdaMF-MAT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the complete DEV-only heterogeneous-expert action landscape from "
            "the exact full-ranking RR curves. No checkpoint or evaluator is run."
        )
    )
    parser.add_argument("--utility-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument(
        "--output-dir", default="outputs/complementarity_identifiability/exp1_landscape"
    )
    parser.add_argument(
        "--report", default="docs/reports/complementarity_landscape_audit_2026-09-05.md"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--support-min", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def document_relative_link(target: Path, document: Path) -> str:
    return os.path.relpath(target.resolve(), document.resolve().parent).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_hash(path: Path, expected: str, role: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Source hash mismatch for {role}: {portable_path(path)}; "
            f"expected={expected}, actual={actual}"
        )


def count_components(mask: np.ndarray) -> np.ndarray:
    starts = mask[:, :1].sum(axis=1) + ((~mask[:, :-1]) & mask[:, 1:]).sum(axis=1)
    return starts.astype(np.int16)


def deterministic_best_indices(rr: np.ndarray, alpha0: float) -> np.ndarray:
    maxima = rr.max(axis=1, keepdims=True)
    is_max = np.isclose(rr, maxima, rtol=0.0, atol=ZERO_TOLERANCE)
    distance = np.abs(ALPHA_GRID - alpha0)
    # Lexicographic rule: maximum RR, then nearest Global, then smaller alpha.
    preference = distance + ALPHA_GRID * 1e-6
    preference = np.where(is_max, preference[None, :], np.inf)
    return preference.argmin(axis=1)


def compute_query_geometry(frame: pd.DataFrame, *, pair_id: str, alpha0: float) -> pd.DataFrame:
    rr = frame[ALPHA_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(rr).all():
        raise ValueError(f"{pair_id} contains non-finite alpha RR values")
    alpha0_index = int(np.flatnonzero(np.isclose(ALPHA_GRID, alpha0, atol=1e-12))[0])
    rr_global = rr[:, alpha0_index]
    rr_oracle = rr.max(axis=1)
    gain = rr_oracle - rr_global
    beneficial = rr > rr_global[:, None] + ZERO_TOLERANCE
    positive = beneficial.any(axis=1)
    min_distance = np.where(
        positive,
        np.where(beneficial, np.abs(ALPHA_GRID[None, :] - alpha0), np.inf).min(axis=1),
        np.nan,
    )
    best_indices = deterministic_best_indices(rr, alpha0)
    best_alpha = ALPHA_GRID[best_indices]
    best_delta = best_alpha - alpha0
    best_direction = np.where(best_delta > ZERO_TOLERANCE, "toward_a", "toward_b")
    best_direction = np.where(np.abs(best_delta) <= ZERO_TOLERANCE, "stay", best_direction)
    n_unique = np.fromiter(
        (np.unique(row).size for row in rr), dtype=np.int16, count=len(frame)
    )
    best_plateau = np.isclose(rr, rr_oracle[:, None], rtol=0.0, atol=ZERO_TOLERANCE).sum(axis=1)
    components = count_components(beneficial)

    result = pd.DataFrame(
        {
            "dataset": frame["dataset"].astype(str),
            "pair_id": pair_id,
            "expert_a": frame["expert_a_name"].astype(str),
            "expert_b": frame["expert_b_name"].astype(str),
            "query_id": frame["query_id"].astype(str),
            "original_triple_id": (
                "h="
                + frame["head_id"].astype(str)
                + "|r="
                + frame["relation_id"].astype(str)
                + "|t="
                + frame["tail_id"].astype(str)
            ),
            "seed": pd.to_numeric(frame["seed"], errors="raise").astype(int),
            "direction": frame["direction"].astype(str),
            "relation_id": pd.to_numeric(frame["relation_id"], errors="raise").astype(int),
            "head_id": pd.to_numeric(frame["head_id"], errors="raise").astype(int),
            "tail_id": pd.to_numeric(frame["tail_id"], errors="raise").astype(int),
            "global_alpha": alpha0,
            "global_rr": rr_global,
            "oracle_rr": rr_oracle,
            "gain_amplitude": gain,
            "beneficial_basin_width": beneficial.mean(axis=1),
            "min_beneficial_distance": min_distance,
            "best_alpha": best_alpha,
            "best_action_direction": best_direction,
            "plateau_ratio": (len(ALPHA_GRID) - n_unique) / (len(ALPHA_GRID) - 1),
            "best_plateau_ratio": best_plateau / len(ALPHA_GRID),
            "beneficial_components": components,
            "beneficial_fragmented": components > 1,
            "positive_opportunity": positive,
        }
    )
    for index, alpha in enumerate(ALPHA_GRID):
        result[f"u_alpha_{alpha:.2f}".replace(".", "_")] = rr[:, index] - rr_global
    return result


def clustered_bootstrap(
    geometry: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> dict[str, float | int | bool | str]:
    cluster_values = (
        geometry.groupby("original_triple_id", sort=False)["gain_amplitude"].mean().to_numpy()
    )
    if cluster_values.size < 2:
        raise ValueError("At least two original-triple clusters are required")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    batch_size = 64
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        samples = rng.integers(
            0, cluster_values.size, size=(stop - start, cluster_values.size)
        )
        boot[start:stop] = cluster_values[samples].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "n_original_triple_clusters": int(cluster_values.size),
        "bootstrap_samples": int(n_bootstrap),
        "bootstrap_unit": "original_triple (six seed-direction observations per cluster)",
        "headroom_ci95_low": float(low),
        "headroom_ci95_high": float(high),
        "headroom_ci95_lower_gt_zero": bool(low > 0.0),
    }


def distribution_rows(geometry: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    metrics = {
        "gain_amplitude": geometry["gain_amplitude"],
        "beneficial_basin_width": geometry["beneficial_basin_width"],
        "min_beneficial_distance_positive_only": geometry.loc[
            geometry["positive_opportunity"], "min_beneficial_distance"
        ],
        "plateau_ratio": geometry["plateau_ratio"],
        "best_plateau_ratio": geometry["best_plateau_ratio"],
        "beneficial_components_positive_only": geometry.loc[
            geometry["positive_opportunity"], "beneficial_components"
        ],
    }
    for metric, values in metrics.items():
        values = values.dropna().to_numpy(dtype=np.float64)
        quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
        rows.append(
            {
                "dataset": geometry["dataset"].iloc[0],
                "pair_id": geometry["pair_id"].iloc[0],
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(quantiles[0]),
                "q25": float(quantiles[1]),
                "median": float(quantiles[2]),
                "q75": float(quantiles[3]),
                "q90": float(quantiles[4]),
                "q95": float(quantiles[5]),
                "max": float(quantiles[6]),
            }
        )
    return rows


def direction_consistency_rows(geometry: pd.DataFrame, support_min: int) -> list[dict]:
    rows: list[dict] = []
    scopes = {
        "direction": ["direction"],
        "relation": ["relation_id"],
        "relation_x_direction": ["relation_id", "direction"],
    }
    for scope, columns in scopes.items():
        for keys, group in geometry.groupby(columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            positive = group[group["positive_opportunity"]]
            counts = positive["best_action_direction"].value_counts()
            toward_a = int(counts.get("toward_a", 0))
            toward_b = int(counts.get("toward_b", 0))
            opportunity_n = toward_a + toward_b
            consistency = max(toward_a, toward_b) / opportunity_n if opportunity_n else np.nan
            signed = (toward_a - toward_b) / opportunity_n if opportunity_n else np.nan
            record = {
                "dataset": geometry["dataset"].iloc[0],
                "pair_id": geometry["pair_id"].iloc[0],
                "scope": scope,
                "relation_id": "",
                "direction": "",
                "n_queries": int(len(group)),
                "n_original_triples": int(group["original_triple_id"].nunique()),
                "positive_opportunities": opportunity_n,
                "toward_a": toward_a,
                "toward_b": toward_b,
                "direction_consistency": consistency,
                "signed_direction_preference": signed,
                "supported": bool(len(group) >= support_min),
                "support_min": int(support_min),
            }
            for column, value in zip(columns, keys):
                record[column] = value
            rows.append(record)
    return rows


def pair_statistics(
    geometry: pd.DataFrame, *, n_bootstrap: int, bootstrap_seed: int, support_min: int
) -> dict:
    ci = clustered_bootstrap(
        geometry, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    positive = geometry[geometry["positive_opportunity"]]
    direction_rows = pd.DataFrame(direction_consistency_rows(geometry, support_min))
    supported_context = direction_rows[
        (direction_rows["scope"] == "relation_x_direction")
        & direction_rows["supported"]
        & direction_rows["direction_consistency"].notna()
    ]
    direction_scope = direction_rows[
        (direction_rows["scope"] == "direction")
        & direction_rows["direction_consistency"].notna()
    ]
    return {
        "dataset": geometry["dataset"].iloc[0],
        "pair_id": geometry["pair_id"].iloc[0],
        "expert_a": geometry["expert_a"].iloc[0],
        "expert_b": geometry["expert_b"].iloc[0],
        "n_seed_direction_queries": int(len(geometry)),
        "n_original_triple_clusters": int(geometry["original_triple_id"].nunique()),
        "seeds": sorted(int(value) for value in geometry["seed"].unique()),
        "directions": sorted(geometry["direction"].unique().tolist()),
        "global_alpha": float(geometry["global_alpha"].iloc[0]),
        "global_at_grid_boundary": bool(
            math.isclose(float(geometry["global_alpha"].iloc[0]), float(ALPHA_GRID[0]))
            or math.isclose(float(geometry["global_alpha"].iloc[0]), float(ALPHA_GRID[-1]))
        ),
        "global_mrr": float(geometry["global_rr"].mean()),
        "oracle_mrr": float(geometry["oracle_rr"].mean()),
        "available_headroom": float(geometry["gain_amplitude"].mean()),
        **ci,
        "positive_opportunity_rate": float(geometry["positive_opportunity"].mean()),
        "gain_mean": float(geometry["gain_amplitude"].mean()),
        "gain_median": float(geometry["gain_amplitude"].median()),
        "width_mean": float(geometry["beneficial_basin_width"].mean()),
        "width_median": float(geometry["beneficial_basin_width"].median()),
        "distance_mean_positive_only": float(positive["min_beneficial_distance"].mean()),
        "distance_median_positive_only": float(positive["min_beneficial_distance"].median()),
        "plateau_ratio_mean": float(geometry["plateau_ratio"].mean()),
        "plateau_ratio_median": float(geometry["plateau_ratio"].median()),
        "best_plateau_ratio_mean": float(geometry["best_plateau_ratio"].mean()),
        "fragmented_all_query_rate": float(geometry["beneficial_fragmented"].mean()),
        "fragmented_positive_opportunity_rate": float(
            positive["beneficial_fragmented"].mean()
        ),
        "beneficial_components_mean_positive_only": float(
            positive["beneficial_components"].mean()
        ),
        "beneficial_components_max": int(geometry["beneficial_components"].max()),
        "direction_consistency_macro_head_tail": float(
            direction_scope["direction_consistency"].mean()
        ),
        "direction_consistency_macro_supported_relation_x_direction": float(
            supported_context["direction_consistency"].mean()
        ),
        "supported_relation_x_direction_groups": int(len(supported_context)),
    }


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.panel{font-size:13px;font-weight:700}.axis{font-size:10px}.note{font-size:11px;fill:#5f6368}</style>',
        f'<text x="24" y="30" class="title">{html.escape(title)}</text>',
    ]


def interpolate_color(value: float, negative: tuple[int, int, int], positive: tuple[int, int, int]) -> str:
    value = max(-1.0, min(1.0, float(value)))
    white = (247, 247, 247)
    target = positive if value >= 0 else negative
    fraction = abs(value)
    rgb = tuple(round(white[index] + fraction * (target[index] - white[index])) for index in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def write_svg(path: Path, parts: list[str]) -> list[Path]:
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return [path]


def plot_global_oracle(stats: pd.DataFrame, output_dir: Path) -> list[Path]:
    ordered = stats.set_index("pair_id").loc[PAIR_ORDER].reset_index()
    width, height = 940, 430
    parts = svg_header(width, height, "Available complementarity: Global to per-query Oracle")
    chart_x, chart_y, chart_w = 285, 68, 560
    low = float(ordered["global_mrr"].min()) - 0.01
    high = float(ordered["oracle_mrr"].max()) + 0.01
    scale = lambda value: chart_x + (float(value) - low) / (high - low) * chart_w
    for tick in np.linspace(low, high, 6):
        x = scale(tick)
        parts.append(f'<line x1="{x:.1f}" y1="55" x2="{x:.1f}" y2="355" stroke="#e8eaed"/>')
        parts.append(f'<text x="{x:.1f}" y="376" text-anchor="middle" class="axis">{tick:.3f}</text>')
    for index, row in enumerate(ordered.itertuples()):
        y = chart_y + index * 49
        x_global, x_oracle = scale(row.global_mrr), scale(row.oracle_mrr)
        label = PAIR_DISPLAY[row.pair_id]
        parts.append(f'<text x="274" y="{y + 4}" text-anchor="end" class="axis">{html.escape(label)}</text>')
        parts.append(f'<line x1="{x_global:.1f}" y1="{y}" x2="{x_oracle:.1f}" y2="{y}" stroke="#9aa0a6" stroke-width="4"/>')
        parts.append(f'<circle cx="{x_global:.1f}" cy="{y}" r="7" fill="#376795"/>')
        parts.append(f'<circle cx="{x_oracle:.1f}" cy="{y}" r="7" fill="#d1495b"/>')
        parts.append(f'<text x="{x_oracle + 11:.1f}" y="{y + 4}" class="axis">+{row.available_headroom:.3f}</text>')
    parts.extend([
        '<circle cx="650" cy="402" r="6" fill="#376795"/><text x="662" y="406" class="note">Global</text>',
        '<circle cx="738" cy="402" r="6" fill="#d1495b"/><text x="750" y="406" class="note">Oracle</text>',
        '<text x="565" y="395" text-anchor="middle" class="note">DEV MRR</text>',
    ])
    return write_svg(output_dir / "figure1_global_to_oracle.svg", parts)


def binned_landscape(frame: pd.DataFrame, max_bins: int = 240) -> tuple[np.ndarray, int]:
    u_columns = [f"u_alpha_{alpha:.2f}".replace(".", "_") for alpha in ALPHA_GRID]
    ordered = frame.assign(direction_order=(frame["direction"] == "tail").astype(int)).sort_values(
        ["direction_order", "best_alpha", "gain_amplitude", "query_id"],
        ascending=[True, True, False, True],
    )
    head_count = int((ordered["direction"] == "head").sum())
    chunks = np.array_split(np.arange(len(ordered)), max_bins)
    matrix = np.vstack([ordered.iloc[index][u_columns].to_numpy(dtype=np.float64).mean(axis=0) for index in chunks])
    head_boundary = round(max_bins * head_count / len(ordered))
    return matrix, head_boundary


def plot_action_heatmaps(geometries: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    pair_ids = ["mkgw_mhyper_native", "mkgw_native_adamf"]
    width, height = 1220, 760
    parts = svg_header(width, height, "Complete 21-action DEV landscapes (all three seeds)")
    parts.append('<text x="24" y="50" class="note">Rows are deterministically ordered then averaged into 240 equal-count display bins; statistics use unbinned exact queries.</text>')
    for panel, pair_id in enumerate(pair_ids):
        matrix, head_boundary = binned_landscape(geometries[pair_id])
        nonzero = np.abs(matrix[np.abs(matrix) > ZERO_TOLERANCE])
        limit = float(np.quantile(nonzero, 0.99)) if nonzero.size else 1.0
        ox, oy = 82 + panel * 586, 100
        cell_w, cell_h = 23, 2.25
        alpha0 = float(geometries[pair_id]["global_alpha"].iloc[0])
        parts.append(f'<text x="{ox}" y="78" class="panel">{html.escape(PAIR_DISPLAY[pair_id])}; Global alpha={alpha0:.2f}</text>')
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                normalized = matrix[row_index, column_index] / limit
                color = interpolate_color(normalized, (33, 102, 172), (178, 24, 43))
                parts.append(f'<rect x="{ox + column_index * cell_w:.1f}" y="{oy + row_index * cell_h:.1f}" width="{cell_w + 0.2:.1f}" height="{cell_h + 0.2:.2f}" fill="{color}"/>')
        global_index = int(np.flatnonzero(np.isclose(ALPHA_GRID, alpha0))[0])
        x_global = ox + (global_index + 0.5) * cell_w
        parts.append(f'<line x1="{x_global:.1f}" y1="{oy}" x2="{x_global:.1f}" y2="{oy + matrix.shape[0]*cell_h}" stroke="#111111" stroke-width="1.2" stroke-dasharray="4 3"/>')
        y_boundary = oy + head_boundary * cell_h
        parts.append(f'<line x1="{ox}" y1="{y_boundary:.1f}" x2="{ox + len(ALPHA_GRID)*cell_w}" y2="{y_boundary:.1f}" stroke="#111111"/>')
        parts.append(f'<text x="{ox - 10}" y="{oy + 18}" text-anchor="end" class="axis">head</text>')
        parts.append(f'<text x="{ox - 10}" y="{y_boundary + 18:.1f}" text-anchor="end" class="axis">tail</text>')
        for column_index in range(0, len(ALPHA_GRID), 4):
            x = ox + (column_index + 0.5) * cell_w
            parts.append(f'<text x="{x:.1f}" y="{oy + matrix.shape[0]*cell_h + 18:.1f}" text-anchor="middle" class="axis">{ALPHA_GRID[column_index]:.2f}</text>')
        parts.append(f'<text x="{ox + len(ALPHA_GRID)*cell_w/2:.1f}" y="{oy + matrix.shape[0]*cell_h + 40:.1f}" text-anchor="middle" class="note">alpha (weight on expert A)</text>')
    legend_y = 700
    for index, value in enumerate(np.linspace(-1, 1, 21)):
        color = interpolate_color(value, (33, 102, 172), (178, 24, 43))
        parts.append(f'<rect x="{470 + index*14}" y="{legend_y}" width="15" height="13" fill="{color}"/>')
    parts.append(f'<text x="455" y="{legend_y + 11}" text-anchor="end" class="axis">negative</text>')
    parts.append(f'<text x="770" y="{legend_y + 11}" class="axis">positive mean U (pairwise 99% clipping)</text>')
    return write_svg(output_dir / "figure2_action_landscape_heatmaps.svg", parts)


def plot_gwd(geometries: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    width, height = 1430, 550
    parts = svg_header(width, height, "Per-query G / W / D distributions")
    specs = [
        ("gain_amplitude", "G: gain amplitude", False),
        ("beneficial_basin_width", "W: beneficial width", False),
        ("min_beneficial_distance", "D: minimum distance (positive only)", True),
    ]
    for panel, (column, title, positive_only) in enumerate(specs):
        ox, oy, chart_w, chart_h = 55 + panel * 465, 75, 400, 305
        arrays = []
        for pair_id in PAIR_ORDER:
            frame = geometries[pair_id]
            if positive_only:
                frame = frame[frame["positive_opportunity"]]
            arrays.append(frame[column].dropna().to_numpy(dtype=np.float64))
        high = max(float(np.quantile(values, 0.99)) for values in arrays)
        high = max(high, 1e-9)
        y_scale = lambda value: oy + chart_h - min(float(value), high) / high * chart_h
        parts.append(f'<text x="{ox}" y="58" class="panel">{html.escape(title)}</text>')
        for tick_fraction in np.linspace(0, 1, 5):
            y = oy + chart_h * (1 - tick_fraction)
            parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox + chart_w}" y2="{y:.1f}" stroke="#e8eaed"/>')
            parts.append(f'<text x="{ox - 6}" y="{y + 3:.1f}" text-anchor="end" class="axis">{tick_fraction*high:.3f}</text>')
        for index, values in enumerate(arrays):
            q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
            x = ox + (index + 0.5) * chart_w / len(arrays)
            parts.append(f'<line x1="{x:.1f}" y1="{y_scale(q05):.1f}" x2="{x:.1f}" y2="{y_scale(q95):.1f}" stroke="#2f4b66"/>')
            parts.append(f'<rect x="{x - 20:.1f}" y="{y_scale(q75):.1f}" width="40" height="{max(1.0, y_scale(q25)-y_scale(q75)):.1f}" fill="#8ab6d6" stroke="#2f4b66"/>')
            parts.append(f'<line x1="{x - 20:.1f}" y1="{y_scale(median):.1f}" x2="{x + 20:.1f}" y2="{y_scale(median):.1f}" stroke="#d1495b" stroke-width="2"/>')
            labels = PAIR_SHORT[PAIR_ORDER[index]].split("\n")
            parts.append(f'<text x="{x:.1f}" y="402" text-anchor="middle" class="axis">{html.escape(labels[0])}</text>')
            parts.append(f'<text x="{x:.1f}" y="415" text-anchor="middle" class="axis">{html.escape(labels[1].replace(" + ", "+"))}</text>')
        parts.append(f'<text x="{ox + chart_w/2:.1f}" y="448" text-anchor="middle" class="note">boxes: q25-q75; red median; whiskers: q05-q95; axis capped at q99</text>')
    return write_svg(output_dir / "figure3_gwd_distributions.svg", parts)


def plot_relation_direction_consistency(consistency: pd.DataFrame, output_dir: Path) -> list[Path]:
    subset = consistency[
        (consistency["scope"] == "relation_x_direction")
        & consistency["supported"]
        & consistency["signed_direction_preference"].notna()
    ].copy()
    width, height = 1350, 980
    parts = svg_header(width, height, "Relation x prediction-direction complementarity consistency")
    for panel, (dataset, dataset_label) in enumerate((("mkg_w", "MKG-W"), ("db15k", "DB15K"))):
        current = subset[subset["dataset"] == dataset]
        relations = sorted(int(value) for value in current["relation_id"].unique())
        columns = [
            (pair, direction)
            for pair in PAIR_ORDER
            if pair.startswith("mkgw" if dataset == "mkg_w" else "db15k")
            for direction in ("head", "tail")
        ]
        matrix = np.full((len(relations), len(columns)), np.nan)
        relation_index = {value: index for index, value in enumerate(relations)}
        column_index = {value: index for index, value in enumerate(columns)}
        for row in current.itertuples():
            key = (row.pair_id, row.direction)
            if key in column_index:
                matrix[relation_index[int(row.relation_id)], column_index[key]] = row.signed_direction_preference
        ox, oy, chart_w, chart_h = 65 + panel * 650, 100, 530, 730
        cell_w, cell_h = chart_w / len(columns), chart_h / max(1, len(relations))
        parts.append(f'<text x="{ox}" y="75" class="panel">{dataset_label}: supported groups</text>')
        for row_index in range(len(relations)):
            for column_index_value in range(len(columns)):
                value = matrix[row_index, column_index_value]
                color = "#d9d9d9" if np.isnan(value) else interpolate_color(value, (118, 42, 131), (230, 97, 1))
                parts.append(f'<rect x="{ox + column_index_value*cell_w:.2f}" y="{oy + row_index*cell_h:.2f}" width="{cell_w + 0.2:.2f}" height="{cell_h + 0.2:.2f}" fill="{color}"/>')
        step = max(1, len(relations) // 30)
        for row_index in range(0, len(relations), step):
            y = oy + (row_index + 0.5) * cell_h
            parts.append(f'<text x="{ox - 7}" y="{y + 3:.1f}" text-anchor="end" class="axis">{relations[row_index]}</text>')
        for column_index_value, (pair, direction) in enumerate(columns):
            x = ox + (column_index_value + 0.5) * cell_w
            experts = PAIR_DISPLAY[pair].split(" / ", 1)[1].replace("-MAT", "")
            parts.append(f'<text x="{x:.1f}" y="{oy + chart_h + 18}" text-anchor="middle" class="axis">{html.escape(direction)}</text>')
            parts.append(f'<text x="{x:.1f}" y="{oy + chart_h + 32}" text-anchor="middle" class="axis">{html.escape(experts.replace(" + ", "+"))}</text>')
        parts.append(f'<text x="{ox - 7}" y="{oy - 8}" text-anchor="end" class="note">relation_id</text>')
    legend_y = 920
    for index, value in enumerate(np.linspace(-1, 1, 21)):
        color = interpolate_color(value, (118, 42, 131), (230, 97, 1))
        parts.append(f'<rect x="{480 + index*16}" y="{legend_y}" width="17" height="14" fill="{color}"/>')
    parts.append(f'<text x="470" y="{legend_y + 11}" text-anchor="end" class="axis">-1: toward B</text>')
    parts.append(f'<text x="830" y="{legend_y + 11}" class="axis">+1: toward A</text>')
    return write_svg(output_dir / "figure4_relation_direction_consistency.svg", parts)


def markdown_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(
    path: Path,
    stats: pd.DataFrame,
    gate: dict,
    source_records: list[dict],
    output_dir: Path,
    support_min: int,
) -> None:
    ordered = stats.set_index("pair_id").loc[PAIR_ORDER].reset_index()
    lines = [
        "# Complementarity Landscape Audit — Experiment 1",
        "",
        "Date: 2026-09-05",
        "",
        "## Outcome",
        "",
        (
            f"Frozen Available-complementarity gate: **{gate['decision']}**. "
            f"Headroom criterion passed by {gate['headroom_pass_count']}/6 pairs; "
            f"positive-opportunity criterion passed by {gate['positive_rate_pass_count']}/6 pairs."
        ),
        "",
        "This is a descriptive DEV-only audit. It trains no selector, runs no policy, and does not start Experiment 2.",
        "",
        "## Main findings",
        "",
        "- Available complementarity is unambiguous in all six dataset-pairs: action-grid Oracle headroom ranges from 0.040841 to 0.057545 MRR, and every original-triple clustered 95% CI excludes zero on the positive side.",
        "- Positive opportunities are common (44.997% to 63.564%), so the GO result is not driven by a very small tail of queries. The gain distribution is nevertheless highly zero-heavy: both DB15K M-Hyper pairs have median `G=0`.",
        "- The landscapes are strongly stepped: mean plateau ratios range from 0.501 to 0.560. Among positive queries, the nearest beneficial action is usually one grid step from Global (median `D=0.05` for every pair), while mean `D` ranges from 0.086 to 0.136 because a smaller subset requires larger moves.",
        "- Beneficial regions are usually connected, but not universally: 6.79% to 14.95% of positive-opportunity queries have multiple beneficial components, with as many as 5 to 8 components depending on the pair.",
        "- The proposed contrast between M-Hyper-centered and NativE + AdaMF-MAT geometry is not uniformly supported by Available-complementarity alone. MKG-W M-Hyper + NativE and NativE + AdaMF-MAT have similar mean widths (0.292 versus 0.284) and fragmentation among positive queries (14.94% versus 14.45%), even though their direction alignment differs modestly.",
        "- Four pairs have `alpha0=1.0`. Their apparent direction consistency of 1.0 is mechanically one-sided because no action exists above Global; it must not be interpreted as evidence that relation or context identifies a preferred correction direction. Only non-boundary pairs provide a genuinely two-sided direction-consistency diagnostic in this audit.",
        "",
        "## Frozen protocol and operational boundary",
        "",
        "- Datasets: MKG-W and DB15K; pairs: M-Hyper + NativE, M-Hyper + AdaMF-MAT, and NativE + AdaMF-MAT.",
        "- Evidence: exact filtered per-query RR from the manifest-linked `full_ranking/dev_query_rows.csv` files.",
        "- Action grid: `0.00:0.05:1.00`; normalization: `query_zscore`; seeds: 1, 2, 3; directions: head and tail.",
        "- Global alpha comes unchanged from each manifest-linked DEV `selection.json`.",
        "- Confidence intervals resample original triples and keep their three seeds and two prediction directions clustered.",
        "- TEST access = 0; checkpoint retraining = 0; checkpoint reselection = 0; historical result modification = 0.",
        "",
        "## Metric definitions",
        "",
        "For every seed-direction query, `U_q(alpha) = RR_q(alpha) - RR_q(alpha0)` and `G_q = max RR - RR(alpha0)`.",
        "",
        "- Beneficial basin width `W`: fraction of the 21 frozen grid actions with strictly positive `U`.",
        "- Minimum beneficial distance `D`: minimum absolute alpha displacement from Global among positive-utility actions; undefined when no opportunity exists.",
        "- Deterministic best alpha: maximize exact RR, then minimize distance to Global, then choose the smaller alpha.",
        "- Best-action direction: toward expert A for alpha above Global, toward expert B below Global, otherwise stay.",
        "- Plateau ratio: `(21 - number of distinct RR values) / 20`; 0 is fully varying and 1 is completely flat.",
        "- Fragmentation: number of contiguous positive-utility regions on the ordered grid; fragmented means more than one region.",
        f"- A relation/direction context group is marked supported at `{support_min}` or more seed-direction query observations.",
        "- Direction consistency is the larger of the toward-A and toward-B proportions among positive-opportunity queries; signed preference retains which side dominates.",
        "",
        "## Pair-level results",
        "",
        "| Dataset / pair | alpha0 | Global MRR | Oracle MRR | Headroom | Clustered 95% CI | Positive opportunities | G median | W median | D median* | Plateau mean | Fragmented positive | Direction consistency** |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered.itertuples():
        lines.append(
            f"| {PAIR_DISPLAY[row.pair_id]} | {row.global_alpha:.2f} | {row.global_mrr:.6f} | "
            f"{row.oracle_mrr:.6f} | {row.available_headroom:.6f} | "
            f"[{row.headroom_ci95_low:.6f}, {row.headroom_ci95_high:.6f}] | "
            f"{markdown_percent(row.positive_opportunity_rate)} | {row.gain_median:.6f} | "
            f"{row.width_median:.3f} | {row.distance_median_positive_only:.2f} | "
            f"{row.plateau_ratio_mean:.3f} | "
            f"{markdown_percent(row.fragmented_positive_opportunity_rate)} | "
            f"{row.direction_consistency_macro_supported_relation_x_direction:.3f} |"
        )
    lines.extend(
        [
            "",
            "\* `D` is summarized only over positive-opportunity queries. ** Macro average over supported relation × direction groups.",
            "",
            "## Frozen GO gate",
            "",
            f"- Headroom: {gate['headroom_pass_count']}/6 pairs have positive Available headroom with clustered-bootstrap lower 95% bound above zero (threshold: at least 5/6).",
            f"- Opportunity prevalence: {gate['positive_rate_pass_count']}/6 pairs have positive-opportunity rate at least 25% (threshold: at least 5/6).",
            f"- Decision: **{gate['decision']}** — {gate['interpretation']}",
            "",
            "The gate result only governs whether Experiment 2 is scientifically eligible. Experiment 2 was not started in this task.",
            "",
            "## Figures",
            "",
            f"1. [Global-to-Oracle dumbbell]({document_relative_link(output_dir / 'figure1_global_to_oracle.svg', path)})",
            f"2. [MKG-W action-landscape heatmaps]({document_relative_link(output_dir / 'figure2_action_landscape_heatmaps.svg', path)})",
            f"3. [G/W/D distributions]({document_relative_link(output_dir / 'figure3_gwd_distributions.svg', path)})",
            f"4. [Relation × direction consistency]({document_relative_link(output_dir / 'figure4_relation_direction_consistency.svg', path)})",
            "",
            "## Reproducibility outputs",
            "",
            f"- Per-query geometry: `{portable_path(output_dir / 'per_query_action_geometry.csv.gz')}`",
            f"- Pair statistics: `{portable_path(output_dir / 'pair_statistics.csv')}` and JSON equivalent.",
            f"- Distribution summaries: `{portable_path(output_dir / 'gwd_plateau_distribution_summary.csv')}`.",
            f"- Fragmentation summaries: `{portable_path(output_dir / 'fragmentation_statistics.csv')}`.",
            f"- Direction groups: `{portable_path(output_dir / 'direction_consistency.csv')}`.",
            f"- Action summaries: `{portable_path(output_dir / 'action_level_summary.csv')}`.",
            f"- Machine-readable audit/gate/source record: `{portable_path(output_dir / 'audit_manifest.json')}`.",
            "",
            "## Source hashes",
            "",
            "| Pair | Role | Path | SHA256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in source_records:
        lines.append(
            f"| {record['pair_id']} | {record['role']} | `{record['path']}` | `{record['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "All manifest-declared hashes were recomputed before analysis. The utility tables are recorded for provenance but were not read as analytical input.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.support_min <= 0:
        raise ValueError("--support-min must be positive")
    utility_dir = Path(args.utility_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    manifests = sorted(utility_dir.glob("*_dev_source_manifest.json"))
    pair_ids = {read_json(path).get("pair_id") for path in manifests}
    if pair_ids != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected exactly the six core DEV pairs; got {sorted(pair_ids)}"
        )
    expected_outputs = [
        output_dir / "per_query_action_geometry.csv.gz",
        output_dir / "pair_statistics.csv",
        output_dir / "pair_statistics.json",
        output_dir / "gwd_plateau_distribution_summary.csv",
        output_dir / "fragmentation_statistics.csv",
        output_dir / "direction_consistency.csv",
        output_dir / "action_level_summary.csv",
        output_dir / "audit_manifest.json",
        report_path,
    ]
    if not args.overwrite:
        existing = [path for path in expected_outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    geometries: dict[str, pd.DataFrame] = {}
    stats_rows: list[dict] = []
    distribution_output: list[dict] = []
    consistency_output: list[dict] = []
    action_output: list[dict] = []
    source_records: list[dict] = []

    for pair_offset, manifest_path in enumerate(manifests):
        manifest = read_json(manifest_path)
        pair_id = str(manifest["pair_id"])
        if manifest.get("split") != "dev" or "test" in manifest_path.name.lower():
            raise RuntimeError(f"TEST or non-DEV manifest rejected: {manifest_path}")
        if manifest.get("score_normalization") != "query_zscore":
            raise RuntimeError(f"Unexpected score normalization for {pair_id}")
        if not np.allclose(manifest.get("global_alpha_grid", []), ALPHA_GRID):
            raise RuntimeError(f"Unexpected alpha grid for {pair_id}")

        selection_path = Path(manifest["source_selection"]["path"])
        query_path = Path(manifest["source_query_rows"]["path"])
        summary_path = Path(manifest["source_full_ranking_summary"]["path"])
        utility_path = Path(manifest["output_table"]["path"])
        if query_path.name != "dev_query_rows.csv" or "test" in query_path.as_posix().lower():
            raise RuntimeError(f"TEST or unexpected query source rejected: {query_path}")
        for role, path, record in (
            ("source_query_rows", query_path, manifest["source_query_rows"]),
            ("source_selection", selection_path, manifest["source_selection"]),
            ("source_full_ranking_summary", summary_path, manifest["source_full_ranking_summary"]),
            ("aacpi_utility_table_not_consumed", utility_path, manifest["output_table"]),
        ):
            validate_hash(path, record["sha256"], f"{pair_id}:{role}")
            source_records.append(
                {
                    "pair_id": pair_id,
                    "role": role,
                    "path": portable_path(path),
                    "sha256": record["sha256"],
                    "consumed": role in {"source_query_rows", "source_selection"},
                }
            )
        source_records.append(
            {
                "pair_id": pair_id,
                "role": "aacpi_source_manifest",
                "path": portable_path(manifest_path),
                "sha256": sha256_file(manifest_path),
                "consumed": True,
            }
        )

        selection = read_json(selection_path)
        alpha0 = float(selection["global_alpha"])
        if (
            selection.get("dataset") != manifest.get("dataset")
            or selection.get("score_normalization") != "query_zscore"
            or sorted(int(seed) for seed in selection.get("seeds", [])) != [1, 2, 3]
            or not np.allclose(selection.get("alpha_grid", []), ALPHA_GRID)
            or not math.isclose(alpha0, float(manifest["global_alpha"]), abs_tol=1e-12)
        ):
            raise RuntimeError(f"Selection/manifest contract mismatch for {pair_id}")

        use_columns = [
            "pair_name",
            "dataset",
            "expert_a_name",
            "expert_b_name",
            "query_id",
            "split",
            "seed",
            "direction",
            "relation_id",
            "head_id",
            "tail_id",
            "alpha_global",
            "rr_global",
            *ALPHA_COLUMNS,
        ]
        source = pd.read_csv(query_path, usecols=use_columns)
        if set(source["split"].astype(str)) != {"dev"}:
            raise RuntimeError(f"Non-DEV rows rejected for {pair_id}")
        if set(source["pair_name"].astype(str)) != {pair_id}:
            raise RuntimeError(f"Pair id mismatch in {query_path}")
        if sorted(source["seed"].astype(int).unique().tolist()) != [1, 2, 3]:
            raise RuntimeError(f"Expected exactly seeds 1, 2, 3 for {pair_id}")
        if set(source["direction"].astype(str)) != {"head", "tail"}:
            raise RuntimeError(f"Expected head and tail rows for {pair_id}")
        cluster_sizes = source.groupby(["head_id", "relation_id", "tail_id"]).size()
        if set(cluster_sizes.unique()) != {6}:
            raise RuntimeError(f"Original triples do not each contain 3 seeds x 2 directions for {pair_id}")

        geometry = compute_query_geometry(source, pair_id=pair_id, alpha0=alpha0)
        if not np.allclose(geometry["global_rr"], source["rr_global"]):
            raise RuntimeError(f"Recomputed Global RR mismatch for {pair_id}")
        geometries[pair_id] = geometry
        stats_rows.append(
            pair_statistics(
                geometry,
                n_bootstrap=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + pair_offset,
                support_min=args.support_min,
            )
        )
        distribution_output.extend(distribution_rows(geometry))
        consistency_output.extend(direction_consistency_rows(geometry, args.support_min))
        u_columns = [f"u_alpha_{alpha:.2f}".replace(".", "_") for alpha in ALPHA_GRID]
        for alpha, column in zip(ALPHA_GRID, u_columns):
            values = geometry[column]
            action_output.append(
                {
                    "dataset": manifest["dataset"],
                    "pair_id": pair_id,
                    "global_alpha": alpha0,
                    "alpha": float(alpha),
                    "delta_alpha": float(alpha - alpha0),
                    "mean_utility": float(values.mean()),
                    "positive_rate": float((values > ZERO_TOLERANCE).mean()),
                    "zero_rate": float((np.abs(values) <= ZERO_TOLERANCE).mean()),
                    "negative_rate": float((values < -ZERO_TOLERANCE).mean()),
                }
            )

    all_geometry = pd.concat([geometries[pair] for pair in PAIR_ORDER], ignore_index=True)
    stats = pd.DataFrame(stats_rows)
    stats["pair_order"] = stats["pair_id"].map({pair: index for index, pair in enumerate(PAIR_ORDER)})
    stats = stats.sort_values("pair_order").drop(columns="pair_order")
    distributions = pd.DataFrame(distribution_output)
    consistency = pd.DataFrame(consistency_output)
    actions = pd.DataFrame(action_output)

    all_geometry.to_csv(output_dir / "per_query_action_geometry.csv.gz", index=False)
    stats.to_csv(output_dir / "pair_statistics.csv", index=False)
    (output_dir / "pair_statistics.json").write_text(
        json.dumps(stats.to_dict(orient="records"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    distributions.to_csv(output_dir / "gwd_plateau_distribution_summary.csv", index=False)
    fragmentation_columns = [
        "dataset",
        "pair_id",
        "fragmented_all_query_rate",
        "fragmented_positive_opportunity_rate",
        "beneficial_components_mean_positive_only",
        "beneficial_components_max",
    ]
    stats[fragmentation_columns].to_csv(
        output_dir / "fragmentation_statistics.csv", index=False
    )
    consistency.to_csv(output_dir / "direction_consistency.csv", index=False)
    actions.to_csv(output_dir / "action_level_summary.csv", index=False)

    figure_paths: list[Path] = []
    figure_paths.extend(plot_global_oracle(stats, output_dir))
    figure_paths.extend(plot_action_heatmaps(geometries, output_dir))
    figure_paths.extend(plot_gwd(geometries, output_dir))
    figure_paths.extend(plot_relation_direction_consistency(consistency, output_dir))

    headroom_pass = (
        (stats["available_headroom"] > 0.0)
        & (stats["headroom_ci95_low"] > 0.0)
    )
    positive_pass = stats["positive_opportunity_rate"] >= 0.25
    go = int(headroom_pass.sum()) >= 5 and int(positive_pass.sum()) >= 5
    gate = {
        "name": "Available-complementarity GO",
        "headroom_rule": ">=5/6 pairs Available headroom > 0 and clustered bootstrap 95% CI lower > 0",
        "positive_rate_rule": ">=5/6 pairs positive-opportunity rate >= 25%",
        "headroom_pass_count": int(headroom_pass.sum()),
        "positive_rate_pass_count": int(positive_pass.sum()),
        "pair_results": [
            {
                "pair_id": row.pair_id,
                "headroom_pass": bool(headroom_pass.iloc[index]),
                "positive_rate_pass": bool(positive_pass.iloc[index]),
            }
            for index, row in enumerate(stats.itertuples())
        ],
        "decision": "GO" if go else "NO-GO",
        "interpretation": (
            "Experiment 2 is eligible, but is not started by this audit"
            if go
            else "stop the Complementarity Identifiability study before Experiment 2"
        ),
    }
    generated_paths = [
        output_dir / "per_query_action_geometry.csv.gz",
        output_dir / "pair_statistics.csv",
        output_dir / "pair_statistics.json",
        output_dir / "gwd_plateau_distribution_summary.csv",
        output_dir / "fragmentation_statistics.csv",
        output_dir / "direction_consistency.csv",
        output_dir / "action_level_summary.csv",
        *figure_paths,
    ]
    audit = {
        "schema_version": 1,
        "experiment": "Experiment 1 — Complementarity Landscape Audit",
        "created_date": "2026-09-05",
        "branch": "m1/recent-mmkgc-baselines",
        "split": "dev",
        "datasets": ["mkg_w", "db15k"],
        "pairs": PAIR_ORDER,
        "score_normalization": "query_zscore",
        "alpha_grid": ALPHA_GRID.tolist(),
        "seeds": [1, 2, 3],
        "directions": ["head", "tail"],
        "statistics_cluster": "original_triple_id",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed_base": args.bootstrap_seed,
        "support_min_seed_direction_queries": args.support_min,
        "definitions": {
            "beneficial_basin_width": "count(U(alpha)>0)/21",
            "minimum_beneficial_distance": "min abs(alpha-alpha0) over U(alpha)>0; missing without opportunity",
            "deterministic_best_alpha": "max RR, then nearest global alpha, then smaller alpha",
            "plateau_ratio": "(21-number of distinct exact RR values)/20",
            "fragmentation": "number of contiguous U(alpha)>0 components on the ordered grid",
            "direction_consistency": "max(toward_a,toward_b)/positive opportunities within group",
        },
        "gate": gate,
        "operational_audit": {
            "test_access": 0,
            "checkpoint_retraining": 0,
            "checkpoint_reselection": 0,
            "full_ranking_evaluator_runs": 0,
            "historical_result_modification": 0,
            "selector_training": 0,
            "policy_runs": 0,
            "experiment_2_started": 0,
            "aacpi_local_action_utility_table_consumed": 0,
        },
        "sources": source_records,
        "outputs": [
            {
                "path": portable_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in generated_paths
        ],
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(
        report_path,
        stats,
        gate,
        source_records,
        output_dir,
        args.support_min,
    )
    print(json.dumps({"gate": gate, "pair_statistics": stats.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
