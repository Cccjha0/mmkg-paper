from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.training.src.data.dataset_spec import MMKG_GENERAL_V1
from ml.training.src.data.feature_bundle import load_processed_feature_bundle
from router.schemas import CleanRouterFeatureRecord, GeneralCleanRouterFeatureRecord, PosthocRouterFeatureRecord


def load_cache_bundle(cache_dir: str | Path) -> dict:
    """Load the frozen OpenBG cache contract (legacy default)."""
    cache_dir = Path(cache_dir)
    text_feat = torch.load(cache_dir / "text_emb.pt", map_location="cpu").float()
    img_feat = torch.load(cache_dir / "img_emb.pt", map_location="cpu").float()
    has_img = torch.load(cache_dir / "has_img.pt", map_location="cpu").bool()
    return {
        "cache_dir": cache_dir.as_posix(),
        "text_feat": text_feat,
        "img_feat": img_feat,
        "has_img": has_img,
    }


def load_general_cache_bundle(processed_dir: str | Path) -> dict:
    """Load canonical features and independent availability masks."""
    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing canonical dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != MMKG_GENERAL_V1:
        raise ValueError(
            f"Expected manifest protocol {MMKG_GENERAL_V1!r}, got {manifest.get('protocol')!r}."
        )
    dataset = str(manifest.get("dataset", "")).strip()
    if not dataset:
        raise ValueError("Canonical dataset manifest is missing a non-empty dataset name.")
    features = load_processed_feature_bundle(processed_dir)
    similarity = manifest.get("cross_modal_similarity", {})
    cosine_enabled = bool(similarity.get("shared_embedding_space", False)) and similarity.get("type") == "cosine"
    if cosine_enabled and features.text_features.size(1) != features.image_features.size(1):
        raise ValueError("Manifest declares a shared cosine space, but text/image feature dimensions differ.")
    return {
        "cache_dir": processed_dir.as_posix(),
        "dataset": dataset,
        "protocol_version": MMKG_GENERAL_V1,
        "text_feat": features.text_features,
        "img_feat": features.image_features,
        "has_text": features.has_text,
        "has_img": features.has_img,
        "cross_modal_cosine_enabled": cosine_enabled,
    }


def load_run_config(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config_merged.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_merged.json under run dir: {run_dir}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def infer_cache_dir(cache_dir: str | None, run_dir: str | None) -> str:
    if cache_dir:
        return cache_dir
    if run_dir:
        cfg = load_run_config(run_dir)
        return str(cfg["dataset"]["cache_dir"])
    raise ValueError("Either --cache-dir or --run-dir is required to build router features.")


def load_relation_prior_map(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in rows:
        relation_id = int(row["relation_id"])
        prior_flag = row.get("is_fusion_prior", row.get("is_visual_prior", 0))
        out[relation_id] = {
            "relation_gain_prior": float(row["mean_delta_rr"]),
            "relation_fusion_win_rate": float(row["fusion_win_rate"]),
            "relation_support": int(row["n_queries"]),
            "relation_is_visual_prior": int(row.get("is_visual_prior", prior_flag)),
            "relation_is_fusion_prior": int(prior_flag),
        }
    return out


def cosine_for_entity(cache_bundle: dict, entity_id: int) -> float:
    text_vec = cache_bundle["text_feat"][entity_id]
    img_vec = cache_bundle["img_feat"][entity_id]
    if not bool(cache_bundle["has_img"][entity_id].item()):
        return 0.0
    text_norm = float(text_vec.norm().item())
    img_norm = float(img_vec.norm().item())
    if text_norm <= 0.0 or img_norm <= 0.0:
        return 0.0
    return float(F.cosine_similarity(text_vec.unsqueeze(0), img_vec.unsqueeze(0), dim=1).item())


def missing_replaced_flag(cache_bundle: dict, entity_id: int) -> int:
    return int(not bool(cache_bundle["has_img"][entity_id].item()))


def get_observed_entity_id(row: dict) -> int:
    direction = str(row["direction"])
    if direction == "tail":
        return int(row["head_id"])
    if direction == "head":
        return int(row["tail_id"])
    raise ValueError(f"Unknown direction: {direction}")


def cosine_for_observed_entity(cache_bundle: dict, entity_id: int) -> float:
    return cosine_for_entity(cache_bundle, entity_id)


def missing_replaced_flag_for_observed_entity(cache_bundle: dict, entity_id: int) -> int:
    return missing_replaced_flag(cache_bundle, entity_id)


def observed_has_img_flag(cache_bundle: dict, entity_id: int) -> int:
    return int(bool(cache_bundle["has_img"][entity_id].item()))


def _merge_query_eval_pair(gate_rows: list[dict], residual_rows: list[dict]) -> list[tuple[dict, dict]]:
    gate_by_id = {row["query_id"]: row for row in gate_rows}
    residual_by_id = {row["query_id"]: row for row in residual_rows}
    if len(gate_by_id) != len(gate_rows) or len(residual_by_id) != len(residual_rows):
        raise RuntimeError("Duplicate query_id detected in gate or residual query rows.")
    gate_ids = set(gate_by_id)
    residual_ids = set(residual_by_id)
    if gate_ids != residual_ids:
        missing_in_gate = len(residual_ids - gate_ids)
        missing_in_residual = len(gate_ids - residual_ids)
        raise RuntimeError(
            f"query_id mismatch between experts: missing_in_gate={missing_in_gate}, missing_in_residual={missing_in_residual}"
        )
    return [(gate_by_id[qid], residual_by_id[qid]) for qid in sorted(gate_ids)]


def _require_single_metadata(rows: list[dict], field: str, expert: str) -> str:
    values = {str(row.get(field, "")).strip() for row in rows}
    if len(values) != 1 or "" in values:
        raise RuntimeError(f"{expert} query rows must contain exactly one non-empty {field}: {sorted(values)}")
    return next(iter(values))


def _validate_general_sources(
    gate_rows: list[dict],
    residual_rows: list[dict],
    cache_bundle: dict,
) -> tuple[str, str]:
    gate_dataset = _require_single_metadata(gate_rows, "dataset", "gate")
    residual_dataset = _require_single_metadata(residual_rows, "dataset", "residual")
    gate_protocol = _require_single_metadata(gate_rows, "protocol_version", "gate")
    residual_protocol = _require_single_metadata(residual_rows, "protocol_version", "residual")
    cache_dataset = str(cache_bundle.get("dataset", "")).strip()
    cache_protocol = str(cache_bundle.get("protocol_version", "")).strip()
    if gate_dataset != residual_dataset or gate_dataset != cache_dataset:
        raise RuntimeError(
            "General router sources must use one dataset: "
            f"gate={gate_dataset!r}, residual={residual_dataset!r}, cache={cache_dataset!r}."
        )
    if gate_protocol != residual_protocol or gate_protocol != cache_protocol or gate_protocol != MMKG_GENERAL_V1:
        raise RuntimeError(
            "General router sources must use mmkg_general_v1: "
            f"gate={gate_protocol!r}, residual={residual_protocol!r}, cache={cache_protocol!r}."
        )
    return gate_dataset, gate_protocol


def _default_prior(relation_id: int, relation_prior_map: dict[int, dict]) -> dict:
    return relation_prior_map.get(
        relation_id,
        {
            "relation_gain_prior": 0.0,
            "relation_fusion_win_rate": 0.0,
            "relation_support": 0,
            "relation_is_visual_prior": 0,
            "relation_is_fusion_prior": 0,
        },
    )


def build_clean_feature_rows(
    gate_rows: list[dict],
    residual_rows: list[dict],
    relation_prior_map: dict[int, dict],
    cache_bundle: dict,
    label_by_query_id: dict[str, dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for gate, _residual in _merge_query_eval_pair(gate_rows, residual_rows):
        relation_id = int(gate["relation_id"])
        observed_entity_id = get_observed_entity_id(gate)
        prior = _default_prior(relation_id, relation_prior_map)
        label_row = label_by_query_id.get(gate["query_id"]) if label_by_query_id is not None else None

        record = CleanRouterFeatureRecord(
            query_id=str(gate["query_id"]),
            split=str(gate["split"]),
            seed=int(gate["seed"]),
            direction=str(gate["direction"]),
            relation_id=relation_id,
            relation_name=str(gate["relation_name"]),
            head_id=int(gate["head_id"]),
            tail_id=int(gate["tail_id"]),
            observed_entity_id=observed_entity_id,
            observed_has_img=observed_has_img_flag(cache_bundle, observed_entity_id),
            observed_text_img_cosine=cosine_for_observed_entity(cache_bundle, observed_entity_id),
            observed_img_missing_replaced=missing_replaced_flag_for_observed_entity(cache_bundle, observed_entity_id),
            relation_gain_prior=float(prior["relation_gain_prior"]),
            relation_fusion_win_rate=float(prior["relation_fusion_win_rate"]),
            relation_support=int(prior["relation_support"]),
            relation_is_visual_prior=int(prior["relation_is_visual_prior"]),
            label_gain=int(label_row["label_gain"]) if label_row is not None else None,
            delta_threshold=float(label_row["delta_threshold"]) if label_row is not None else None,
        )
        rows.append(record.to_dict())
    return rows


def build_general_clean_feature_rows(
    gate_rows: list[dict],
    residual_rows: list[dict],
    relation_prior_map: dict[int, dict],
    cache_bundle: dict,
    label_by_query_id: dict[str, dict] | None = None,
) -> list[dict]:
    """Build legal router inputs without leaking target modality masks."""
    dataset, protocol_version = _validate_general_sources(gate_rows, residual_rows, cache_bundle)
    rows: list[dict] = []
    for gate, residual in _merge_query_eval_pair(gate_rows, residual_rows):
        for field in ("split", "seed", "direction", "relation_id", "head_id", "tail_id"):
            if str(gate[field]) != str(residual[field]):
                raise RuntimeError(
                    f"General router expert mismatch for query_id={gate['query_id']}: "
                    f"{field} gate={gate[field]!r}, residual={residual[field]!r}."
                )
        relation_id = int(gate["relation_id"])
        entity_id = get_observed_entity_id(gate)
        prior = _default_prior(relation_id, relation_prior_map)
        label_row = label_by_query_id.get(gate["query_id"]) if label_by_query_id is not None else None
        if label_row is not None:
            label_dataset = str(label_row.get("dataset", "")).strip()
            label_protocol = str(label_row.get("protocol_version", "")).strip()
            if label_dataset != dataset or label_protocol != protocol_version:
                raise RuntimeError(
                    f"Gain label metadata mismatch for query_id={gate['query_id']}: "
                    f"dataset={label_dataset!r}, protocol={label_protocol!r}."
                )
        has_text = bool(cache_bundle["has_text"][entity_id].item())
        has_img = bool(cache_bundle["has_img"][entity_id].item())
        cosine_valid = has_text and has_img and bool(cache_bundle.get("cross_modal_cosine_enabled", False))
        cosine = cosine_for_entity(cache_bundle, entity_id) if cosine_valid else 0.0
        record = GeneralCleanRouterFeatureRecord(
            query_id=str(gate["query_id"]),
            split=str(gate["split"]),
            seed=int(gate["seed"]),
            direction=str(gate["direction"]),
            relation_id=relation_id,
            relation_name=str(gate["relation_name"]),
            head_id=int(gate["head_id"]),
            tail_id=int(gate["tail_id"]),
            observed_entity_id=entity_id,
            observed_has_text=int(has_text),
            observed_has_img=int(has_img),
            observed_modality_count=int(has_text) + int(has_img),
            observed_text_img_cosine=cosine,
            observed_text_img_cosine_valid=int(cosine_valid),
            relation_gain_prior=float(prior["relation_gain_prior"]),
            relation_fusion_win_rate=float(prior["relation_fusion_win_rate"]),
            relation_support=int(prior["relation_support"]),
            relation_is_fusion_prior=int(prior["relation_is_fusion_prior"]),
            label_gain=int(label_row["label_gain"]) if label_row is not None else None,
            delta_threshold=float(label_row["delta_threshold"]) if label_row is not None else None,
        )
        rows.append(record.to_dict())
    return rows


def build_posthoc_feature_rows(
    gate_rows: list[dict],
    residual_rows: list[dict],
    relation_prior_map: dict[int, dict],
    cache_bundle: dict,
    label_by_query_id: dict[str, dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for gate, residual in _merge_query_eval_pair(gate_rows, residual_rows):
        relation_id = int(gate["relation_id"])
        target_entity_id = int(gate["target_entity_id"])
        prior = _default_prior(relation_id, relation_prior_map)
        label_row = label_by_query_id.get(gate["query_id"]) if label_by_query_id is not None else None

        record = PosthocRouterFeatureRecord(
            query_id=str(gate["query_id"]),
            split=str(gate["split"]),
            seed=int(gate["seed"]),
            direction=str(gate["direction"]),
            relation_id=relation_id,
            relation_name=str(gate["relation_name"]),
            head_id=int(gate["head_id"]),
            tail_id=int(gate["tail_id"]),
            target_entity_id=target_entity_id,
            target_position=str(gate["target_position"]),
            target_has_img=int(gate["target_has_img"]),
            target_regime=str(gate["target_regime"]),
            relation_gain_prior=float(prior["relation_gain_prior"]),
            relation_fusion_win_rate=float(prior["relation_fusion_win_rate"]),
            relation_support=int(prior["relation_support"]),
            relation_is_visual_prior=int(prior["relation_is_visual_prior"]),
            text_img_cosine=cosine_for_entity(cache_bundle, target_entity_id),
            img_is_missing_replaced=missing_replaced_flag(cache_bundle, target_entity_id),
            fusion_margin=float(gate["score_margin"]),
            struct_margin=float(residual["score_margin"]),
            fusion_correct_score=float(gate["correct_score"]),
            struct_correct_score=float(residual["correct_score"]),
            delta_margin=float(gate["score_margin"]) - float(residual["score_margin"]),
            rr_fusion=float(gate["rr"]),
            rr_struct=float(residual["rr"]),
            rank_fusion=int(gate["rank"]),
            rank_struct=int(residual["rank"]),
            label_gain=int(label_row["label_gain"]) if label_row is not None else None,
            delta_threshold=float(label_row["delta_threshold"]) if label_row is not None else None,
        )
        rows.append(record.to_dict())
    return rows


def _summarize_feature_rows_impl(
    train_rows_by_delta: dict[str, list[dict]],
    test_rows: list[dict],
    numeric_cols: list[str],
) -> dict:
    def missing_rate(rows: list[dict], col: str) -> float:
        if not rows:
            return 0.0
        miss = sum(1 for row in rows if row.get(col) in ("", None))
        return float(miss / len(rows))

    def numeric_range(rows: list[dict], col: str) -> dict:
        values = [float(row[col]) for row in rows if row.get(col) not in ("", None)]
        if not values:
            return {"min": None, "max": None}
        return {"min": min(values), "max": max(values)}

    def by_seed(rows: list[dict]) -> dict:
        counter: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            seed = str(row["seed"])
            counter[seed]["rows"] += 1
            if row.get("label_gain") not in ("", None):
                counter[seed]["positive"] += int(row["label_gain"])
        payload = {}
        for seed, stats in sorted(counter.items(), key=lambda item: int(item[0])):
            rows_n = int(stats["rows"])
            entry = {"rows": rows_n}
            if "positive" in stats:
                entry["positive_rate"] = float(stats["positive"] / rows_n) if rows_n else 0.0
            payload[seed] = entry
        return payload

    train_summary = {}
    for delta_tag, rows in sorted(train_rows_by_delta.items()):
        train_summary[delta_tag] = {
            "rows": len(rows),
            "by_seed": by_seed(rows),
            "missing_rate": {col: missing_rate(rows, col) for col in numeric_cols},
            "numeric_range": {col: numeric_range(rows, col) for col in numeric_cols},
        }

    return {
        "train": train_summary,
        "test": {
            "rows": len(test_rows),
            "by_seed": by_seed(test_rows),
            "missing_rate": {col: missing_rate(test_rows, col) for col in numeric_cols},
            "numeric_range": {col: numeric_range(test_rows, col) for col in numeric_cols},
        },
        "feature_columns": numeric_cols,
    }


def summarize_clean_feature_rows(train_rows_by_delta: dict[str, list[dict]], test_rows: list[dict]) -> dict:
    numeric_cols = [
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "observed_text_img_cosine",
    ]
    return _summarize_feature_rows_impl(train_rows_by_delta, test_rows, numeric_cols)


def summarize_general_clean_feature_rows(train_rows_by_delta: dict[str, list[dict]], test_rows: list[dict]) -> dict:
    numeric_cols = [
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "relation_is_fusion_prior",
        "observed_has_text",
        "observed_has_img",
        "observed_modality_count",
        "observed_text_img_cosine",
        "observed_text_img_cosine_valid",
    ]
    return _summarize_feature_rows_impl(train_rows_by_delta, test_rows, numeric_cols)


def summarize_posthoc_feature_rows(train_rows_by_delta: dict[str, list[dict]], test_rows: list[dict]) -> dict:
    numeric_cols = [
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "text_img_cosine",
        "fusion_margin",
        "struct_margin",
        "fusion_correct_score",
        "struct_correct_score",
        "delta_margin",
        "rr_fusion",
        "rr_struct",
        "rank_fusion",
        "rank_struct",
    ]
    return _summarize_feature_rows_impl(train_rows_by_delta, test_rows, numeric_cols)


# Backward-compatible aliases while migration is in progress.
build_feature_rows = build_posthoc_feature_rows
summarize_feature_rows = summarize_posthoc_feature_rows
