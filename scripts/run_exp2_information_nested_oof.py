from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp2_information_common import (
    ACTION_FIELDS,
    ALPHAS,
    RR_COLUMNS,
    ZERO_TOLERANCE,
    action_descriptors,
    grouped_folds,
    load_contract,
    policy_metrics,
    portable_path,
    project_latents,
    reject_test_path,
    representation_features,
    select_global_alpha,
    select_probe_actions,
    sha256_array,
    sha256_file,
    sign_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen Experiment 2 representation/learner strict nested OOF probe.")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--representation", choices=("X1", "X2", "X3", "X4", "X5", "X6"), required=True)
    parser.add_argument("--learner", choices=("linear_huber", "hist_gbdt", "mlp_low", "mlp_high", "set_encoder"), required=True)
    parser.add_argument("--asset-dir", default="outputs/complementarity_identifiability/exp2_information/assets")
    parser.add_argument("--candidate-dir", default="outputs/complementarity_identifiability/exp2_information/candidate_assets")
    parser.add_argument("--utility-manifest-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument("--phase4a-root", default="outputs/aacpi/phase4a")
    parser.add_argument("--exp1-stats", default="outputs/complementarity_identifiability/exp1_landscape/pair_statistics.csv")
    parser.add_argument("--exp1-manifest", default="outputs/complementarity_identifiability/exp1_landscape/audit_manifest.json")
    parser.add_argument("--contract", default="docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json")
    parser.add_argument("--output-root", default="outputs/complementarity_identifiability/exp2_information/runs")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def align_indices(source_ids: np.ndarray, target_ids: np.ndarray, label: str) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    if len(lookup) != len(source_ids) or set(lookup) != set(target_ids.astype(str)):
        raise RuntimeError(f"{label} query inventory mismatch")
    return np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)


def expand_tabular(query_features: np.ndarray, query_indices: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
    base = query_features[query_indices]
    return np.column_stack(
        (
            np.repeat(base, len(ALPHAS), axis=0),
            np.tile(descriptors, (len(query_indices), 1)),
        )
    ).astype(np.float32)


def standardize(train: np.ndarray, evaluate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    scale = train.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return ((train - mean) / scale).astype(np.float32), ((evaluate - mean) / scale).astype(np.float32)


def train_predict_tabular(
    learner: str,
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    contract: dict,
    device: str,
    seed: int,
) -> np.ndarray:
    if learner == "linear_huber":
        from sklearn.linear_model import SGDRegressor

        train, evaluate = standardize(x_train, x_eval)
        model = SGDRegressor(
            loss="huber", penalty="l2", alpha=float(config["alpha"]), epsilon=float(config["epsilon"]),
            max_iter=int(config["max_iter"]), tol=None, random_state=seed,
            learning_rate="invscaling", eta0=0.01, average=True,
        )
        model.fit(train, y_train)
        return model.predict(evaluate).astype(np.float64)
    if learner == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=float(config["learning_rate"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]), max_iter=int(config["max_iter"]),
            l2_regularization=float(config["l2_regularization"]), random_state=seed,
        )
        model.fit(x_train, y_train)
        return model.predict(x_eval).astype(np.float64)
    if learner not in {"mlp_low", "mlp_high"}:
        raise ValueError(f"Not a tabular learner: {learner}")
    import torch
    import torch.nn.functional as functional

    requested = device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train, evaluate = standardize(x_train, x_eval)
    width, layers = int(config["hidden_width"]), int(config["hidden_layers"])
    modules: list[torch.nn.Module] = []
    input_width = train.shape[1]
    for _ in range(layers):
        modules.extend((torch.nn.Linear(input_width, width), torch.nn.ReLU()))
        input_width = width
    modules.append(torch.nn.Linear(input_width, 1))
    model = torch.nn.Sequential(*modules).to(requested)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0)
    batch_size = int(contract["training"]["batch_size"])
    generator = torch.Generator(device=requested); generator.manual_seed(seed)
    train_x = torch.from_numpy(train).to(requested)
    train_y = torch.from_numpy(y_train.astype(np.float32)).to(requested)
    model.train()
    for _ in range(int(config["epochs"])):
        permutation = torch.randperm(len(train_x), generator=generator, device=requested)
        for start in range(0, len(train_x), batch_size):
            indices = permutation[start : start + batch_size]
            prediction = model(train_x[indices]).squeeze(1)
            loss = functional.smooth_l1_loss(prediction, train_y[indices], beta=0.02)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(contract["training"]["max_gradient_norm"])); optimizer.step()
    model.eval(); predictions = []
    evaluate_x = torch.from_numpy(evaluate)
    with torch.inference_mode():
        for start in range(0, len(evaluate_x), batch_size):
            predictions.append(model(evaluate_x[start : start + batch_size].to(requested)).squeeze(1).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def train_predict_set(
    config: dict,
    query_context: np.ndarray,
    candidate_features: np.ndarray,
    candidate_mask: np.ndarray,
    query_train: np.ndarray,
    y_train: np.ndarray,
    query_eval: np.ndarray,
    descriptors: np.ndarray,
    contract: dict,
    device: str,
    seed: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    requested = device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    context_mean = query_context[query_train].mean(axis=0, dtype=np.float64)
    context_scale = query_context[query_train].std(axis=0, dtype=np.float64)
    context_scale[context_scale < 1e-8] = 1.0
    normalized_context = ((query_context - context_mean) / context_scale).astype(np.float32)

    class SetEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            candidate_width = int(config["candidate_width"])
            pooled_width = int(config["pooled_width"])
            self.phi = torch.nn.Sequential(
                torch.nn.Linear(candidate_features.shape[2], candidate_width), torch.nn.ReLU(),
                torch.nn.Linear(candidate_width, candidate_width), torch.nn.ReLU(),
            )
            self.rho = torch.nn.Sequential(
                torch.nn.Linear(2 * candidate_width + query_context.shape[1] + len(ACTION_FIELDS), pooled_width), torch.nn.ReLU(),
                torch.nn.Linear(pooled_width, 1),
            )

        def forward(self, candidates, mask, context, actions):
            encoded = self.phi(candidates)
            weights = mask.unsqueeze(-1).to(encoded.dtype)
            mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            maximum = encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
            return self.rho(torch.cat((mean, maximum, context, actions), dim=1)).squeeze(1)

    model = SetEncoder().to(requested)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0)
    batch_size = int(contract["training"]["set_batch_size"])
    total_train = len(query_train) * len(ALPHAS)
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    model.train()
    for _ in range(int(config["epochs"])):
        permutation = torch.randperm(total_train, generator=generator)
        for start in range(0, total_train, batch_size):
            flat = permutation[start : start + batch_size].numpy()
            local_query, action = flat // len(ALPHAS), flat % len(ALPHAS)
            global_query = query_train[local_query]
            candidates = torch.from_numpy(candidate_features[global_query]).to(requested)
            mask = torch.from_numpy(candidate_mask[global_query]).to(requested)
            context = torch.from_numpy(normalized_context[global_query]).to(requested)
            action_tensor = torch.from_numpy(descriptors[action]).to(requested)
            target = torch.from_numpy(y_train[local_query, action].astype(np.float32)).to(requested)
            prediction = model(candidates, mask, context, action_tensor)
            loss = functional.smooth_l1_loss(prediction, target, beta=0.02)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(contract["training"]["max_gradient_norm"])); optimizer.step()
    model.eval(); result = np.empty((len(query_eval), len(ALPHAS)), dtype=np.float64)
    flat_eval = np.arange(len(query_eval) * len(ALPHAS))
    with torch.inference_mode():
        for start in range(0, len(flat_eval), batch_size):
            flat = flat_eval[start : start + batch_size]
            local_query, action = flat // len(ALPHAS), flat % len(ALPHAS)
            global_query = query_eval[local_query]
            prediction = model(
                torch.from_numpy(candidate_features[global_query]).to(requested),
                torch.from_numpy(candidate_mask[global_query]).to(requested),
                torch.from_numpy(normalized_context[global_query]).to(requested),
                torch.from_numpy(descriptors[action]).to(requested),
            )
            result[local_query, action] = prediction.cpu().numpy()
    return result


def query_features_for_scope(
    representation: str,
    static: np.ndarray,
    z_a: np.ndarray | None,
    z_b: np.ndarray | None,
    train_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    if representation not in {"X5", "X6"}:
        return static
    if z_a is None or z_b is None:
        raise ValueError("X5 requires frozen raw query latents")
    return np.column_stack((static, project_latents(z_a, z_b, train_mask, seed))).astype(np.float32)


def inner_score(
    rr: np.ndarray,
    predicted: np.ndarray,
    global_index: int,
) -> tuple[float, float]:
    chosen = select_probe_actions(predicted, global_index)
    rows = np.arange(len(rr))
    gain = float((rr[rows, chosen] - rr[:, global_index]).mean())
    actual_u = rr - rr[:, [global_index]]
    mask = np.arange(len(ALPHAS)) != global_index
    correlation = sign_metrics(actual_u[:, mask], predicted[:, mask])["spearman_pred_u_actual_u"]
    return gain, float(correlation)


def load_inputs(args, contract, contract_path: Path):
    asset_manifest_path = Path(args.asset_dir) / f"{args.pair_id}_query_information_manifest.json"
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    if asset_manifest.get("split") != "dev" or int(asset_manifest.get("test_rows_accessed", -1)) != 0:
        raise RuntimeError("Invalid query-information asset")
    contract_sources = [row for row in asset_manifest.get("sources", []) if row.get("role") == "feature_contract"]
    if len(contract_sources) != 1 or contract_sources[0].get("sha256") != sha256_file(contract_path):
        raise RuntimeError("Query-information asset was built under a different feature contract")
    asset_path = Path(asset_manifest["output"]["path"])
    if sha256_file(asset_path) != asset_manifest["output"]["sha256"]:
        raise RuntimeError("Query-information asset hash mismatch")
    with np.load(asset_path, allow_pickle=False) as payload:
        query_ids = payload["query_id"].astype(str)
        frame = pd.DataFrame(
            {
                "query_id": query_ids,
                "seed": payload["seed"].astype(int),
                "direction": payload["direction"].astype(str),
                "head_id": payload["head_id"].astype(int),
                "relation_id": payload["relation_id"].astype(int),
                "tail_id": payload["tail_id"].astype(int),
            }
        )
        static_x4 = payload["features_x4"].astype(np.float32)
    frame["original_triple_id"] = "h=" + frame.head_id.astype(str) + "|r=" + frame.relation_id.astype(str) + "|t=" + frame.tail_id.astype(str)
    utility_manifest_path = Path(args.utility_manifest_dir) / f"{args.pair_id}_dev_source_manifest.json"
    utility_manifest = json.loads(utility_manifest_path.read_text(encoding="utf-8"))
    if utility_manifest.get("split") != "dev":
        raise RuntimeError("Exact RR utility manifest is not DEV-only")
    query_path = Path(utility_manifest["source_query_rows"]["path"])
    if sha256_file(query_path) != utility_manifest["source_query_rows"]["sha256"]:
        raise RuntimeError("Exact RR source hash mismatch")
    exact = pd.read_csv(query_path, usecols=["query_id", *RR_COLUMNS])
    order = align_indices(exact.query_id.to_numpy(str), query_ids, "exact RR")
    rr = exact.iloc[order][RR_COLUMNS].to_numpy(np.float64)
    feature_fields = representation_features(contract)
    asset_fields = asset_manifest["feature_fields_x4"]
    wanted = feature_fields[args.representation] if args.representation != "X6" else feature_fields["X5"]
    indices = [asset_fields.index(field) for field in wanted]
    static = static_x4[:, indices]
    z_a = z_b = None
    latent_manifest_path = None
    if args.representation in {"X5", "X6"}:
        latent_manifest_path = Path(args.phase4a_root) / args.pair_id / "latent_extraction_manifest.json"
        latent_manifest = json.loads(latent_manifest_path.read_text(encoding="utf-8"))
        if latent_manifest.get("split") != "dev" or int(latent_manifest.get("test_rows_accessed", -1)) != 0:
            raise RuntimeError("Frozen latent manifest is not DEV-only")
        latent_path = Path(latent_manifest["output"]["path"])
        if sha256_file(latent_path) != latent_manifest["output"]["sha256"]:
            raise RuntimeError("Frozen latent hash mismatch")
        with np.load(latent_path, allow_pickle=False) as latent:
            order = align_indices(latent["query_id"].astype(str), query_ids, "latent")
            z_a, z_b = latent["z_a"][order].astype(np.float32), latent["z_b"][order].astype(np.float32)
    candidates = candidate_mask = candidate_manifest_path = None
    if args.representation == "X6":
        candidate_manifest_path = Path(args.candidate_dir) / f"{args.pair_id}_union_top100_manifest.json"
        candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        x6 = contract["representations"]["X6_candidate"]
        if (
            candidate_manifest.get("split") != "dev"
            or int(candidate_manifest.get("test_rows_accessed", -1)) != 0
            or int(candidate_manifest.get("top_k_per_expert", -1)) != int(x6["top_k_per_expert"])
            or int(candidate_manifest.get("maximum_union_size", -1)) != int(x6["maximum_union_size"])
            or candidate_manifest.get("candidate_fields") != x6["candidate_fields"]
            or candidate_manifest.get("candidate_embeddings_used") is not False
            or candidate_manifest.get("candidate_identity_feature_used") is not False
        ):
            raise RuntimeError("X6 candidate asset violates the frozen DEV feature contract")
        contract_hash = sha256_file(contract_path)
        if not any(
            row.get("path") == portable_path(contract_path) and row.get("sha256") == contract_hash
            for row in candidate_manifest.get("sources", [])
        ):
            raise RuntimeError("X6 candidate asset was built under a different feature contract")
        candidate_path = Path(candidate_manifest["output"]["path"])
        if sha256_file(candidate_path) != candidate_manifest["output"]["sha256"]:
            raise RuntimeError("X6 candidate asset hash mismatch")
        with np.load(candidate_path, allow_pickle=False) as candidate:
            order = align_indices(candidate["query_id"].astype(str), query_ids, "candidate")
            candidates = candidate["candidate_features"][order].astype(np.float32)
            candidate_mask = candidate["candidate_mask"][order].astype(bool)
    return frame, rr, static, z_a, z_b, candidates, candidate_mask, [asset_manifest_path, utility_manifest_path, query_path, latent_manifest_path, candidate_manifest_path]


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = load_contract(contract_path)
    exp1_manifest_path = Path(args.exp1_manifest)
    reject_test_path(exp1_manifest_path)
    exp1_manifest = json.loads(exp1_manifest_path.read_text(encoding="utf-8"))
    if exp1_manifest.get("gate", {}).get("decision") != "GO":
        raise RuntimeError("Experiment 1 Available Complementarity Gate did not pass")
    compatible = contract["learner_compatibility"].get(args.representation, [])
    if args.learner not in compatible:
        raise ValueError(f"{args.learner} is not frozen for {args.representation}; expected {compatible}")
    output_dir = Path(args.output_root) / args.pair_id / args.representation.lower() / args.learner
    for path in (contract_path, output_dir, Path(args.asset_dir), Path(args.utility_manifest_dir)):
        reject_test_path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "input_audit.json"
    if (output_dir / "metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    frame, rr, static, z_a, z_b, candidates, candidate_mask, source_paths = load_inputs(args, contract, contract_path)
    dataset = "mkg_w" if args.pair_id.startswith("mkgw_") else "db15k"
    nested = contract["nested_cv"]
    outer_vector, outer_audit = grouped_folds(frame, int(nested["outer_folds"]), int(nested["outer_fold_seed"]))
    exp1_stats = pd.read_csv(args.exp1_stats)
    available = float(exp1_stats.loc[exp1_stats.pair_id == args.pair_id, "available_headroom"].iloc[0])
    configs = list(contract["hyperparameters"][args.learner])
    source_records = []
    for path in [contract_path, exp1_manifest_path, Path(args.exp1_stats), *[item for item in source_paths if item is not None]]:
        source_records.append({"path": portable_path(path), "sha256": sha256_file(path)})
    audit = {
        "schema_version": 1, "experiment": "Experiment 2 — Information–Identifiability Audit",
        "split": "dev", "dataset": dataset, "pair_id": args.pair_id, "representation": args.representation, "learner": args.learner,
        "n_queries": len(frame), "n_original_triples": int(frame.original_triple_id.nunique()),
        "outer_fold_audit": outer_audit, "available_headroom_exp1": available,
        "contract_sha256": sha256_file(contract_path), "sources": source_records,
        "test_rows_accessed": 0, "full_dev_global_used_for_heldout": 0,
        "checkpoint_training": 0, "checkpoint_reselection": 0, "final_policy_development": 0,
        "dry_run": bool(args.dry_run),
    }
    fold_globals = []
    for fold in range(int(nested["outer_folds"])):
        global_index, train_mrr = select_global_alpha(rr, outer_vector != fold)
        fold_globals.append({"outer_fold": fold + 1, "global_index": global_index, "global_alpha": float(ALPHAS[global_index]), "outer_train_global_mrr": train_mrr})
    audit["fold_specific_globals"] = fold_globals
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"[DRY-RUN OK] {args.pair_id} {args.representation} {args.learner}: queries={len(frame)} globals={[row['global_alpha'] for row in fold_globals]}")
        return

    oof = np.full((len(frame), len(ALPHAS)), np.nan, dtype=np.float64)
    global_indices = np.full(len(frame), -1, dtype=np.int16)
    selection_rows, inner_rows, training_rows = [], [], []
    for outer_fold in range(int(nested["outer_folds"])):
        outer_train = outer_vector != outer_fold
        outer_hold = ~outer_train
        global_index = fold_globals[outer_fold]["global_index"]
        descriptors = action_descriptors(global_index)
        target = rr - rr[:, [global_index]]
        inner_frame = frame.loc[outer_train].reset_index(drop=True)
        inner_vector, _ = grouped_folds(inner_frame, int(nested["inner_folds"]), int(nested["inner_fold_seed"]) + outer_fold)
        outer_indices = np.flatnonzero(outer_train)
        latent_inner_cache = {}
        if args.representation in {"X5", "X6"}:
            for inner_fold in range(int(nested["inner_folds"])):
                inner_train_indices = outer_indices[inner_vector != inner_fold]
                projection_seed = int(contract["training"]["model_seed"]) + outer_fold * 100 + inner_fold * 2
                latent_inner_cache[inner_fold] = query_features_for_scope(
                    args.representation, static, z_a, z_b,
                    np.isin(np.arange(len(frame)), inner_train_indices), projection_seed,
                )
        config_scores = []
        for config_index, config in enumerate(configs):
            gains, correlations = [], []
            for inner_fold in range(int(nested["inner_folds"])):
                inner_train_indices = outer_indices[inner_vector != inner_fold]
                inner_valid_indices = outer_indices[inner_vector == inner_fold]
                seed = int(contract["training"]["model_seed"]) + outer_fold * 1000 + config_index * 20 + inner_fold
                if args.representation == "X6":
                    features = latent_inner_cache[inner_fold]
                    predicted = train_predict_set(config, features, candidates, candidate_mask, inner_train_indices, target[inner_train_indices], inner_valid_indices, descriptors, contract, args.device, seed)
                else:
                    features = latent_inner_cache[inner_fold] if args.representation == "X5" else static
                    x_train = expand_tabular(features, inner_train_indices, descriptors)
                    x_valid = expand_tabular(features, inner_valid_indices, descriptors)
                    prediction_flat = train_predict_tabular(args.learner, config, x_train, target[inner_train_indices].reshape(-1), x_valid, contract, args.device, seed)
                    predicted = prediction_flat.reshape(len(inner_valid_indices), len(ALPHAS))
                predicted[:, global_index] = 0.0
                gain, correlation = inner_score(rr[inner_valid_indices], predicted, global_index)
                gains.append(gain); correlations.append(correlation)
                inner_rows.append({"outer_fold": outer_fold + 1, "inner_fold": inner_fold + 1, "config_index": config_index, "config": json.dumps(config, sort_keys=True), "probe_delta_mrr": gain, "spearman": correlation})
            score = (float(np.mean(gains)), float(np.mean(correlations)), -config_index)
            config_scores.append(score)
        selected_index = max(range(len(configs)), key=lambda index: config_scores[index])
        selected_config = configs[selected_index]
        selection_rows.append({"outer_fold": outer_fold + 1, "selected_config_index": selected_index, "selected_config": json.dumps(selected_config, sort_keys=True), "inner_probe_delta_mrr": config_scores[selected_index][0], "inner_spearman": config_scores[selected_index][1], "fold_global_alpha": float(ALPHAS[global_index])})
        seed = int(contract["training"]["model_seed"]) + outer_fold * 1000 + selected_index * 20 + 19
        train_indices, hold_indices = np.flatnonzero(outer_train), np.flatnonzero(outer_hold)
        evaluation_indices = np.concatenate((train_indices, hold_indices))
        if args.representation == "X6":
            projection_seed = int(contract["training"]["model_seed"]) + outer_fold * 100 + 90
            features = query_features_for_scope(args.representation, static, z_a, z_b, outer_train, projection_seed)
            predicted_all = train_predict_set(selected_config, features, candidates, candidate_mask, train_indices, target[train_indices], evaluation_indices, descriptors, contract, args.device, seed)
        else:
            projection_seed = int(contract["training"]["model_seed"]) + outer_fold * 100 + 90
            features = query_features_for_scope(args.representation, static, z_a, z_b, outer_train, projection_seed)
            x_train = expand_tabular(features, train_indices, descriptors)
            x_evaluate = expand_tabular(features, evaluation_indices, descriptors)
            predicted_all = train_predict_tabular(args.learner, selected_config, x_train, target[train_indices].reshape(-1), x_evaluate, contract, args.device, seed).reshape(len(evaluation_indices), len(ALPHAS))
        predicted_all[:, global_index] = 0.0
        predicted_train = predicted_all[: len(train_indices)]
        predicted_hold = predicted_all[len(train_indices) :]
        oof[hold_indices] = predicted_hold
        global_indices[hold_indices] = global_index
        train_chosen = select_probe_actions(predicted_train, global_index)
        train_gain = rr[train_indices, train_chosen] - rr[train_indices, global_index]
        training_rows.append({"outer_fold": outer_fold + 1, "count": len(train_indices), "training_gain": float(train_gain.mean()), "positive_gain_rate": float((train_gain > ZERO_TOLERANCE).mean()), "negative_transfer_rate": float((train_gain < -ZERO_TOLERANCE).mean())})
        print(f"[OUTER] {args.pair_id} {args.representation} {args.learner} fold={outer_fold+1} alpha0={ALPHAS[global_index]:.2f} config={selected_index}", flush=True)
    if not np.isfinite(oof).all() or np.any(global_indices < 0):
        raise RuntimeError("Incomplete outer OOF predictions")
    bootstrap = contract["bootstrap"]
    metrics, chosen = policy_metrics(rr, oof, global_indices, frame.original_triple_id.to_numpy(str), available, int(bootstrap["samples"]), int(bootstrap["seed"]))
    actual_u = rr - rr[np.arange(len(rr)), global_indices][:, None]
    non_global = np.arange(len(ALPHAS))[None, :] != global_indices[:, None]
    metrics.update(sign_metrics(actual_u[non_global], oof[non_global]))
    weighted_training_gain = float(np.average([row["training_gain"] for row in training_rows], weights=[row["count"] for row in training_rows]))
    metrics.update({"training_gain": weighted_training_gain, "oof_gain": metrics["delta_mrr"], "train_oof_generalization_gap": weighted_training_gain - metrics["delta_mrr"]})
    slice_rows = []
    scopes = [("direction", value, frame.direction.astype(str) == value) for value in ("head", "tail")]
    scopes += [("seed", str(seed), frame.seed.astype(int) == seed) for seed in (1, 2, 3)]
    scopes += [("seed_x_direction", f"{seed}_{direction}", (frame.seed.astype(int) == seed) & (frame.direction.astype(str) == direction)) for seed in (1, 2, 3) for direction in ("head", "tail")]
    for offset, (scope, value, mask) in enumerate(scopes):
        result, _ = policy_metrics(rr[mask], oof[mask], global_indices[mask], frame.loc[mask, "original_triple_id"].to_numpy(str), available, int(bootstrap["samples"]), int(bootstrap["seed"]) + offset + 1)
        slice_actual = actual_u[mask]
        slice_non_global = np.arange(len(ALPHAS))[None, :] != global_indices[mask, None]
        result.update(sign_metrics(slice_actual[slice_non_global], oof[mask][slice_non_global]))
        slice_rows.append({"dataset": dataset, "pair_id": args.pair_id, "representation": args.representation, "learner": args.learner, "scope": scope, "value": value, **result})
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "oof_action_predictions.npz"
    np.savez_compressed(predictions_path, query_id=frame.query_id.to_numpy(str), outer_fold=outer_vector + 1, fold_global_index=global_indices, predicted_u=oof.astype(np.float32), chosen_action_index=chosen)
    pd.DataFrame(selection_rows).to_csv(output_dir / "outer_fold_selections.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output_dir / "inner_cv_results.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output_dir / "training_fold_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(output_dir / "seed_direction_metrics.csv", index=False)
    payload = {
        **audit, "dry_run": False, "metrics": metrics,
        "prediction_artifact": {"path": portable_path(predictions_path), "sha256": sha256_file(predictions_path), "predicted_u_sha256": sha256_array(oof.astype(np.float32))},
        "operational_audit": {"test_access": 0, "full_dev_global_for_heldout": 0, "selector_development": 0, "aacpi_resurrection": 0},
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {args.pair_id} {args.representation} {args.learner}: OOF gain={metrics['delta_mrr']:+.6f}")


if __name__ == "__main__":
    main()
