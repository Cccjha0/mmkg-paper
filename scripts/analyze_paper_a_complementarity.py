"""A01/A04/E01: cached, retrospective diagnostics; no model fitting or policy choice."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT = ROOT / "outputs/paper_a_safe_correction/information_boundary_v2"
OUT = ROOT / "outputs/paper_a_safe_correction/complementarity_review_v1"
PROTOCOL = ROOT / "docs/protocols/paper_a_complementarity_diagnostics.md"
PAIRS = {"mkgw_mhyper_native": "W-N", "mkgw_mhyper_adamf": "W-A",
         "db15k_mhyper_native": "D-N", "db15k_mhyper_adamf": "D-A",
         "mkgw_native_adamf": "W-NA", "db15k_native_adamf": "D-NA"}
KEY = ["query_id", "seed", "direction", "head_id", "relation_id", "tail_id"]
TRIPLE = ["head_id", "relation_id", "tail_id"]
SUPPORT = ["neither", "image-only", "text-only", "both"]
RANK_BINS = ["equal", "(1,2]", "(2,4]", ">4"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def observable_support(frame, has_img, has_text):
    """Lookup ONLY the known entity; unknown endpoint IDs need not even be valid."""
    require(frame.direction.isin(["head", "tail"]).all(), "Unknown prediction direction")
    known = np.where(frame.direction == "tail", frame.head_id, frame.tail_id).astype(np.int64)
    require(((known >= 0) & (known < len(has_img))).all(), "Known entity out of range")
    require(len(has_img) == len(has_text), "Mask universe mismatch")
    code = np.asarray(has_img, dtype=bool)[known].astype(int) + 2 * np.asarray(has_text, dtype=bool)[known]
    return known, np.asarray(SUPPORT)[code]


def rank_ratio_bin(rank_a, rank_b):
    lo, hi = np.minimum(rank_a, rank_b), np.maximum(rank_a, rank_b)
    return np.select([hi == lo, hi <= 2 * lo, hi <= 4 * lo], RANK_BINS[:3], default=RANK_BINS[3])


def summarize(frame, n_total):
    n = len(frame)
    if not n:
        return {"n_observations": 0, "n_triples": 0, "n_directional_queries": 0,
                "mass": 0., "delta_contribution": 0.}
    a, b, ref, adc = (frame[c].to_numpy() for c in ["rr_a", "rr_b", "reference_rr", "adc_rr"])
    ra, rb = frame.rank_a.to_numpy(), frame.rank_b.to_numpy()
    delta = adc - ref
    oracle = np.maximum(a, b).mean()
    return dict(n_observations=n, n_triples=len(frame[TRIPLE].drop_duplicates()),
                n_directional_queries=len(frame[TRIPLE + ["direction"]].drop_duplicates()),
                mass=n/n_total, primary_mrr=a.mean(), secondary_mrr=b.mean(),
                primary_advantage=a.mean()-b.mean(), a_win_rate=(a>b).mean(), b_win_rate=(b>a).mean(),
                tie_rate=(a==b).mean(), oracle_mrr=oracle, best_single_mrr=max(a.mean(), b.mean()),
                oracle_gap=oracle-max(a.mean(), b.mean()), oracle_minus_primary=oracle-a.mean(),
                rank_disagree_rate=(ra!=rb).mean(), mean_abs_log2_rank_ratio=np.abs(np.log2(ra/rb)).mean(),
                gold_hit10_disagree_rate=((ra<=10)!=(rb<=10)).mean(), global_mrr=ref.mean(),
                adc_mrr=adc.mean(), delta_mrr=delta.mean(), delta_contribution=delta.sum()/n_total,
                mean_gain=np.maximum(delta,0).mean(), mean_loss=np.maximum(-delta,0).mean(),
                harm_rate=(delta<0).mean(), gain_rate=(delta>0).mean(),
                action_rate=(frame.applied_alpha != frame.anchor_alpha).mean(),
                fallback_rate=frame.explicit_fallback.mean())


def join_oof(raw, oof):
    require(not raw.query_id.duplicated().any() and not oof.query_id.duplicated().any(), "Duplicate query identity")
    require(len(raw) == len(oof), "OOF coverage differs")
    merged = raw.merge(oof, on=KEY, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_oof"))
    require((merged._merge == "both").all(), "OOF join lost query metadata")
    require(np.allclose(merged.rr_a, merged.rr_a_oof, rtol=0, atol=1e-12), "OOF endpoint changed")
    require(merged.groupby(TRIPLE).fold.nunique().eq(1).all(), "Triple crosses OOF folds")
    require(set(merged.fold) == {1,2,3,4,5}, "Missing OOF fold")
    return merged.drop(columns=["_merge", "rr_a_oof"]).rename(columns={
        "rr_method":"adc_rr", "rr_global_crossfit":"reference_rr", "alpha_applied":"applied_alpha",
        "alpha0":"anchor_alpha", "fallback":"explicit_fallback"})


def main():
    import torch
    from ml.training.src.data.feature_bundle import _sha256_tensor

    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    manifest_paths = [ROOT / "paper_a_draft/rerun_source_manifest.json",
                      ROOT / "paper_a_draft/data_checkpoint_source_manifest.json"]
    expected = {}
    for path in manifest_paths:
        expected.update(json.loads(path.read_text())["sources"])

    def bind(path, old=True):
        rel = path.relative_to(ROOT).as_posix()
        if rel not in sources:
            sources[rel] = sha(path)
        if old:
            require(rel in expected and sources[rel] == expected[rel], "Previously bound source changed: " + rel)
        return path

    for path in manifest_paths + [PROTOCOL, Path(__file__), ROOT / "ml/training/src/data/feature_bundle.py"]:
        bind(path, False)
    cfg_path = bind(ROOT / "configs/paper_a_dynasemble_controls.json")
    configs = json.loads(cfg_path.read_text())["pairs"]
    masks, mask_provenance = {}, []
    for dataset in ("mkg_w", "db15k"):
        config = next(r for r in configs if r["dataset"] == dataset)
        run = next(r for r in config["runs"] if r["seed"] == 1)["expert_a_run"]
        path = bind(ROOT / run / "best.ckpt")
        dataset_path = bind(ROOT / f"outputs/paper_a_safe_correction/data_checkpoint_review_v1/{dataset}_split_manifest.json")
        manifest = json.loads(dataset_path.read_text())["training_manifest"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        fingerprints = {k:_sha256_tensor(state[k]) for k in ("has_img", "has_text")}
        for k,v in fingerprints.items():
            require(v == manifest["hashes"]["canonical_features"][k], "Modality mask hash mismatch")
        masks[dataset] = tuple(state[k].numpy().copy() for k in ("has_img", "has_text"))
        mask_frame = pd.DataFrame({"entity_id":np.arange(len(masks[dataset][0])),
                                   "has_img":masks[dataset][0].astype(int), "has_text":masks[dataset][1].astype(int)})
        mask_frame.to_csv(OUT / f"{dataset}_entity_support.csv.gz", index=False,
                          compression={"method":"gzip", "mtime":0})
        counts = {SUPPORT[i]:int(((mask_frame.has_img+2*mask_frame.has_text)==i).sum()) for i in range(4)}
        mask_provenance.append(dict(dataset=dataset, checkpoint=path.relative_to(ROOT).as_posix(),
                                    checkpoint_sha256=sha(path), tensor_sha256=fingerprints, entity_counts=counts))
        del state

    overall, strata, relations, checks = [], [], [], []
    raw_cols = KEY + ["split", "rank_a", "rank_b", "rr_a", "rr_b", "rr_oracle", "score_information_contract"]
    for pair,label in PAIRS.items():
        folder = INPUT / pair
        dataset = "mkg_w" if pair.startswith("mkgw") else "db15k"
        for split in ("dev_oof", "test"):
            if split == "dev_oof":
                raw = pd.read_csv(bind(folder / "full_ranking/dev_query_rows.csv"), usecols=raw_cols)
                oof = pd.read_csv(bind(folder / "p3_ablation/dev_p3_selected_query_rows.csv"), usecols=
                                  KEY + ["fold", "rr_a", "rr_method", "rr_global_crossfit", "alpha_applied", "alpha0", "fallback"])
                frame = join_oof(raw, oof)
                split_name = "dev"
            else:
                frame = pd.read_csv(bind(folder / "test_anchored/test_locked_query_rows.csv"), usecols=
                                    raw_cols + ["rr_anchored_locked", "rr_global", "alpha_anchored_locked", "alpha0_locked", "anchored_fallback"])
                frame = frame.rename(columns={"rr_anchored_locked":"adc_rr", "rr_global":"reference_rr",
                      "alpha_anchored_locked":"applied_alpha", "alpha0_locked":"anchor_alpha", "anchored_fallback":"explicit_fallback"})
                split_name = "test"
            require(frame.explicit_fallback.isin([0,1,False,True]).all(), "Invalid fallback flag")
            values = frame[["adc_rr","reference_rr","applied_alpha","anchor_alpha"]].to_numpy()
            require(np.isfinite(values).all() and ((values>=0)&(values<=1)).all(), "Invalid policy output")
            require(frame.split.eq(split_name).all(), "Split mixed")
            require(frame.score_information_contract.eq("unfiltered_features_and_normalization_v2").all(), "Old geometry contract")
            require(set(frame.seed)=={1,2,3} and set(frame.direction)=={"head","tail"}, "Seed/direction coverage")
            require(frame.groupby(TRIPLE).size().eq(6).all(), "Missing seed-direction observation")
            for k in ("a", "b"):
                ranks = frame["rank_"+k].to_numpy()
                require(np.isfinite(ranks).all() and (ranks>=1).all() and (ranks==ranks.astype(int)).all(), "Invalid rank")
                require(np.allclose(1/ranks, frame["rr_"+k], rtol=0, atol=1e-12), "Endpoint RR/rank mismatch")
            require(np.allclose(frame.rr_oracle, np.maximum(frame.rr_a,frame.rr_b), rtol=0, atol=1e-12), "Oracle differs from endpoints")
            frame["known_entity_id"], frame["support"] = observable_support(frame, *masks[dataset])
            frame["winner"] = np.select([frame.rr_a>frame.rr_b,frame.rr_b>frame.rr_a],["A","B"],default="tie")
            frame["rank_ratio_bin"] = rank_ratio_bin(frame.rank_a.to_numpy(), frame.rank_b.to_numpy())
            frame["explicit_fallback"] = frame.explicit_fallback.astype(bool)
            total = summarize(frame, len(frame))
            if split == "dev_oof":
                recorded = pd.read_csv(bind(folder / "p3_ablation/dev_p3_results.csv")).set_index("config_id")
                for actual, expected_mrr in ((total["adc_mrr"],recorded.loc["expanded_selected","mrr"]),
                                             (total["global_mrr"],recorded.loc["global","mrr"])):
                    require(np.isclose(actual,expected_mrr,rtol=0,atol=1e-12), "OOF published aggregate differs")
            else:
                recorded = pd.read_csv(bind(folder / "test_anchored/test_locked_results.csv"))
                require(any(np.isclose(recorded.mrr,total["adc_mrr"],rtol=0,atol=1e-12)), "TEST ADC aggregate differs")
            overall.append(dict(pair=pair,label=label,split=split,**total))
            decompositions = {}
            for kind,values in (("winner",["A","B","tie"]),("rank_ratio_bin",RANK_BINS),
                                ("direction",["head","tail"]),("support",SUPPORT)):
                rows = [dict(pair=pair,label=label,split=split,stratum=kind,value=value,
                             **summarize(frame[frame[kind]==value],len(frame))) for value in values]
                strata.extend(rows)
                require(sum(r["n_observations"] for r in rows)==len(frame), "Stratum coverage failure")
                require(np.isclose(sum(r["delta_contribution"] for r in rows),total["delta_mrr"],rtol=0,atol=1e-12), "Delta decomposition failure")
                decompositions[kind] = True
            for columns in (["relation_id"],["relation_id","direction"]):
                for keys, group in frame.groupby(columns):
                    keys = keys if isinstance(keys,tuple) else (keys,)
                    parts = dict(zip(columns,keys))
                    relations.append(dict(pair=pair,label=label,split=split,scope="relation" if len(columns)==1 else "relation_direction",
                                          **parts,**summarize(group,len(frame))))
            require(np.isclose(total["oracle_minus_primary"],np.maximum(frame.rr_b-frame.rr_a,0).mean(),rtol=0,atol=1e-12), "Opportunity identity failed")
            require(np.isclose(total["delta_mrr"],total["mean_gain"]-total["mean_loss"],rtol=0,atol=1e-12), "Gain/loss identity failed")
            checks.append(dict(pair=pair,split=split,source_identity_verified=True,
                               seed_direction_coverage_verified=True, endpoint_rank_rr_verified=True,
                               recorded_aggregate_verified=True,
                               oof_identity_verified=split=="dev_oof", decompositions=decompositions))
            print(f"{label} {split}: B wins {100*total['b_win_rate']:.2f}%; oracle gap {total['oracle_gap']:.6f}; ADC {total['delta_mrr']:+.6f}",flush=True)

    outputs = {}
    for name,records in (("pair_summary.csv",overall),("conditional_summary.csv",strata),
                         ("relation_summary.csv.gz",relations)):
        path = OUT / name
        pd.DataFrame(records).to_csv(path,index=False,compression={"method":"gzip","mtime":0} if name.endswith(".gz") else None)
        outputs[name] = sha(path)
    for dataset in masks:
        name = f"{dataset}_entity_support.csv.gz"
        outputs[name] = sha(OUT/name)
    provenance = OUT / "modality_support_provenance.json"
    provenance.write_text(json.dumps(mask_provenance,indent=2)+"\n",encoding="utf-8",newline="\n")
    outputs[provenance.name] = sha(provenance)
    audit = dict(version="complementarity_review_v1",status="complementarity_checks_passed",failures=[],
                 pairs=6,split_cells=12,checks=checks,training_runs=0,new_policy_selection=False,
                 test_used_for_new_selection=False,selector_features_changed=False,
                 modality_support_scope="known entity only; diagnostic, never selector input",
                 evidence_status="retrospective; DEV uses grouped OOF, TEST uses locked policies",
                 rank_disagreement_scope="gold endpoint rank disagreement, not complete candidate-order distance",
                 sources=sources,outputs=outputs)
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8",newline="\n")


if __name__ == "__main__":
    main()
