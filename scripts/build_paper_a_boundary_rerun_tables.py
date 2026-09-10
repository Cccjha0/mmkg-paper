"""Build only corrected manuscript tables from the audited v2 exports; no fitting."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_paper_a_boundary_rerun import KEY, digest

AUDIT = ROOT / "outputs/paper_a_safe_correction/information_boundary_rerun_audit"
INPUT = ROOT / "outputs/paper_a_safe_correction/information_boundary_v2"
OUTPUT = ROOT / "paper_a_draft/tables/rerun"
PAIRS = {"mkgw_mhyper_native": "W-N", "mkgw_mhyper_adamf": "W-A",
         "db15k_mhyper_native": "D-N", "db15k_mhyper_adamf": "D-A",
         "mkgw_native_adamf": "W-NA", "db15k_native_adamf": "D-NA"}


def table(name, caption, header, rows, columns):
    text = ("\\begin{table}[htbp]\n\\centering\\small\n\\caption{" + caption + "}\n"
            "\\label{tab:rerun-" + name + "}\n\\begin{tabular}{" + columns + "}\n\\toprule\n"
            + " & ".join(header) + "\\\\\n\\midrule\n"
            + "\n".join(" & ".join(row) + "\\\\" for row in rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUTPUT / f"{name}.tex").write_text(text, encoding="utf-8")


def main():
    audit = json.loads((AUDIT / "audit.json").read_text())
    assert audit["status"] == "artifact_checks_passed_with_continuous_repeatability_caveat"
    assert not audit["failures"]
    for file, sha in audit["summary_hashes"].items():
        assert digest(AUDIT / file) == sha, file
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(AUDIT / "method_summary.csv")
    summary = summary[summary.split == "test"].set_index(["pair", "method"])
    ablations = pd.read_csv(AUDIT / "frozen_ablation.csv").set_index(["pair", "method"])
    main_rows, effects, ablation_rows, locks, consistency, hits, dev_rows = [], [], [], [], [], [], []
    details = {}
    for pair, label in PAIRS.items():
        base = INPUT / pair
        lock = json.loads((base / "dev_lock/anchored_dev_lock.json").read_text())
        frame = pd.read_csv(base / "test_anchored/test_locked_query_rows.csv")
        entry = next(p for p in audit["pairs"] if p["pair"] == pair)
        assert digest(base / "test_anchored/test_locked_query_rows.csv") == entry["current_artifact_hashes"]["test_anchored/test_locked_query_rows.csv".replace("/", "\\") if "test_anchored\\test_locked_query_rows.csv" in entry["current_artifact_hashes"] else "test_anchored/test_locked_query_rows.csv"]
        delta = frame.rr_anchored_locked - frame.rr_global
        by_seed = delta.groupby(frame.seed).mean()
        by_direction = delta.groupby(frame.direction).mean()
        details[pair] = {"label": label, "seed_deltas": by_seed.to_dict(), "direction_deltas": by_direction.to_dict(),
                         "fallback_rate": float(frame.anchored_fallback.astype(str).str.lower().isin(["true", "1"]).mean()),
                         "primary_mrr": float(frame.rr_a.mean()), "secondary_mrr": float(frame.rr_b.mean()),
                         "equal_mrr": float(frame.rr_equal.mean())}
        row = summary.loc[(pair, "ADC")]
        lo, hi = ast.literal_eval(row.delta_mrr_ci95)
        effects.append([label, f"{row.delta_mrr_vs_global:+.6f}", f"[{lo:+.6f}, {hi:+.6f}]", f"{100*row.harmful_query_rate:.3f}"])
        locks.append([label, f"{lock['alpha0']:.2f}", f"{lock['beta']:.2f}", f"{lock['confidence_threshold']:.2f}", f"{100*details[pair]['fallback_rate']:.2f}"])
        consistency.append([label, *[f"{v:+.6f}" for v in by_seed],
                            f"{by_direction['head']:+.6f}", f"{by_direction['tail']:+.6f}"])
        for method, col in (("Global", "rr_global"), ("ADC", "rr_anchored_locked")):
            values = frame[col].to_numpy()
            hits.append([label, method, f"{values.mean():.6f}",
                         *[f"{(values >= 1/k).mean():.6f}" for k in (1, 3, 10)]])
        oof = pd.read_csv(base / "p3_ablation/dev_p3_selected_query_rows.csv")
        dev_base = pd.read_csv(base / "full_ranking/dev_query_rows.csv", usecols=KEY)
        assert len(oof) == len(dev_base) == len(oof.merge(dev_base, on=KEY, validate="one_to_one"))
        assert oof.groupby(["head_id", "relation_id", "tail_id"]).fold.nunique().max() == 1
        assert oof.fold.nunique() == 5
        p3 = json.loads((base / "p3_ablation/dev_p3_summary.json").read_text())
        selected = next(r for r in p3["results"] if r["config_id"] == "expanded_selected")
        # The exporter uses sequential Python summation; pandas uses a different reduction.
        assert abs(oof.rr_method.mean() - selected["mrr"]) < 1e-12
        oof_delta = oof.rr_method - oof.rr_global_crossfit
        dev_rows.append([label, f"{oof.rr_global_crossfit.mean():.6f}", f"{oof.rr_method.mean():.6f}", f"{oof_delta.mean():+.6f}",
                         str(int((oof_delta.groupby(oof.seed).mean() > 0).sum())) + "/3",
                         str(int((oof_delta.groupby(oof.direction).mean() > 0).sum())) + "/2"])
        if "mhyper" in pair:
            main_rows.append([label, *[f"{summary.loc[(pair, method)].mrr:.6f}" for method in ("Global", "Relation", "Query-soft", "DynaSemble", "ADC")]])
            for variant, title in (("full", "Full"), ("no_bound", r"$\beta=1$"), ("no_fallback", r"$\tau=0$")):
                r = ablations.loc[(pair, variant)]
                ablation_rows.append([label, title, f"{r.delta_mrr_vs_global:+.6f}", f"{100*r.harmful_query_rate:.3f}", f"{r.mean_harm:.6f}"])
            dyna = pd.read_csv(base / "dynasemble/test_query_rows.csv")
            details[pair]["dynasemble_seed_deltas"] = (dyna.rr_dynasemble-dyna.rr_global).groupby(dyna.seed).mean().to_dict()
            details[pair]["dynasemble_zero_weight_fraction"] = (dyna.weight_expert_a == 0).groupby(dyna.seed).mean().to_dict()
    table("main", "Corrected TEST MRR. W: MKG-W; D: DB15K; N: M-Hyper + NativE; A: M-Hyper + AdaMF-MAT. All selectors were refitted on corrected DEV exports.",
          ["Pair", "Global", "Relation", "Query-soft", "DynaSemble", "ADC"], main_rows, "lrrrrr")
    table("effects", r"Corrected ADC TEST effects relative to Global. Intervals use 10,000 original-triple clustered bootstrap resamples. NA denotes NativE + AdaMF-MAT. Harm uses all observations.",
          ["Pair", r"$\Delta$MRR", r"95\% interval", r"Harm (\%)"], effects, "lrlr")
    table("ablation", "Frozen TEST counterfactuals. The classifier, anchor and other control remain fixed; no TEST fitting or selection is performed. Mean harm conditions on a harmful observation.",
          ["Pair", "Variant", r"$\Delta$MRR", r"Harm (\%)", "Mean harm"], ablation_rows, "llrrr")
    table("locks", r"Corrected DEV locks and observed TEST fallback. $\alpha_0$ is the primary weight.",
          ["Pair", r"$\alpha_0$", r"$\beta$", r"$\tau$", r"Fallback (\%)"], locks, "lrrrr")
    table("consistency", "Corrected ADC TEST increments, pooled within each seed or direction.",
          ["Pair", "Seed 1", "Seed 2", "Seed 3", "Head", "Tail"], consistency, "lrrrrr")
    table("hits", "Corrected TEST ranking metrics. Hits are reconstructed from exact reciprocal ranks.",
          ["Pair", "Method", "MRR", "Hits@1", "Hits@3", "Hits@10"], hits, "llrrrr")
    table("dev", "Corrected grouped out-of-fold DEV evaluation, using fold-specific expanded-policy selection. Positive-cell counts refer to the ADC increment over each fold's Global reference.",
          ["Pair", "Global", "ADC", r"$\Delta$MRR", "Seeds", "Directions"], dev_rows, "lrrrrr")
    (AUDIT / "manuscript_details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    manifest = {"version": "information_boundary_v2", "sources": {}, "tables": {}}
    for p in (AUDIT / "audit.json", AUDIT / "method_summary.csv", AUDIT / "frozen_ablation.csv", AUDIT / "manuscript_details.json"):
        manifest["sources"][p.relative_to(ROOT).as_posix()] = digest(p)
    for base in sorted(INPUT.iterdir()):
        if base.name in PAIRS:
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.suffix in (".csv", ".json", ".pkl", ".pt"):
                    manifest["sources"][p.relative_to(ROOT).as_posix()] = digest(p)
    for p in OUTPUT.glob("*.tex"):
        manifest["tables"][p.name] = digest(p)
    (ROOT / "paper_a_draft/rerun_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tables": len(manifest["tables"]), "pairs": len(details)}))


if __name__ == "__main__":
    main()
