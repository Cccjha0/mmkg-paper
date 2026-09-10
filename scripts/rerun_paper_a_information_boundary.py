"""Prepare or execute isolated B07/C04 reexports and policy refits (no base retraining)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_commands(device):
    manifest = json.loads((ROOT / "outputs/paper_a_safe_correction/efficiency/benchmark_manifest.json").read_text(encoding="utf-8"))
    runs = {}
    for asset in manifest["assets"]:
        path = Path(asset["path"])
        if path.name == "best.ckpt":
            runs[(path.parent.parent.name, int(path.parent.name.rsplit("seed", 1)[1]))] = path.parent.as_posix()
    commands = []
    names = {"mhyper": "M-Hyper", "native": "NativE", "adamf_mat": "AdaMF-MAT"}
    for dataset, prefix in (("mkg_w", "mkgw"), ("db15k", "db15k")):
        for left, right in (("mhyper", "native"), ("mhyper", "adamf_mat"), ("native", "adamf_mat")):
            pair = f"{prefix}_{left}_{right.replace('_mat', '')}"
            base = f"outputs/paper_a_safe_correction/information_boundary_v2/{pair}"
            pairs = []
            for seed in (1, 2, 3):
                pairs += ["--run-pair", f"{runs[(dataset+'_'+left,seed)]}::{runs[(dataset+'_'+right,seed)]}"]
            common = ["--pair-name", pair, "--expert-a-name", names[left], "--expert-b-name", names[right], *pairs, "--device", device]
            ranking = ["scripts/eval_heterogeneous_complementarity.py", *common]
            dev = base + "/full_ranking"
            cross = base + "/baseline_crossfit"
            p3 = base + "/p3_ablation"
            lock = base + "/dev_lock"
            test = base + "/test_full_ranking"
            applied = base + "/test_anchored"
            selection = dev + "/selection.json"
            commands += [
                [*ranking, "--split", "dev", "--output-dir", dev],
                ["scripts/crossfit_heterogeneous_dev_policies.py", "--query-rows", dev + "/dev_query_rows.csv", "--selection-json", selection, "--output-dir", cross],
                ["scripts/ablate_anchored_dynamic.py", "--query-rows", cross + "/dev_crossfit_query_rows.csv", "--selection-json", selection, "--output-dir", p3],
                ["scripts/lock_apply_anchored_dynamic.py", "lock", "--dev-query-rows", dev + "/dev_query_rows.csv", "--selection-json", selection, "--crossfit-summary", p3 + "/dev_p3_summary.json", "--output-dir", lock],
                [*ranking, "--split", "test", "--selection-json", selection, "--export-alpha-grid", "--output-dir", test],
                ["scripts/lock_apply_anchored_dynamic.py", "apply", "--test-query-rows", test + "/test_query_rows.csv", "--lock-json", lock + "/anchored_dev_lock.json", "--output-dir", applied],
            ]
            if left == "mhyper":
                dyna = ["scripts/eval_openbg_dynasemble.py", "--pair-name", pair + "_dynasemble",
                        "--expert-a-name", names[left], "--expert-b-name", names[right], *pairs,
                        "--device", device, "--baseline-selection-json", selection, "--output-dir", base + "/dynasemble"]
                commands += [[*dyna, "--stage", "dev", "--reference-query-rows", dev + "/dev_query_rows.csv"],
                             [*dyna, "--stage", "test", "--reference-query-rows", test + "/test_query_rows.csv",
                              "--comparison-query-rows", applied + "/test_locked_query_rows.csv"]]
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    commands = build_commands(args.device)
    required = ("manifest.json", "train.tsv", "valid.tsv", "test.tsv", "entity2id.json",
                "relation2id.json", "text_feat.pt", "img_feat.pt", "has_text.pt", "has_img.pt")
    missing = [f"data/datasets/{dataset}/processed/{name}" for dataset in ("mkg_w", "db15k")
               for name in required if not (ROOT / f"data/datasets/{dataset}/processed/{name}").exists()]
    out = ROOT / "outputs/paper_a_safe_correction/information_boundary/reexport_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"commands": commands, "missing_assets": missing,
                              "scope": "Six pairs: reexport, grouped DEV refit/relock, TEST; four main DynaSemble refits. Downstream figures/statistics/timing still need regeneration.",
                              "status": "blocked_missing_data" if missing else "ready_not_executed"}, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(commands)} commands: {out}", flush=True)
    if args.execute:
        if missing:
            raise SystemExit("Cannot reexport; missing canonical assets: " + ", ".join(missing))
        for command in commands:
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
