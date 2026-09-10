"""Read-only upstream checkout audit and minimal tensor-core migration diff."""
import argparse
import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "48d66b915f64899798f736129fa8c4d0a40fdb78"
FILES = ("NBFNet/script/selector.py", "NBFNet/script/train_selector.py", "NBFNet/nbfnet/util.py",
         "NBFNet/nbfnet/tasks.py", "NBFNet/config/wn18rr.yaml", "NBFNet/config/fb15k237.yaml", "NBFNet/config/codex.yaml")


def audit(upstream, output):
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT or subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Upstream must be a clean checkout of the pinned commit")
    records = {}
    for file in FILES:
        raw = (upstream / file).read_bytes()
        records[file] = {"sha256_bytes": hashlib.sha256(raw).hexdigest(),
                         "sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
                         "url": f"https://github.com/dair-iitd/KGC-Ensemble/blob/{COMMIT}/{file}"}
    source = (upstream / FILES[0]).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Selector")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    method = methods["get_features_and_normalize"]
    records[FILES[0]]["feature_function_lines"] = [method.lineno, method.end_lineno]
    source_features = ast.get_source_segment(source, method)
    # A review diff of actual executable snippets, not a purported whole-project patch.
    current = (ROOT / "scripts/eval_openbg_dynasemble.py").read_text(encoding="utf-8")
    current_tree = ast.parse(current)
    normalization = next(n for n in current_tree.body if isinstance(n, ast.FunctionDef) and n.name == "normalize_and_features")
    diff = "".join(difflib.unified_diff((source_features.rstrip()+"\n").splitlines(True), (ast.get_source_segment(current, normalization).rstrip()+"\n").splitlines(True),
                                     fromfile="upstream/get_features_and_normalize", tofile="local/normalize_and_features"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "normalization_migration.diff").write_text(diff, encoding="utf-8")
    # Extract the actual upstream first-MLP construction and local controlled constructor.
    init_text = ast.get_source_segment(source, methods["__init__"])
    start = init_text.index("        mlp = []")
    end = init_text.index("        self.mlp = nn.Sequential(*mlp)") + len("        self.mlp = nn.Sequential(*mlp)")
    local = (ROOT / "router/dynasemble_controls.py").read_text(encoding="utf-8")
    local_cls = next(n for n in ast.parse(local).body if isinstance(n, ast.ClassDef) and n.name == "ControlSelector")
    local_core = "\n\n".join(ast.get_source_segment(local, n) for n in local_cls.body
                             if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "preactivation", "forward")) + "\n"
    diff = "".join(difflib.unified_diff((init_text[start:end]+"\n").splitlines(True), local_core.splitlines(True),
                                     fromfile="upstream/first_selector_MLP", tofile="local/ControlSelector_core"))
    (output / "selector_migration.diff").write_text(diff, encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"commit": commit, "clean": True, "files": records,
        "scope": "Tensor-core snippets; training/data/initialization changes are enumerated separately in the protocol table."}, indent=2)+"\n", encoding="utf-8")
    print(f"Verified {len(FILES)} upstream files at {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "external/KGC-Ensemble")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/audits/dynasemble_source")
    args = parser.parse_args()
    audit(args.upstream, args.output_dir)
