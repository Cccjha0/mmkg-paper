"""Apply an existing DEV lock with pre-prediction fallback; no fitting/selection.

This is the guarded inference entry point. Frozen historical experiment scripts
remain available for exact source provenance; this path reuses their model/lock.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.anchored_inference import apply_locked_features
from router.query_geometry import QUERY_GEOMETRY_FIELDS
from router.information_boundary import require_score_information_contract
from scripts.analyze_paper_a_conservative_radius import sha, require


def apply_frame(frame, model, lock):
    require_score_information_contract(lock)
    require(tuple(lock['query_geometry_fields']) == tuple(QUERY_GEOMETRY_FIELDS), 'Wrong geometry schema')
    require(all(field in frame for field in QUERY_GEOMETRY_FIELDS), 'Missing feature column')
    matrix = frame[list(QUERY_GEOMETRY_FIELDS)].to_numpy(dtype=np.float64)
    out = apply_locked_features(model, matrix, alpha0=lock['alpha0'], beta=lock['beta'],
                                threshold=lock['confidence_threshold'], alphas=lock['alpha_grid'])
    keys = [k for k in ('split','seed','direction','head_id','relation_id','tail_id') if k in frame]
    result = frame[keys].copy().reset_index(drop=True)
    for name,value in out.items():
        result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-rows', required=True)
    parser.add_argument('--lock-json', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    lock_path, output = Path(args.lock_json), Path(args.output)
    require(not output.exists(), 'Output exists; choose a new output path')
    lock = json.loads(lock_path.read_text())
    model_path = lock_path.parent/lock['model_file']
    require(sha(model_path)==lock['model_sha256'], 'Changed locked model')
    with model_path.open('rb') as handle:
        model = pickle.load(handle)
    with threadpool_limits(1):
        result = apply_frame(pd.read_csv(args.query_rows, float_precision='round_trip'), model, lock)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output,index=False)
    print(f'Applied {len(result)} rows; skipped invalid rows: {int(result.invalid.sum())}; no fitting or selection.')


if __name__ == '__main__':
    main()
