"""B08/B10 local replay of frozen selectors, anchors and cached endpoints."""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.apply_anchored_safe import apply_frame
from scripts.analyze_paper_a_conservative_radius import INPUT, PAIRS, GRID_COLUMNS, sha, require
from scripts.crossfit_heterogeneous_dev_policies import best_alpha, assign_grouped_folds, triple_key

OUT=ROOT/'outputs/paper_a_safe_correction/inference_contract_review_v1'
ALL_PAIRS={**PAIRS,'mkgw_native_adamf':'W-NA','db15k_native_adamf':'D-NA'}
CODE=('router/anchored_inference.py','router/endpoint_audit.py','scripts/apply_anchored_safe.py',
      'scripts/audit_paper_a_inference_contract.py','scripts/audit_paper_a_endpoint_contract.py','tests/test_guarded_inference_endpoints.py',
      'scripts/lock_apply_anchored_dynamic.py','scripts/ablate_anchored_dynamic.py','scripts/crossfit_anchored_dynamic.py',
      'scripts/crossfit_heterogeneous_dev_policies.py','scripts/eval_heterogeneous_complementarity.py',
      'ml/training/src/eval/filtered_ranking.py','router/query_geometry.py',
      'docs/protocols/paper_a_inference_endpoint_contract_review.md','tests/test_endpoint_audit_runner.py')


def main():
    expected_path=ROOT/'paper_a_draft/rerun_source_manifest.json'
    expected=json.loads(expected_path.read_text())['sources']
    sources={p:sha(ROOT/p) for p in CODE}
    sources[expected_path.relative_to(ROOT).as_posix()]=sha(expected_path)
    def bound(path,desired=None):
        rel=path.relative_to(ROOT).as_posix(); value=sha(path)
        require(value==(expected[rel] if desired is None else desired),'Changed input: '+rel)
        sources[rel]=value;return path
    def frame(path):return pd.read_csv(bound(path),float_precision='round_trip')
    checks=[]; anchors=[]; version_warnings=set()
    for pair,label in ALL_PAIRS.items():
        folder=INPUT/pair
        lock=json.loads(bound(folder/'dev_lock/anchored_dev_lock.json').read_text())
        model_path=bound(folder/'dev_lock'/lock['model_file'],lock['model_sha256'])
        with warnings.catch_warnings(record=True) as caught, model_path.open('rb') as handle:
            model=pickle.load(handle)
        version_warnings.update(str(w.message) for w in caught)
        for split,path in [('dev',folder/'dev_lock/dev_locked_query_rows.csv'),('test',folder/'test_anchored/test_locked_query_rows.csv')]:
            original=frame(path)
            out=apply_frame(original,model,lock)
            require(not out.invalid.any(),'New guard rejects an observed row; investigate before replacing results')
            for new,old in [('decision','anchored_decision'),('probability_a','anchored_probability_a'),('continuous','alpha_anchored_continuous')]:
                np.testing.assert_allclose(out[new],original[old],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(out.applied,original.alpha_anchored_locked)
            np.testing.assert_array_equal(out.query_soft_applied,original.alpha_query_soft_locked)
            np.testing.assert_array_equal(out.fallback,original.anchored_fallback.astype(bool))
            grid=original[GRID_COLUMNS].to_numpy()
            for method,column in [('applied','rr_anchored_locked'),('query_soft_applied','rr_query_soft_locked')]:
                rr=grid[np.arange(len(grid)),np.rint(20*out[method]).astype(int)]
                np.testing.assert_allclose(rr,original[column],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(grid[:,0],original.rr_b)
            np.testing.assert_array_equal(grid[:,-1],original.rr_a)
            checks.append(dict(pair=pair,split=split,rows=len(original),guarded_rows=int(out.invalid.sum()),
                               max_decision_error=float(np.abs(out.decision-original.anchored_decision).max()),
                               adc_weights_exact=True,query_soft_weights_exact=True,fallback_exact=True,grid_endpoints_exact=True))
            if split=='dev':
                winner,mrr=best_alpha(original.to_dict('records'),tuple(lock['alpha_grid']))
                require(winner==lock['alpha0'],'Full-DEV anchor mismatch')
                anchors.append(dict(pair=pair,scope='full_dev',fold=0,alpha=winner,mrr=mrr))
        base=frame(folder/'baseline_crossfit/dev_crossfit_query_rows.csv')
        rows=base.to_dict('records'); assignment,_=assign_grouped_folds(rows,5,20260901)
        folds=np.asarray([assignment[triple_key(r)] for r in rows])
        settings=frame(folder/'p3_ablation/dev_p3_selected_by_fold.csv').set_index('fold')
        for fold in range(5):
            winner,mrr=best_alpha([r for r,f in zip(rows,folds) if f!=fold],tuple(lock['alpha_grid']))
            require(winner==settings.loc[fold+1].alpha0,'OOF anchor mismatch')
            anchors.append(dict(pair=pair,scope='oof_train',fold=fold+1,alpha=winner,mrr=mrr))
        print('[REPLAYED] '+pair,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in [('selector_replay',checks),('anchor_replay',anchors)]:
        path=OUT/(name+'.csv');pd.DataFrame(rows).to_csv(path,index=False)
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    report=dict(status='guarded_inference_and_anchor_checks_passed',sources=sources,checks=checks,anchor_checks=len(anchors),
                observed_rows=sum(c['rows'] for c in checks),new_fallback_rows=0,selector_fits=0,base_model_scoring_runs=0,
                test_used_for_selection=False,historical_rows_replaced=False,version_warnings=sorted(version_warnings),
                b08_status='closed_for_guarded_inference_entry_point_and_replayed_locks',
                b10_status='anchor_and_shared_endpoint_tests_passed_real_normalized_endpoint_audit_pending',
                boundary='Cached endpoint equality is checked but is not evidence of unoverridden normalized/raw equivalence; server audit required.')
    path=OUT/'audit.json';path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    sources[path.relative_to(ROOT).as_posix()]=sha(path)
    (ROOT/'paper_a_draft/inference_contract_source_manifest.json').write_text(json.dumps(dict(version='inference_contract_review_v1',sources=sources),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','checks','version_warnings')},indent=2))


if __name__=='__main__':
    with threadpool_limits(1):main()
