"""DEV-only check that reused 13D fits are reproduced by the 9D runtime."""
import json
import sys
from pathlib import Path
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_feature_dimension import OUT, PAIRS, INPUT, FIELDS13, sha, signals
from scripts.analyze_paper_a_matched_alternatives import Inputs, apply
from router.matched_actions import configurations, Action
from scripts.analyze_paper_a_conservative_radius import GRID_COLUMNS
from scripts.ablate_anchored_dynamic import fit_geometry_model


def main():
    inputs=Inputs('DEV');rows=[]
    inputs.bind(Path(__file__),False)
    for pair in PAIRS:
        folder=INPUT/pair/'dev_lock'
        frame=inputs.frame(folder/'dev_locked_query_rows.csv')
        lock=inputs.obj(folder/'anchored_dev_lock.json')
        old=inputs.model(folder/lock['model_file'])
        new,x,_=fit_geometry_model(frame.to_dict('records'),fields=FIELDS13,random_state=20260902)
        errors={}
        for name,a,b in [('imputation',old[0].statistics_,new[0].statistics_),
                         ('mean',old[1].mean_,new[1].mean_),('scale',old[1].scale_,new[1].scale_),
                         ('coef',old[2].coef_,new[2].coef_),('intercept',old[2].intercept_,new[2].intercept_),
                         ('decision',old.decision_function(x),new.decision_function(x))]:
            errors[name]=float(np.max(np.abs(a-b)))
        np.testing.assert_allclose(old.decision_function(x),new.decision_function(x),rtol=0,atol=1e-8)
        # Small solver/version differences are measured, not called bitwise equal.
        # Exact equality of every tested DEV action verifies the relevant search.
        old_sig,new_sig=signals(frame,old,FIELDS13),signals(frame,new,FIELDS13)
        grid=frame[GRID_COLUMNS].to_numpy()
        for action in [*configurations('ADC+Global'),Action('Shrink',1.,0.)]:
            before=apply(grid,old_sig,lock['alpha0'],action)
            after=apply(grid,new_sig,lock['alpha0'],action)
            np.testing.assert_array_equal(before[0],after[0])
            np.testing.assert_array_equal(before[1],after[1])
        rows.append(dict(pair=pair,errors=errors,all_42_dev_action_vectors_equal=True))
    report=dict(status='full_13d_runtime_fit_replayed',sklearn=sklearn.__version__,historical_sklearn='1.7.2',
                verification_fits=4,policy_selection=False,test_opened=False,sources=inputs.sources,checks=rows)
    (OUT/'runtime_fit_replay.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
