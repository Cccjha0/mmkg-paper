import unittest
import numpy as np
import pandas as pd
from router.scope_boundary_diagnostics import training_map,assign_strata,stage_summary,conditional_intervals


class ScopeBoundaryTests(unittest.TestCase):
    def test_training_boundaries_and_distinct_support(self):
        rows=[]
        for relation,gap in enumerate((-.006,-.005,0,.005,.006)):
            for seed in (1,2,3):rows.append(dict(head_id=relation,relation_id=relation,tail_id=20,seed=seed,direction='tail',rr_a=gap,rr_b=0.))
        frame=pd.DataFrame(rows)
        self.assertEqual(training_map(frame,min_triples=1).stratum.tolist(),['Secondary-favored','Close','Close','Close','Primary-favored'])
        self.assertEqual(training_map(frame,min_triples=2).stratum.tolist(),['Unsupported']*5)

    def test_assignment_has_no_held_out_gold_or_outcomes(self):
        train=pd.DataFrame([dict(head_id=0,relation_id=1,tail_id=2,direction='head',rr_a=.1,rr_b=.2)])
        m=training_map(train,min_triples=1)
        held=pd.DataFrame(dict(relation_id=[1,1,9],direction=['head','tail','head'],gold=[3,4,5],rr_a=[1,0,0]))
        expected=assign_strata(m,held.relation_id,held.direction)
        held['gold']=-999;held['rr_a']=1-held.rr_a
        np.testing.assert_array_equal(expected,assign_strata(m,held.relation_id,held.direction))
        self.assertEqual(expected.tolist(),['Secondary-favored','Unsupported','Unsupported'])

    def test_four_stages_include_gate_noop_overlap(self):
        s=stage_summary([1,1,1,.5],[1]*4,[.5,1,1,.5],[1,1,0,0],[1,1,1,.5],[1]*4)
        self.assertEqual([s[k] for k in ('rejected_active_n','rejected_idle_n','accepted_idle_n','active_n')],[1]*4)
        self.assertEqual(s['fallback'],.5);self.assertEqual(s['intervention'],.25)
        self.assertEqual(s['utility'],s['mean_gain']-s['mean_loss'])
        with self.assertRaises(ValueError):stage_summary([1],[1],[.5],[False],[1],[1])

    def test_conditional_ratio_uses_resampled_denominator(self):
        clusters=np.repeat(np.arange(2),6);groups=np.array(['Close']+['Primary-favored']*5+['Close']*6)
        values=np.r_[np.ones(6),np.repeat(3.,6)][:,None]
        r=conditional_intervals(clusters,groups,values,9,300)
        self.assertEqual(r['Close'][0],(1.,3.));self.assertEqual(r['Primary-favored'][0],(1.,1.))
        self.assertEqual(r['Unsupported']['valid_replicates'],0)
        self.assertTrue(np.isnan(r['Unsupported'][0][0]))
        with self.assertRaises(ValueError):conditional_intervals(clusters[:-1],groups[:-1],values[:-1],9,20)


if __name__=='__main__':unittest.main()
