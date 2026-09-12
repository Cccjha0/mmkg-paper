import unittest
import numpy as np
import pandas as pd
from router.group_scope_diagnostics import relation_map,aggregate,top_concentration,METRICS


def row(n,g,a,relation):
    d=a-g
    return dict(n_observations=n,n_triples=n//6,relation_id=relation,global_mrr=g,adc_mrr=a,delta_mrr=d,
        mean_gain=max(d,0),mean_loss=max(-d,0),harm_rate=float(d<0),gain_rate=float(d>0),action_rate=float(d!=0),fallback_rate=0.)


class GroupScopeTests(unittest.TestCase):
    def test_train_frequency_endpoints_and_tie_order(self):
        triples=[(j,r,10000+j) for r,n in enumerate((0,1,99,100,999,1000,1000)) for j in range(n)]
        m=relation_map(triples,list('abcdefg'))
        self.assertEqual(m.frequency.tolist(),['Train-zero','1-99','1-99','100-999','100-999','1000+','1000+'])
        self.assertEqual(m.loc[5,'named_group'],'R1');self.assertEqual(m.loc[6,'named_group'],'R2')
        np.testing.assert_array_equal(m.train_triples,[0,1,99,100,999,1000,1000])
        with self.assertRaises(ValueError):relation_map(triples+[triples[0]],list('abcdefg'))

    def test_weighted_absolute_metrics_and_negative_contribution(self):
        f=pd.DataFrame([row(6,.4,.5,0),row(18,.5,.45,1)])
        r=aggregate(f,24,6)
        self.assertAlmostEqual(r['global_mrr'],.475);self.assertAlmostEqual(r['adc_mrr'],.4625)
        self.assertLess(r['delta_mrr'],0)
        parts=[aggregate(f.iloc[[i]],24,6) for i in range(2)]
        self.assertLess(parts[1]['delta_contribution'],0)
        self.assertAlmostEqual(sum(p['delta_contribution'] for p in parts),r['delta_mrr'])
        c=top_concentration(f.iloc[:1],f)
        self.assertEqual(c['mean_gain_share'],1.);self.assertEqual(c['mean_loss_share'],0.)

    def test_empty_groups_and_support_semantics(self):
        f=pd.DataFrame([row(6,.4,.4,0)])
        r=aggregate(f.iloc[:0],6,6)
        self.assertEqual(r['delta_contribution'],0.);self.assertTrue(np.isnan(r['global_mrr']))
        self.assertTrue(np.isnan(top_concentration(f,f)['mean_gain_share']))
        with self.assertRaises(ValueError):aggregate(f,6,3)


if __name__=='__main__':unittest.main()
