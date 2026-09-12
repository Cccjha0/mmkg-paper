"""Support-defined partitions and descriptive weighted outcome aggregation."""
import numpy as np
import pandas as pd

FREQUENCY=('Train-zero','1-99','100-999','1000+')
TOP=('R1','R2','R3','R4','R5','Other')
METRICS=('global_mrr','adc_mrr','delta_mrr','mean_gain','mean_loss','harm_rate','gain_rate','action_rate','fallback_rate')


def relation_map(training_triples,relation_names):
    triples=np.asarray(training_triples,dtype=np.int64)
    if triples.ndim!=2 or triples.shape[1]!=3:raise ValueError('Expected head/relation/tail triples')
    if len(np.unique(triples,axis=0))!=len(triples):raise ValueError('Duplicate training triple')
    if np.any((triples[:,1]<0)|(triples[:,1]>=len(relation_names))):raise ValueError('Unknown relation ID')
    counts=np.bincount(triples[:,1],minlength=len(relation_names))
    order=sorted(range(len(counts)),key=lambda r:(-counts[r],r))
    selected={r:f'R{i+1}' for i,r in enumerate(order[:5])}
    rows=[]
    for r,n in enumerate(counts):
        frequency='Train-zero' if n==0 else ('1-99' if n<100 else ('100-999' if n<1000 else '1000+'))
        rows.append(dict(relation_id=r,relation_iri=relation_names[r],train_triples=int(n),frequency=frequency,
            named_group=selected.get(r,'Other'),train_rank=order.index(r)+1))
    return pd.DataFrame(rows)


def aggregate(part,total_n,observations_per_triple):
    """Input rows must be disjoint relations, or disjoint relation/direction cells."""
    if total_n<=0 or observations_per_triple not in (3,6):raise ValueError('Invalid population definition')
    n=int(part.n_observations.sum());t=int(part.n_triples.sum())
    if n!=observations_per_triple*t or n>total_n:raise ValueError('Support mismatch')
    if len(part) and (not np.isfinite(part[list(METRICS)].to_numpy()).all() or (part.n_observations<=0).any()):
        raise ValueError('Invalid component statistics')
    r={k:float(np.sum(part.n_observations*part[k])/n) if n else np.nan for k in METRICS}
    if n and (abs(r['adc_mrr']-r['global_mrr']-r['delta_mrr'])>1e-12 or
            abs(r['mean_gain']-r['mean_loss']-r['delta_mrr'])>1e-12):raise ValueError('Outcome identity failure')
    return dict(n_observations=n,n_triples=t,n_relations=int(part.relation_id.nunique()) if 'relation_id' in part else 0,
        mass=n/total_n,delta_contribution=n/total_n*r['delta_mrr'] if n else 0.,**r)


def top_concentration(top,all_rows):
    top_n=top.n_observations.sum();n=all_rows.n_observations.sum()
    out=dict(observation_share=float(top_n/n))
    for name in ('mean_gain','mean_loss'):
        numerator=float(np.sum(top.n_observations*top[name]));denominator=float(np.sum(all_rows.n_observations*all_rows[name]))
        out[name+'_share']=numerator/denominator if denominator>0 else np.nan
    out['top_contribution']=float(np.sum(top.n_observations*top.delta_mrr)/n)
    out['other_contribution']=float(np.sum(all_rows.n_observations*all_rows.delta_mrr)/n)-out['top_contribution']
    return out
