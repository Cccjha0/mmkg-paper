"""Retrospective diagnostics; these functions never alter a selector or its features."""
import numpy as np
import pandas as pd

TRIPLE=['head_id','relation_id','tail_id']
BINS=('Secondary-favored','Close','Primary-favored','Unsupported')


def training_map(train,band=.005,min_triples=50):
    if band<0 or min_triples<1:raise ValueError('Invalid descriptive thresholds')
    if not np.isfinite(train[['rr_a','rr_b']].to_numpy()).all():raise ValueError('Nonfinite endpoint RR')
    rows=[]
    for (relation,direction),part in train.groupby(['relation_id','direction'],sort=True):
        n=len(part[TRIPLE].drop_duplicates());gap=float((part.rr_a-part.rr_b).mean())
        name='Unsupported' if n<min_triples else ('Secondary-favored' if gap < -band else
            ('Primary-favored' if gap>band else 'Close'))
        rows.append(dict(relation_id=relation,direction=direction,training_triples=n,
            training_observations=len(part),training_gap=gap,stratum=name))
    return pd.DataFrame(rows,columns=['relation_id','direction','training_triples','training_observations','training_gap','stratum'])


def assign_strata(mapping,relation,direction):
    """Held-out API accepts neither gold IDs nor outcomes."""
    if len(relation)!=len(direction):raise ValueError('Unaligned observable keys')
    index=mapping.set_index(['relation_id','direction'])
    if not index.index.is_unique:raise ValueError('Duplicate training cell')
    keys=pd.MultiIndex.from_arrays([relation,direction])
    return index.stratum.reindex(keys).fillna('Unsupported').to_numpy()


def stage_summary(alpha,anchor,raw_alpha,rejected,rr,reference):
    alpha,anchor,raw_alpha,rr,reference=[np.asarray(x,float) for x in (alpha,anchor,raw_alpha,rr,reference)]
    rejected=np.asarray(rejected,bool);n=len(alpha)
    if not all(v.shape==(n,) for v in (anchor,raw_alpha,rejected,rr,reference)):raise ValueError('Unaligned rows')
    if not np.isfinite(np.column_stack([alpha,anchor,raw_alpha,rr,reference])).all():raise ValueError('Nonfinite rows')
    active=alpha!=anchor;raw=raw_alpha!=anchor;d=rr-reference
    if not np.array_equal(active,raw&~rejected):raise ValueError('Full action does not follow gate')
    if not np.array_equal(alpha,np.where(rejected,anchor,raw_alpha)):raise ValueError('Accepted raw action changed')
    if np.any(d[~active]!=0):raise ValueError('Inactive RR changed')
    h=d<0;b=d>0
    counts=dict(n=n,raw_active_n=int(raw.sum()),rejected_n=int(rejected.sum()),
        rejected_active_n=int((rejected&raw).sum()),rejected_idle_n=int((rejected&~raw).sum()),
        accepted_idle_n=int((~rejected&~raw).sum()),active_n=int(active.sum()),
        harm_n=int(h.sum()),benefit_n=int(b.sum()),unchanged_n=int((d==0).sum()))
    if sum(counts[k] for k in ('rejected_active_n','rejected_idle_n','accepted_idle_n','active_n'))!=n:
        raise ValueError('Stage partition failure')
    return dict(**counts,utility=float(d.mean()) if n else np.nan,
        mean_gain=float(np.maximum(d,0).mean()) if n else np.nan,
        mean_loss=float(np.maximum(-d,0).mean()) if n else np.nan,
        conditional_gain=float(d[b].mean()) if b.any() else np.nan,
        conditional_loss=float(-d[h].mean()) if h.any() else np.nan,
        intervention=float(active.mean()) if n else np.nan,
        fallback=float(rejected.mean()) if n else np.nan,
        harm=float(h.mean()) if n else np.nan,benefit=float(b.mean()) if n else np.nan,
        harm_active=float(h.sum()/active.sum()) if active.any() else np.nan)


def conditional_intervals(clusters,groups,values,seed,replicates=10000):
    """Bootstrap ratios over whole six-observation clusters, holding maps fixed."""
    clusters=np.asarray(clusters);groups=np.asarray(groups);values=np.asarray(values,float)
    counts=np.bincount(clusters)
    if not len(counts) or not np.all(counts==6):raise ValueError('Need complete six-row triple clusters')
    if values.ndim!=2 or len(values)!=len(clusters) or groups.shape!=clusters.shape or not np.isfinite(values).all():
        raise ValueError('Invalid conditional observations')
    if not np.isin(groups,BINS).all():raise ValueError('Unknown stratum')
    k=len(counts);width=values.shape[1]
    totals=np.zeros((k,len(BINS),width+1))
    for j,name in enumerate(BINS):
        mask=groups==name;totals[:,j,0]=np.bincount(clusters,weights=mask,minlength=k)
        for col in range(width):totals[:,j,col+1]=np.bincount(clusters,weights=mask*values[:,col],minlength=k)
    rng=np.random.Generator(np.random.PCG64(seed));draws=np.full((replicates,len(BINS),width),np.nan)
    for start in range(0,replicates,32):
        b=min(32,replicates-start);ix=rng.integers(k,size=(b,k))
        weights=np.bincount((ix+np.arange(b)[:,None]*k).ravel(),minlength=b*k).reshape(b,k)
        t=(weights@totals.reshape(k,-1)).reshape(b,len(BINS),width+1)
        draws[start:start+b]=np.divide(t[:,:,1:],t[:,:,:1],out=np.full((b,len(BINS),width),np.nan),where=t[:,:,:1]>0)
    out={}
    for j,name in enumerate(BINS):
        valid=np.isfinite(draws[:,j,0]);out[name]=dict(valid_replicates=int(valid.sum()))
        for col in range(width):
            lo,hi=np.quantile(draws[valid,j,col],[.025,.975],method='linear') if valid.any() else (np.nan,np.nan)
            out[name][col]=(float(lo),float(hi))
    return out
