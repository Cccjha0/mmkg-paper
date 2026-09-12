"""Offline outcome populations, additive gain/loss accounting and cluster ratios."""
import numpy as np

FIELDS=('n','active_n','harm_n','benefit_n','unchanged_n','loss_sum','gain_sum','delta_sum',
    'severe_10_n','severe_50_n','top1_n','top1_retained_n','top1_lost_n','top1_outside10_n',
    'active_top1_n','active_top1_retained_n','active_top1_lost_n','active_top1_outside10_n','new_top1_n','hits1_delta_sum')
RATIOS={
    'intervention':('active_n','n'),'harm':('harm_n','n'),'harm_active':('harm_n','active_n'),
    'benefit':('benefit_n','n'),'unchanged':('unchanged_n','n'),
    'conditional_loss':('loss_sum','harm_n'),'conditional_gain':('gain_sum','benefit_n'),
    'mean_loss':('loss_sum','n'),'mean_gain':('gain_sum','n'),'utility':('delta_sum','n'),
    'severe_10':('severe_10_n','n'),'severe_10_active':('severe_10_n','active_n'),
    'severe_50':('severe_50_n','n'),'severe_50_active':('severe_50_n','active_n'),
    'top1_retention':('top1_retained_n','top1_n'),'top1_loss':('top1_lost_n','top1_n'),
    'top1_outside10':('top1_outside10_n','top1_n'),
    'top1_active_retention':('active_top1_retained_n','active_top1_n'),
    'top1_active_loss':('active_top1_lost_n','active_top1_n'),
    'top1_active_outside10':('active_top1_outside10_n','active_top1_n'),
    'new_top1':('new_top1_n','n'),'hits1_delta':('hits1_delta_sum','n')}


def integer_ranks(rr):
    rr=np.asarray(rr,float)
    if rr.ndim!=1 or not np.isfinite(rr).all() or np.any((rr<=0)|(rr>1)):
        raise ValueError('Finite positive reciprocal ranks required')
    ranks=np.rint(1/rr).astype(np.int64)
    if not np.allclose(rr,1/ranks,atol=1e-12,rtol=0):raise ValueError('Not integer filtered ranks')
    return ranks


def observations(rr,reference,alpha,anchor):
    rr,reference,alpha,anchor=[np.asarray(v,float) for v in (rr,reference,alpha,anchor)]
    if not all(v.ndim==1 and v.shape==rr.shape and np.isfinite(v).all() for v in (reference,alpha,anchor)):
        raise ValueError('Unaligned finite outcome/action vectors required')
    rank,base=integer_ranks(rr),integer_ranks(reference)
    if np.any(rank>10**8) or np.any(base>10**8):raise ValueError('Ranks exceed safe exact event arithmetic')
    delta=rr-reference;harm=delta<0;benefit=delta>0;active=alpha!=anchor
    if not np.array_equal(active,abs(alpha-anchor)>1e-12):raise ValueError('Ambiguous weight tolerance')
    if not np.array_equal(harm,delta < -1e-12) or not np.array_equal(benefit,delta>1e-12):
        raise ValueError('Exact outcome signs differ from historical tolerance')
    if np.any((harm|benefit)&~active):raise ValueError('Inactive rows have changed RR')
    top1=base==1;retained=top1&(rank==1);lost=top1&(rank>1);outside=top1&(rank>10)
    severe10=harm & (10*(rank-base)>=rank*base)
    severe50=harm & (2*(rank-base)>=rank*base)
    x=np.column_stack([np.ones(len(rr)),active,harm,benefit,delta==0,np.maximum(-delta,0),np.maximum(delta,0),delta,
        severe10,severe50,top1,retained,lost,outside,active&top1,active&retained,active&lost,active&outside,
        (base>1)&(rank==1),(rank==1).astype(int)-top1.astype(int)])
    return x,rank,base


def ratios(totals):
    totals=np.asarray(totals,float)
    out={}
    for name,(num,den) in RATIOS.items():
        a,b=totals[...,FIELDS.index(num)],totals[...,FIELDS.index(den)]
        out[name]=np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
    return out


def summarize(x):
    t=x.sum(axis=0);r={name:float(v) for name,v in ratios(t).items()}
    r.update({name:int(v) if name=='n' or name.endswith('_n') else float(v) for name,v in zip(FIELDS,t)})
    h=x[:,FIELDS.index('harm_n')].astype(bool);a=x[:,FIELDS.index('active_n')].astype(bool)
    losses=x[h,FIELDS.index('loss_sum')]
    r['active_unchanged_n']=int((a & (x[:,FIELDS.index('unchanged_n')]>0)).sum())
    r['inactive_n']=int((~a).sum())
    if len(losses):
        for q,v in zip(('q50','q90','q95','q99'),np.quantile(losses,[.5,.9,.95,.99],method='linear')):r['loss_'+q]=float(v)
        r['loss_max']=float(losses.max());k=int(np.ceil(.1*len(losses)))
        r['worst_tenth_harm_n']=k;r['worst_tenth_loss_share']=float(np.sort(losses)[-k:].sum()/losses.sum())
    else:
        r.update({k:np.nan for k in ('loss_q50','loss_q90','loss_q95','loss_q99','loss_max','worst_tenth_loss_share')})
        r['worst_tenth_harm_n']=0
    return r


def cluster_ratio_intervals(x,clusters,seed,replicates=10000):
    x=np.asarray(x,float);clusters=np.asarray(clusters)
    if x.ndim!=2 or x.shape[1]!=len(FIELDS) or len(x)!=len(clusters) or not np.isfinite(x).all():
        raise ValueError('Invalid additive observation matrix')
    if not np.issubdtype(clusters.dtype,np.integer) or np.any(clusters<0):raise ValueError('Nonnegative integer clusters required')
    counts=np.bincount(clusters)
    if not len(counts) or not np.all(counts==6):raise ValueError('Every original triple must retain six observations')
    if replicates<2:raise ValueError('Need at least two bootstrap draws')
    k=len(counts);sums=np.column_stack([np.bincount(clusters,weights=x[:,j]) for j in range(x.shape[1])])
    rng=np.random.Generator(np.random.PCG64(seed));samples={n:np.empty(replicates) for n in RATIOS}
    for start in range(0,replicates,32):
        b=min(32,replicates-start);ix=rng.integers(k,size=(b,k))
        w=np.bincount((ix+np.arange(b)[:,None]*k).ravel(),minlength=b*k).reshape(b,k)
        values=ratios(w@sums)
        for name,v in values.items():samples[name][start:start+b]=v
    result={}
    for name,v in samples.items():
        valid=np.isfinite(v);lo,hi=np.quantile(v[valid],[.025,.975],method='linear') if valid.any() else (np.nan,np.nan)
        result[name+'_lo']=float(lo);result[name+'_hi']=float(hi);result[name+'_valid_replicates']=int(valid.sum())
    return result
