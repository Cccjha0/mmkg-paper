"""Fixed-setting component contrasts and outcome-free intervention matching."""
import numpy as np
from router.matched_actions import ALPHAS, Action, grid_indices, weights

METHODS=('ADC','Center-0.5','Linear-p','Clip-g','Shrink-fixed','Radius-1','No-gate','Global')
COMPARATORS=METHODS[1:-1]


def component_weights(g,p,invalid,anchor,beta,tau,method):
    if method not in METHODS or not 0<beta<=.5:
        raise ValueError('Unknown method or radius outside original family')
    if method=='Global':return weights(g,p,invalid,anchor,Action('ADC+Global',0,0))[0]
    if method!='Center-0.5':
        family={'ADC':'ADC+Global','Linear-p':'Linear-p','Clip-g':'Clip-g',
                'Shrink-fixed':'Shrink-gate','Radius-1':'ADC+Global','No-gate':'ADC+Global'}[method]
        return weights(g,p,invalid,anchor,Action(family,1. if method=='Radius-1' else beta,
                                               0. if method=='No-gate' else tau))[0]
    g,p,invalid=np.asarray(g),np.asarray(p),np.asarray(invalid,dtype=bool)
    if g.ndim!=1 or g.shape!=p.shape or p.shape!=invalid.shape:
        raise ValueError('Unaligned preference arrays')
    # Preserve the same fallback target and projection ties; only proposal origin changes.
    invalid=invalid|~np.isfinite(g)|~np.isfinite(p)
    g,p=np.where(invalid,0,g),np.where(invalid,.5,p)
    fallback=invalid|(np.abs(2*p-1)<tau)
    z=np.where(fallback,anchor,np.clip(.5+beta*np.tanh(g),0,1))
    return ALPHAS[grid_indices(z,anchor)]


def match_interventions(strata,repetitions,left_active,right_active):
    """Uniformly thin each side to min(active units) per observable stratum.

    Return marginal keep probabilities, quotas and one realized pair of masks.
    Random expectation is evaluated on each discrete action, never at a mean weight.
    """
    strata,repetitions=np.asarray(strata),np.asarray(repetitions)
    left,right=np.asarray(left_active,dtype=bool),np.asarray(right_active,dtype=bool)
    if not (strata.ndim==1 and strata.shape==repetitions.shape==left.shape==right.shape):
        raise ValueError('Unaligned matching vectors')
    if np.any(repetitions<=0) or not np.all(repetitions==np.rint(repetitions)):
        raise ValueError('Positive integer repetition counts required')
    order=np.argsort(strata,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(strata[order]))+1]
    groups=np.split(order,starts[1:]) if len(order) else []
    a,b=np.zeros(len(left)),np.zeros(len(right));qa,qb=np.zeros(len(left),bool),np.zeros(len(right),bool)
    # A single fixed realization is an integer-quota check, never a reported result.
    rng=np.random.default_rng(2026091231);ledger=[]
    for idx in groups:
        if len(np.unique(repetitions[idx]))!=1:raise ValueError('Mixed repetitions within a stratum')
        li,ri=idx[left[idx]],idx[right[idx]];k=min(len(li),len(ri));rep=int(repetitions[idx[0]])
        if k:
            a[li]=k/len(li);b[ri]=k/len(ri)
            qa[rng.choice(li,k,replace=False)]=True;qb[rng.choice(ri,k,replace=False)]=True
        ledger.append(dict(stratum=int(strata[idx[0]]),repetitions=rep,left_units=len(li),right_units=len(ri),
                           retained_units=k,retained_observations=k*rep))
    target=sum(r['retained_observations'] for r in ledger)
    if int(repetitions@qa)!=target or int(repetitions@qb)!=target:raise ValueError('Realized quota mismatch')
    if not np.isclose(repetitions@a,target,atol=1e-9,rtol=0) or not np.isclose(repetitions@b,target,atol=1e-9,rtol=0):
        raise ValueError('Expected quota mismatch')
    return a,b,ledger


def outcome_summary(rr,reference,alpha,anchor,keep=None):
    rr,reference,alpha,anchor=map(np.asarray,(rr,reference,alpha,anchor))
    if not (rr.ndim==1 and rr.shape==reference.shape==alpha.shape==anchor.shape) or not len(rr):
        raise ValueError('Unaligned outcomes')
    prob=np.ones(len(rr)) if keep is None else np.asarray(keep)
    if prob.shape!=rr.shape or np.any((prob<0)|(prob>1)):raise ValueError('Invalid keep probabilities')
    d=rr-reference;active=alpha!=anchor;harm=d<0;benefit=d>0
    if np.any((d!=0)&~active):raise ValueError('Inactive weight changes RR')
    n=len(rr);ni=float(prob@active);nh=float(prob@harm)
    return dict(n=n,active_n=ni,harm_n=nh,benefit_n=float(prob@benefit),
        mrr=float(reference.mean()+prob@d/n),utility=float(prob@d/n),harm=nh/n,
        harm_active=nh/ni if ni else np.nan,intervention=ni/n,
        mean_loss=float(prob@np.maximum(-d,0)/n),mean_gain=float(prob@np.maximum(d,0)/n),
        movement=float(prob@np.abs(alpha-anchor)/n))
