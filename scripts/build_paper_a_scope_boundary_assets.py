"""E03/E04 formal scope and additional-pair evidence tables."""
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_scope_boundary import OUT,sha
from scripts.build_paper_a_selection_seed_assets import table


def number(v,scale=1,digits=2):return '--' if pd.isna(v) else f'{v*scale:.{digits}f}'
def interval(r,key):return f'{r[key]*1e4:+.2f} [{r[key+"_lo"]*1e4:+.2f},{r[key+"_hi"]*1e4:+.2f}]'


def main():
    p=OUT/'audit.json';a=json.loads(p.read_text());assert a['status']=='scope_boundary_checks_passed'
    sources={**a['sources'],**a['outputs']}
    for rel,h in sources.items():assert sha(ROOT/rel)==h,rel
    sources[p.relative_to(ROOT).as_posix()]=sha(p)
    all_rows=pd.read_csv(OUT/'summary.csv');extra=all_rows[all_rows.label.isin(['W-NA','D-NA'])]
    settings=pd.read_csv(OUT/'original_settings.csv');strata=pd.read_csv(OUT/'dev_strata.csv')
    target=ROOT/'paper_a_draft/tables/scope_boundary';target.mkdir(parents=True,exist_ok=True);tables={}
    def save(name,rows,caption,cols,head):
        p=target/(name+'.tex');p.write_text(table(rows,caption,'tab:scope-'+name,cols,head),encoding='utf-8')
        tables[p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    rows=[]
    for _,r in extra[extra.split=='test'].iterrows():
        s=settings[(settings.label==r.label)&(settings.split=='test')].iloc[0]
        rows.append(f'{r.label} & {s.anchor:.2f}/{s.beta:.2f}/{s.tau:.1f} & {100*r.fallback:.2f} & '+
            f'{int(r.accepted_idle_n):,} & {int(r.active_n):,}/{int(r.n):,} & {interval(r,"utility")}'+r'\\')
    save('additional',rows,
        r'Additional-pair boundary outcomes in the corrected information contract. Final full-DEV settings are $(\alpha_0,\beta,\tau)$. Rejected is the threshold fallback percentage over all observations; accepted no-op counts pass the gate but retain the reference after clipping/projection. Changed counts actual nonzero weights. $U=$ADC--Global uses $10^{-4}$ MRR units and the unified 95\% original-triple percentile interval (10,000 draws, all three seeds and both directions retained). Both intervals span zero; neither pair establishes an incremental gain.',
        'llrrll',r'Pair & $\alpha_0/\beta/\tau$ & Rejected (\%) & Accepted no-op & Changed/$N$ & $U$ [95\% CI]')
    rows=[]
    for _,r in strata.iterrows():
        rows.append(f'{r.label} & {r.stratum} & {int(r.n):,} & {100*r.mass:.2f} & '+
            interval(r,'endpoint_gap')+' & '+interval(r,'utility')+f' & {100*r.intervention:.2f}'+r'\\')
    save('dev',rows,
        r'Local strength strata in original held-out DEV. A relation--direction cell is assigned using only the other four folds: at least 50 distinct training triples and pooled endpoint MRR gap below $-0.005$, within $[-0.005,0.005]$, or above $0.005$; unsupported cells are retained. These names describe the training estimate, not the held-out gap. Gap and $U$ use $10^{-4}$ MRR units with conditional-ratio 95\% original-triple percentile intervals; $I$ is actual intervention. Maps/fits are fixed over 10,000 resamples, denominators are recomputed, and all six correlated observations remain together. Intervals are exploratory and unadjusted. A global necessity test cannot be inferred from these local cells.',
        'llrrllr',r'Pair & Train stratum & $n$ & Mass (\%) & Gap [95\% CI] & $U$ [95\% CI] & $I$ (\%)')
    rows=[]
    for _,r in extra.iterrows():
        split='DEV-OOF' if r.split=='dev_oof' else 'TEST'
        rows.append(f'{r.label} & {split} & '+ ' & '.join(f'{int(r[k]):,}' for k in
            ('n','rejected_active_n','rejected_idle_n','accepted_idle_n','active_n'))+r'\\')
    save('stages',rows,
        r'Disjoint additional-pair action ledger. Raw proposals use the original radius and projection with $\tau=0$. Rejected raw-active, rejected raw-idle, accepted raw-idle and changed counts sum to $N$ in each row. The first two sum to threshold fallback; the last two sum to gate acceptance. No invalid feature rows occur in these saved observations. The accepted no-op stage explains why $1-$fallback need not equal actual intervention.',
        'llrrrrr',r'Pair & Split & $N$ & Reject active & Reject idle & Accept idle & Changed')
    rows=[]
    for _,r in extra.iterrows():
        split='DEV-OOF' if r.split=='dev_oof' else 'TEST'
        rows.append(f'{r.label} & {split} & {100*r.b_win_rate:.2f} & {100*r.tie_rate:.2f} & '+
            f'{1e4*r.oracle_gap:.2f} & {1e4*r.grid_oracle_gap:.2f} & {int(r.winner_n):,} & {r.winner_auc:.3f}'+r'\\')
    save('opportunity',rows,
        r'Opportunity and endpoint preference are separate diagnostics. B-win and ties use all observations. $O_{AB}$ is endpoint-oracle minus best-single MRR; $O_{grid}$ is the 21-weight answer-aware grid oracle minus the fold/final Global reference; both use $10^{-4}$ units and require gold RR only after evaluation. Neither is an achievable performance claim. AUROC ranks saved preference margins against A-winning endpoint labels, excludes ties ($n_{win}$), is descriptive without a new interval, and is neither natural-probability calibration nor harm prediction. No diagnostic is added to selector inputs.',
        'llrrrrrr',r'Pair & Split & B-win (\%) & Ties (\%) & $O_{AB}$ & $O_{grid}$ & $n_{win}$ & AUROC')
    rows=[]
    for _,r in extra.iterrows():
        split='DEV-OOF' if r.split=='dev_oof' else 'TEST'
        rows.append(f'{r.label} & {split} & '+ ' & '.join(f'{int(r[k]):,}' for k in ('benefit_n','unchanged_n','harm_n'))+
            ' & '+ ' & '.join(number(r[k],1e4) for k in ('mean_gain','mean_loss'))+' & '+interval(r,'utility')+r'\\')
    save('outcomes',rows,
        r'Additional-pair outcome decomposition. Benefit/unchanged/harm refer to the sign of RR change, not to weight movement. All observations remain in the denominator of unconditional gains $BG$ and losses $HL$; those columns and net $U$ use $10^{-4}$ units. The full-precision identity $BG-HL=U$ is checked before rounding. Intervals condition on the original fitted policies and retain all six seed/direction observations per triple. Full conditional gain/loss, intervention-conditional harm and seed/direction results are bound in the supplement and Appendix~\ref{app:outcome-risk}.',
        'llrrrrrl',r'Pair & Split & Benefit & RR same & Harm & $BG$ & $HL$ & $U$ [95\% CI]')
    rows=[]
    for _,r in extra.iterrows():
        split='DEV-OOF' if r.split=='dev_oof' else 'TEST'
        gate=dict(utility=r.gate_delta,utility_lo=r.gate_lo,utility_hi=r.gate_hi)
        matched=dict(utility=r.matched_gate_delta,utility_lo=r.matched_gate_lo,utility_hi=r.matched_gate_hi)
        rows.append(f'{r.label} & {split} & {r.raw_utility*1e4:+.2f} & '+interval(gate,'utility')+' & '+interval(matched,'utility')+r'\\')
    save('gate',rows,
        r'Fixed-map gate contrasts, all RR effects in $10^{-4}$ units with unified 95\% original-triple percentile intervals. Raw $U$ uses the same original radius with $\tau=0$. Native full-minus-no-gate measures rejection of those fixed proposals. Matched-I uniformly thins the no-gate nonzero proposals to the full gate\textquotesingle s count in each seed/fold/repetition stratum, using exact discrete-action expectations (Appendix~\ref{app:component-ablation}). This matches intervention quantity; direction/amplitude need not match. Positive native gate effects do not establish an optimal gate, useful selection beyond count reduction, or a positive full-minus-Global effect. Both TEST Matched-I intervals span zero; D-NA has a positive DEV Matched-I interval.',
        'lll ll',r'Pair & Split & Raw $U$ & Full--raw [95\% CI] & Matched-I [95\% CI]')
    for rel in ('scripts/build_paper_a_scope_boundary_assets.py','scripts/build_paper_a_selection_seed_assets.py',
                'docs/reports/paper_a_scope_boundary_review_2026-09-13.md'):
        if (ROOT/rel).exists():sources[rel]=sha(ROOT/rel)
    (ROOT/'paper_a_draft/scope_boundary_source_manifest.json').write_text(json.dumps(dict(
        version='scope_boundary_v1',sources=sources,tables=tables),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=len(tables),sources=len(sources))))


if __name__=='__main__':main()
