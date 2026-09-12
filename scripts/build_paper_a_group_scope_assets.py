"""E05 source-bound absolute direction and train-defined relation tables."""
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_group_scope import OUT,sha
from scripts.build_paper_a_selection_seed_assets import table


def metric(x):return '--' if pd.isna(x) else f'{x:.4f}'
def effect(x):
    if pd.isna(x):return '--'
    if x==0:return '0'
    if abs(x*1e4)<.005:return r'$0^{'+('+' if x>0 else '-')+'}$'
    return f'{x*1e4:+.2f}'


def main():
    p=OUT/'audit.json';audit=json.loads(p.read_text());assert audit['status']=='group_scope_checks_passed'
    sources={**audit['sources'],**audit['outputs']}
    for rel,h in sources.items():assert sha(ROOT/rel)==h,rel
    sources[p.relative_to(ROOT).as_posix()]=sha(p)
    d=pd.read_csv(OUT/'directions.csv');g=pd.read_csv(OUT/'groups.csv');m=pd.read_csv(OUT/'relation_map.csv')
    c=pd.read_csv(OUT/'concentration.csv')
    target=ROOT/'paper_a_draft/tables/group_scope';target.mkdir(parents=True,exist_ok=True);tables={}
    def save(name,rows,caption,cols,head):
        p=target/(name+'.tex');p.write_text(table(rows,caption,'tab:group-'+name,cols,head),encoding='utf-8')
        tables[p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    for split in ('test','dev_oof'):
        rows=[]
        for _,r in d[d.split==split].iterrows():
            rows.append(f'{r.label} & {r.direction} & {int(r.n_triples):,} & {int(r.n_observations):,} & '+
                metric(r.global_mrr)+' & '+metric(r.adc_mrr)+' & '+effect(r.delta_mrr)+r'\\')
        save('direction_'+split,rows,
            ('Locked TEST' if split=='test' else 'Original grouped held-out DEV')+
            r' absolute direction metrics on all six fixed pairs. $T$ counts original triples within the direction, and $N=3T$ counts paired-seed observations. These are not counts of independent query keys. Global and ADC MRR pool all three seeds with equal row weight; $U=$ADC--Global uses $10^{-4}$ MRR units, computed before display rounding. Negative directions are retained. This is a descriptive breakdown; no new directional significance claim is made. DEV uses each fold\textquotesingle s fitted policy and reference, whereas TEST uses the final DEV-locked objects.',
            'llrrrrr',r'Pair & Direction & $T$ & $N$ & Global MRR & ADC MRR & $U$')
    rows=[]
    for ds in ('mkg_w','db15k'):
        for _,r in m[(m.dataset==ds)&(m.named_group!='Other')].sort_values('train_rank').iterrows():
            name=r.relation_iri.strip('<>').rsplit('/',1)[-1].rsplit('#',1)[-1].replace('_',r'\_')
            rows.append(('MKG-W' if ds=='mkg_w' else 'DB15K')+f' & {r.named_group} & {int(r.relation_id)} & '+
                r'\texttt{'+name+'} & '+f'{int(r.train_triples):,}'+r'\\')
    save('mapping',rows,
        r'Named relation groups fixed using effective base-model TRAIN support only. R1--R5 are the five largest counts within each dataset, with ascending canonical relation ID breaking count ties. Other contains every remaining relation; the same map applies to all three pairs and both splits. Identifier is the terminal component of the canonical relation IRI, not an inferred semantic label; full IRIs and all 448 relation IDs are supplied in the mapping CSV. MKG-W uses 34,196 effective TRAIN triples; DB15K uses 71,300 after removing its 7,922 DEV holdout. No TEST metric determines membership or display order.',
        'llrlr',r'Dataset & Group & Relation ID & Identifier & TRAIN triples')
    caption=(r'Absolute Global/ADC MRR and signed contributions for a fixed relation partition. Both directions and all three seeds are pooled: $T_g$ is group original-triple support and $N_g=6T_g$ is group observation support; $N$ is the whole-pair count. $U_g$ is the conditional ADC--Global difference; $C_g=(N_g/N)U_g$ is its contribution to the whole pair. Both use $10^{-4}$ units, and $\sum_g C_g=U$ is checked at full precision. Empty groups have undefined MRR and zero contribution. $0^{+}$/$0^{-}$ marks a nonzero value below display precision, while 0 is exact. Values are descriptive; no group significance or causal claim is made. ')
    def group_rows(frame):
        return [f'{r.label} & {r["group"]} & {int(r.n_triples):,} & {int(r.n_observations):,} & '+
            metric(r.global_mrr)+' & '+metric(r.adc_mrr)+' & '+effect(r.delta_mrr)+' & '+effect(r.delta_contribution)+r'\\'
            for _,r in frame.iterrows()]
    for split in ('test','dev_oof'):
        save('frequency_'+split,group_rows(g[(g.split==split)&(g.partition=='frequency')]),
            ('Locked TEST. ' if split=='test' else 'Original grouped held-out DEV. ')+caption+
            r'The four groups use 0, 1--99, 100--999 and at least 1,000 effective TRAIN triples per relation. These are support bins, not semantic families. All bins and effect signs are retained; no outcome sorting is used.',
            'llrrrrrr',r'Pair & TRAIN bin & $T_g$ & $N_g$ & Global & ADC & $U_g$ & $C_g$')
    for ds in ('mkg_w','db15k'):
        save('named_'+ds,group_rows(g[(g.split=='test')&(g.partition=='named_group')&(g.dataset==ds)]),
            'Locked TEST on '+('MKG-W. ' if ds=='mkg_w' else 'DB15K. ')+caption+
            r'R1--R5 follow Table~\ref{tab:group-mapping}, including every positive, zero and negative result; Other completes the partition. DEV counterparts, every relation and all relation--direction cells remain in the bound CSV supplement.',
            'llrrrrrr',r'Pair & Relation group & $T_g$ & $N_g$ & Global & ADC & $U_g$ & $C_g$')
    rows=[]
    for _,r in c.iterrows():
        split='DEV-OOF' if r.split=='dev_oof' else 'TEST'
        rows.append(f'{r.label} & {split} & {100*r.observation_share:.2f} & {100*r.mean_gain_share:.2f} & '+
            f'{100*r.mean_loss_share:.2f} & '+effect(r.top_contribution)+' & '+effect(r.other_contribution)+r'\\')
    save('concentration',rows,
        r'Contribution of the five TRAIN-most-supported relations. N-share uses the whole pair\textquotesingle s observation count; gain/loss shares use sums of positive RR gains and RR losses separately, rather than a ratio to near-zero net utility. $C_5$ and $C_{Other}$ use $10^{-4}$ MRR units and sum to the pair\textquotesingle s net increment. This measures concentration in prechosen frequent relations, not the greatest achievable concentration after selecting TEST winners. Each dataset has shared supports across its three pairs; those repeated workloads are not independent datasets.',
        'llrrrrr',r'Pair & Split & N-share (\%) & Gain share (\%) & Loss share (\%) & $C_5$ & $C_{Other}$')
    for rel in ('scripts/build_paper_a_group_scope_assets.py','scripts/build_paper_a_selection_seed_assets.py',
        'docs/reports/paper_a_group_scope_review_2026-09-13.md'):
        if (ROOT/rel).exists():sources[rel]=sha(ROOT/rel)
    (ROOT/'paper_a_draft/group_scope_source_manifest.json').write_text(json.dumps(dict(
        version='group_scope_v1',sources=sources,tables=tables),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=len(tables),sources=len(sources))))


if __name__=='__main__':main()
