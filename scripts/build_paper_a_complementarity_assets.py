"""Build source-bound nearest-method and complementarity tables/figure for A01/A04/E01."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_complementarity import OUT, PAIRS, SUPPORT, RANK_BINS, sha

PAPER = ROOT / "paper_a_draft"
TABLES = PAPER / "tables/complementarity"
FIGURES = PAPER / "figures/complementarity"
REFERENCE = ROOT / "docs/protocols/paper_a_nearest_methods_reference.json"
REPORT = ROOT / "docs/reports/paper_a_complementarity_review_2026-09-12.md"


def main():
    audit = json.loads((OUT/"audit.json").read_text())
    assert audit["status"] == "complementarity_checks_passed" and not audit["failures"]
    assert not audit["new_policy_selection"] and not audit["selector_features_changed"]
    for rel,digest in audit["sources"].items():
        assert sha(ROOT/rel)==digest,rel
    for rel,digest in audit["outputs"].items():
        assert sha(OUT/rel)==digest,rel
    TABLES.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    tables,figures = {},{}

    def table(name,caption,label,columns,header,rows,tabularx=False):
        environment = "tabularx" if tabularx else "tabular"
        text = (r"\begin{table}[htbp]"+"\n"+r"\centering\footnotesize\setlength{\tabcolsep}{3pt}"+"\n"+
                r"\caption{"+caption+"}\n"+r"\label{"+label+"}\n"+r"\begin{"+environment+"}"+
                (r"{\linewidth}" if tabularx else "")+"{"+columns+"}\n"+r"\toprule"+"\n"+
                " & ".join(header)+r"\\"+"\n"+r"\midrule"+"\n"+
                "\n".join(" & ".join(row)+r"\\[2pt]" if row else r"\midrule" for row in rows)+"\n"+
                r"\bottomrule"+"\n"+r"\end{"+environment+"}\n"+r"\end{table}"+"\n")
        path=TABLES/(name+".tex")
        path.write_text(text,encoding="utf-8",newline="\n")
        tables[path.relative_to(PAPER).as_posix()]=sha(path)

    ragged = r">{\raggedright\arraybackslash}X"
    rows=[
        [r"DynaSemble\newline\citep{dynasemble}", "DEV gold-versus-negative margin loss",
         "One textual coefficient fixed to 1; no DEV-Global-centered radius",
         r"Query-dependent nonnegative coefficient in released core; no $\beta$ cap",
         "No explicit confidence gate returning a static mixture; zero is a coefficient endpoint",
         "Textual/structural complementarity; our transfer controls in Appendix~\\ref{app:dynasemble}"],
        [r"MoSE~\citep{mose}", "Modality CE training; DEV RankBoost (BI) or CE meta-learning (MI)",
         "No static-mixture-centered correction in the defined inference rules",
         "Equal scores (AI), relation weights (BI), candidate-score meta-learner (MI)",
         "No explicit return-to-static rejection gate in these inference rules",
         "Modality-split representations and prediction; literature comparison here"],
        ["Query-soft",r"Shared DEV winner-label logistic fit, Eq.~\eqref{eq:label}",
         r"$\alpha_0$ only for non-finite fallback",r"$\alpha=p_A(q)$; full interval before grid projection",
         "Non-finite fallback; no low-confidence adjustment rejection",
         r"Same fitted object; Table~\ref{tab:matched-test}"],
        [r"Shrink-gate / Clip-g", "Same winner fit and 41 DEV choices per family",
         r"DEV $\alpha_0$",r"$(1-\lambda)\alpha_0+\lambda p_A$ / clipped-logit offset",
         r"Same confidence gate as ADC; zero strength returns Global",
         r"Matched action alternatives; Appendix~\ref{app:matched}"],
        ["ADC", "Winner-label logistic fit; DEV MRR selects action settings",
         r"DEV $\alpha_0$, Eq.~\eqref{eq:anchor}",r"$\clip(\alpha_0+\beta\tanh(g),0,1)$; Eq.~\eqref{eq:bound}",
         r"Low confidence/non-finite input returns Global; Eq.~\eqref{eq:fallback}",
         "Reference-relative utility, loss and intervention; empirical trade-offs"]]
    table("nearest",r"Nearest-method distinctions. DynaSemble: Section 3 and the pinned released core (Appendix~\ref{app:dynasemble}). MoSE: Sections 3.5--3.6, Eqs.~(1)--(9), and Appendix A. The last three rows use the same preference fit in our matched study and their actions are projected onto the declared grid. A fixed coefficient, numerical zero or unchanged ranking is not an explicit rejection rule. All ADC queries still receive a prediction.",
          "tab:nearest-methods",r"@{}>{\raggedright\arraybackslash}p{.11\linewidth}"+ragged*5+r"@{}",
          ["Method","Supervision","Reference","Action space","Rejection meaning","Empirical question"],rows,True)

    summary=pd.read_csv(OUT/"pair_summary.csv").set_index(["pair","split"])
    conditional=pd.read_csv(OUT/"conditional_summary.csv").set_index(["pair","stratum","value","split"])
    def pct(v): return "--" if pd.isna(v) else f"{100*v:.2f}"
    def num(v): return "--" if pd.isna(v) else f"{v:+.6f}"
    rows=[]
    for pair,label in PAIRS.items():
        for split,short in (("dev_oof","DEV"),("test","TEST")):
            r=summary.loc[pair,split]
            rows.append([label,short,pct(r.a_win_rate),pct(r.b_win_rate),pct(r.tie_rate),
                         f"{r.oracle_gap:.6f}",f"{r.mean_abs_log2_rank_ratio:.3f}",pct(r.gold_hit10_disagree_rate),num(r.delta_mrr)])
    table("pairs",r"Endpoint complementarity and actual ADC utility for every fixed pair. A/B/= are winner/tie percentages; $O$ is endpoint Oracle minus best single MRR; $D_r=\E|\log_2(r_A/r_B)|$; $X_{10}$ is gold Hits@10 disagreement (\%). $U$ is ADC minus the corresponding Global. DEV utility uses grouped held-out predictions, TEST uses the locked policy. These are descriptive observations, not new pair selection.",
          "tab:complement-pairs","llrrrrrrr",["Pair","Split","A (\\%)","B (\\%)","= (\\%)",r"$O$",r"$D_r$",r"$X_{10}$",r"$U$"],rows)

    for name,stratum,values,caption in (
        ("winners","winner",["A","B","tie"],"Endpoint-winner strata. The winner is an answer-aware evaluation label, never an inference feature."),
        ("rank_bins","rank_ratio_bin",RANK_BINS,r"Gold-rank-ratio strata, using $\rho=\max(r_A,r_B)/\min(r_A,r_B)$. These compare endpoint gold ranks, not complete candidate orderings.")):
        rows=[]
        for pair,label in list(PAIRS.items())[:4]:
            for value in values:
                d,t=(conditional.loc[pair,stratum,value,split] for split in ("dev_oof","test"))
                rows.append([label,value,pct(d.mass),num(d.delta_mrr),num(d.delta_contribution),
                             pct(t.mass),num(t.delta_mrr),num(t.delta_contribution)])
            if label!="D-A": rows.append([])
        table(name,caption+r" Mass is the percentage of seed-direction observations. $U_g$ is conditional ADC-minus-Global MRR; $C_g=\Pr(g)U_g$ is its contribution to overall utility. Contributions sum to the pair's $U$. DEV uses OOF predictions. All additional-pair strata remain in the accompanying data; no group is used to fit or choose a policy.",
              "tab:complement-"+name,"llrrrrrr",["Pair","Stratum","DEV \\%",r"DEV $U_g$",r"DEV $C_g$","TEST \\%",r"TEST $U_g$",r"TEST $C_g$"],rows)

    rows=[]
    for pair,label in list(PAIRS.items())[:4]:
        for value in SUPPORT:
            d,t=(conditional.loc[pair,"support",value,split] for split in ("dev_oof","test"))
            rows.append([label,value,f"{int(d.n_directional_queries):,}/{int(t.n_directional_queries):,}",
                         pct(d.b_win_rate),num(d.delta_mrr),pct(t.b_win_rate),num(t.delta_mrr),num(t.delta_contribution)])
        if label!="D-A":rows.append([])
    table("support",r"Observable modality support of the known query entity only: $h$ for tail prediction and $t$ for head prediction. $Q$ counts directional triple queries with seeds collapsed, not independent entities; each has three seed observations. B is the secondary-winner percentage. Empty states have undefined conditional means and zero contribution. Tiny DB15K groups are retained without inferential claims. Coverage flags measure availability, not quality, and are never passed to ADC.",
          "tab:complement-support","lllrrrrr",["Pair","Known support",r"$Q$ DEV/TEST","DEV B \\%",r"DEV $U_g$","TEST B \\%",r"TEST $U_g$",r"TEST $C_g$"],rows)

    rows=[]
    for pair,label in list(PAIRS.items())[:4]:
        for value in ("head","tail"):
            d,t=(conditional.loc[pair,"direction",value,split] for split in ("dev_oof","test"))
            rows.append([label,value,pct(d.b_win_rate),f"{d.oracle_gap:.6f}",num(d.delta_mrr),
                         pct(t.b_win_rate),f"{t.oracle_gap:.6f}",num(t.delta_mrr)])
    table("directions",r"Direction-conditioned secondary-winner share, endpoint Oracle gap $O$ and actual ADC utility. The direction is observable, but the outcome statistics use labels only after prediction. DEV utility is grouped OOF. A larger opportunity or winner share does not guarantee larger or positive utility.",
          "tab:complement-directions","llrrrrrr",["Pair","Direction","DEV B \\%",r"DEV $O$",r"DEV $U$","TEST B \\%",r"TEST $O$",r"TEST $U$"],rows)

    relations=pd.read_csv(OUT/"relation_summary.csv.gz")
    rd=relations[relations.scope=="relation_direction"]
    rd=rd[rd.split=="dev_oof"].merge(rd[rd.split=="test"],on=["pair","relation_id","direction"],
                                       how="outer",suffixes=("_dev","_test"),validate="one_to_one",indicator=True)
    rd["plotted"]=(rd._merge=="both")&(rd.n_triples_dev>=30)&(rd.n_triples_test>=30)
    counts=[]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"savefig.facecolor":"white"})
    fig,axes=plt.subplots(2,2,figsize=(7.0,5.3),layout="constrained")
    for ax,(pair,label) in zip(axes.flat,list(PAIRS.items())[:4]):
        all_cells=rd[rd.pair==pair]; cells=all_cells[all_cells.plotted]
        for direction,color,marker in (("head","#2563a6","o"),("tail","#c47426","^")):
            part=cells[cells.direction==direction]
            size=8+1.4*np.sqrt(np.minimum(part.n_triples_dev,part.n_triples_test))
            ax.scatter(1000*part.delta_mrr_dev,1000*part.delta_mrr_test,s=size,c=color,marker=marker,
                       alpha=.65,edgecolors="white",linewidths=.3,label=direction)
        ax.axhline(0,color="#7c858d",lw=.7);ax.axvline(0,color="#7c858d",lw=.7)
        ax.set_title(f"{label}: {len(cells)} relation-direction cells",loc="left",fontweight="bold")
        ax.set_xlabel("DEV OOF ADC − Global (×10⁻³ MRR)")
        ax.set_ylabel("TEST ADC − Global (×10⁻³ MRR)")
        ax.grid(alpha=.12)
        if label=="W-N":ax.legend(frameon=False,fontsize=7,loc="upper left")
        both_zero=int(((cells.delta_mrr_dev==0)&(cells.delta_mrr_test==0)).sum())
        if both_zero:
            ax.text(.97,.03,f"{both_zero} cells at (0, 0)",transform=ax.transAxes,
                    ha="right",va="bottom",fontsize=7,color="#46505a")
        counts.append(dict(pair=pair,all_relation_direction_cells=len(all_cells),plotted_cells=len(cells),
                           excluded_cells=len(all_cells)-len(cells),
                           both_zero=both_zero,
                           dev_positive=int((cells.delta_mrr_dev>0).sum()),test_positive=int((cells.delta_mrr_test>0).sum()),
                           dev_zero=int((cells.delta_mrr_dev==0).sum()),test_zero=int((cells.delta_mrr_test==0).sum()),
                           dev_observation_coverage=float(cells.n_observations_dev.sum()/summary.loc[(pair,"dev_oof"),"n_observations"]),
                           test_observation_coverage=float(cells.n_observations_test.sum()/summary.loc[(pair,"test"),"n_observations"])))
    figure=FIGURES/"relation_direction_utility.pdf"
    fig.savefig(figure)
    fig.savefig(figure.with_suffix(".png"),dpi=180)
    plt.close(fig)
    figures[figure.relative_to(PAPER).as_posix()]=sha(figure)
    counts_path=OUT/"relation_plot_coverage.json"
    counts_path.write_text(json.dumps(counts,indent=2)+"\n",encoding="utf-8",newline="\n")

    report="""# A01/A04/E01：创新范围与互补诊断核查（2026-09-12）

最低关闭条件以“有来源的最近邻差异 + 六 pair 的互补诊断 + 收窄多模态专属性”满足。没有重训或新策略选择；保留 C07/C08 的后验证据地位。

## 文献边界与贡献

[DynaSemble §3](https://aclanthology.org/2024.acl-short.20.pdf) 已从候选分数分布生成 query-dependent 权重。[MoSE §§3.5–3.6/Appendix A](https://aclanthology.org/2022.emnlp-main.719.pdf) 包含平均、关系 boosting 和实例元学习集成。论文差异表逐行给出监督、参照、行动、拒绝与实证对象，相关公式/代码定位保存在 `paper_a_nearest_methods_reference.json`。

ADC 研究的对象是围绕 DEV Global 的行动映射、范围与拒绝调整机制；不声称动态集成、分数几何、tanh、clip、收缩或拒绝操作本身是新发明。Shrink-gate 和 Clip-g 也具有参照与拒绝语义；已有同输入/同拟合对象/同预算实验未支持 tanh 的独特优势。MoSE 是机制比较，没有本研究新跑的 MoSE 数值。

ADC 的输入仍为两套候选分数与方向，既不感知显式模态质量，也不接收模态覆盖字段。论文将其定位为在 MMKGC 上研究的一般分数后处理；不同架构的模态处理是研究动机，当前观察不能因果解释强弱差异，也不能声称这些集成现象为多模态独有。

## 数据与结果

所有数字均从 B07/C04 修正并有 SHA-256 的资产重新聚合。DEV 收益连接到 P3 grouped OOF 输出；TEST 连接到锁定策略。完整 query ID/seed/direction/triple 一一匹配，端点 rank/RR、Oracle 恒等式、分组覆盖和贡献分解均通过检查。

| Pair | DEV B 胜率 | DEV Oracle gap | DEV OOF ADC−Global | TEST B 胜率 | TEST Oracle gap | TEST ADC−Global |
|---|---:|---:|---:|---:|---:|---:|
"""
    for pair,label in PAIRS.items():
        d,t=(summary.loc[pair,split] for split in ("dev_oof","test"))
        report+=f"| {label} | {100*d.b_win_rate:.2f}% | {d.oracle_gap:.6f} | {d.delta_mrr:+.6f} | {100*t.b_win_rate:.2f}% | {t.oracle_gap:.6f} | {t.delta_mrr:+.6f} |\n"
    report+="""
六 pair 均有端点互补机会，但不能把它当作 ADC 可利用性证明。例如 D-N 的 DEV Oracle gap 为 0.036234，OOF ADC 收益仍略负；D-NA 的 TEST 收益也略负。这些 pair 包含失败/近零情形，不能用 TEST 正增益筛选保留。

W-N 的 TEST 净收益 +0.002447 中，A 获胜组贡献 +0.002780，B 获胜组贡献 −0.000414，tie 组 +0.000081。ADC 收益不是简单地“识别次模型赢家并切换”；其动作围绕 Global，标签预测正确与 RR 改善也不是同一目标。D-N 的 B 获胜组贡献为正，但不足以抵消 A 获胜组损失。

排名分歧明确限定为 gold 端点 rank：并非完整候选排序距离。W-N 在 rank ratio (1,2] 组贡献 +0.002452，(2,4] 组略负；D-N 在 ratio>4 组贡献 −0.000120。不能据此推出“更大分歧带来更大收益”。winner/rank 分层是答案感知的事后诊断，从未加入 selector。

方向诊断同时报告 DEV/TEST 的胜率、Oracle gap 和收益。W-N TEST 的 head 次模型胜率较高，ADC 收益却低于 tail。关系图保留所有满足预先记录计数门槛的正、负、零收益 cell；完整 CSV 包括未画出的稀疏关系，未按效果挑关系。

## 可观测模态支持

两个 seed 1 checkpoint 的 SHA-256 和 `has_img/has_text` tensor hashes 与 C01–C03 记录一致；此前全部 18 个 checkpoint 已验证相同 canonical buffers。已知端 lookup 的独立函数只有方向和已知实体决定返回值，未知 gold 的 ID/属性变化不改变支持分组。mask 只是诊断输入，不改变 ADC 的 13 个特征。

W-N/W-A TEST 的已知 query 支持数（去 seed directional triples）为 neither 293、image-only 393、text-only 0、both 7,862；D-N/D-A 为 7、5,035、34、14,728。每个有三个 seed observation，不能把这些重复观测当独立样本。PDF 保留零支持行和极小样本行。

W-A 的 image-only TEST 组收益为 −0.000404，both 组为 +0.001063；D-N both 组为 −0.000140，image-only 组为 +0.000088。该关联没有稳定的单向“缺失越多/越少越受益”规律；availability 不是 quality，实体/关系组成和缺失机制可能混杂。没有训练模态感知 selector 或进行因果模态消融。

## 复核

```powershell
python scripts/analyze_paper_a_complementarity.py
python scripts/build_paper_a_complementarity_assets.py
python -m pytest tests/test_complementarity_diagnostics.py -q
python paper_a_draft/scripts/compile.py
python paper_a_draft/scripts/verify_draft.py
```

这是 CPU 缓存聚合，无需服务器重评。8 项针对信息边界、OOF join 与指标分解的测试覆盖了 gold/未知端变动、已知端敏感性、空分组、Oracle 与混合排序端点的区别。`complementarity_source_manifest.json` 绑定分析输入、输出、参考定位、图和六张表。所有统计均为描述性，未新增显著性检验。
"""
    REPORT.write_text(report,encoding="utf-8",newline="\n")
    sources=dict(audit["sources"])
    for path in [OUT/"audit.json",counts_path,REFERENCE,REPORT,Path(__file__)]+[OUT/name for name in audit["outputs"]]:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    # Bind the already-verified matched/control receipts, without rerunning models.
    for rel in ("paper_a_draft/matched_source_manifest.json","paper_a_draft/dynasemble_source_manifest.json"):
        sources[rel]=sha(ROOT/rel)
    (PAPER/"complementarity_source_manifest.json").write_text(json.dumps(dict(
        version="complementarity_review_v1",sources=sources,tables=tables,figures=figures),indent=2)+"\n",encoding="utf-8",newline="\n")
    print(f"Built {len(tables)} tables, one figure and report; {len(sources)} source bindings")


if __name__=="__main__":
    main()
