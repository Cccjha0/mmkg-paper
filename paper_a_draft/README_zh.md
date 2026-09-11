# Paper A 英文论文初稿

**修复后工作稿（2026-09-12）：** B07/C04 边界重评、D01–D03 健康 DynaSemble、A03/B04/D08 保守半径及 A02/D04–D06 匹配替代方案已写入正文。新增六族共享同一拟合对象、每族41个候选并允许Global；ADC相对纯收缩三组MRR较高，但Clip-g四组TEST点估计均略高于ADC，特定tanh优势尚未建立。新静态表与单列Oracle数值完整。原ADC/Dyna结果保留，新动作先锁DEV再评价TEST。详见 [匹配替代方案核验](../docs/reports/paper_a_matched_alternatives_review_2026-09-12.md)、[边界核验](../docs/reports/paper_a_information_boundary_rerun_review_2026-09-10.md)、[Dyna核验](../docs/reports/paper_a_dynasemble_controls_review_2026-09-11.md)及[保守半径核验](../docs/reports/paper_a_conservative_radius_review_2026-09-11.md)。

LaTeX 编译所需文件在本目录内；重新分析和来源核验需要 `information_boundary_v2` 原始导出以及回传的 Dyna 对照与缓存证据。编译无需重新运行基础模型。目录外的旧 zip 尚未更新。

## 阅读与编辑

- `main.pdf`：已编译的全文，含主文、参考文献和附录。
- `main.tex`：论文主文件。
- `references.bib`：7 条已核实的关键参考文献。
- `main.bbl`：本次编译生成的参考文献，可用于归档。
- `figures/method_overview.pdf`：方法示意图；`figures/conservative/` 为修复后 DEV 半径、TEST 效用—损失图。旧经验诊断图不再引用。
- `tables/rerun/`、`tables/dynasemble/`、`tables/conservative/`、`tables/matched/`：边界重评、Dyna、保守风险及匹配映射/完整静态表。只以 `main.tex` 实际引用的片段计入当前稿。
- `rerun_source_manifest.json`、`dynasemble_source_manifest.json`、`conservative_source_manifest.json`、`matched_source_manifest.json`：四轮证据与表图的 SHA-256。
- `data/`、`tables/` 下的旧表、`source_manifest.json`、`notes/`：历史快照，不作为 v2 数值证据。
- `scripts/`：编译、生成图表、核验和打包脚本。
- `verification.json`：最近一次自动检查结果；`.build/qa/` 为本地逐页版面检查图，不打入交付压缩包。

主文采用通用 `article` 单栏格式，便于修改内容；作者及机构留空，待确定后填写 `\author{}`。暂未绑定目标期刊模板。

## 编译

推荐将整个文件夹上传至 Overleaf，以 `main.tex` 为主文件、pdfLaTeX 为编译器。现有 PDF 图和 LaTeX 表格已经齐全，编译不需要 Python，也不需要访问原实验目录。

本地已安装 TeX Live 或 MiKTeX 时，在此目录运行：

```powershell
python scripts/compile.py
```

也可使用标准命令：

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

主文件需要的常用包包括 `geometry`、`amsmath`、`amssymb`、`amsthm`、`graphicx`、`booktabs`、`tabularx`、`lmodern`、`microtype`、`enumitem`、`placeins`、`flafter`、`float`、`natbib`、`hyperref`、`caption`。

## 重新生成图表

在仓库根目录运行以下命令可从已核验汇总重新出表和编译，不训练模型或修改 TEST 参数：

```powershell
python scripts/build_paper_a_boundary_rerun_tables.py
python scripts/build_paper_a_dynasemble_review_tables.py
python scripts/build_paper_a_conservative_assets.py
python scripts/build_paper_a_matched_assets.py
python paper_a_draft/scripts/compile.py
python paper_a_draft/scripts/verify_draft.py
```

若要重算本轮 DEV 半径与风险汇总，先运行 `scripts/analyze_paper_a_conservative_radius.py`；它只重建 20 个轻量 CPU 逻辑回归折模型。`verify_draft.py` 需要 PyMuPDF 和 Pillow，审计/出表需要 numpy、pandas、torch、scikit-learn、matplotlib；本地使用 `E:/develop/Miniconda3/python.exe`。旧 `build_assets.py` 和 `snapshot_sources.py` 仅处理历史快照，不是当前出表入口。W-N/W-A 分别为 MKG-W 的 M-Hyper+NativE / M-Hyper+AdaMF-MAT，D-N/D-A 对应 DB15K；W-NA/D-NA 为两数据集的 NativE+AdaMF-MAT。

匹配替代方案采用 `scripts/analyze_paper_a_matched_alternatives.py lock-dev` 和独立的 `apply-test` 两阶段；已有TEST审核时拒绝覆盖DEV锁。正常重新出表不需重跑这两阶段。初次Relation审核的float32表示修正、原锁及选择完全一致的证明保留在结果目录的 `provenance/`。

## 写作和证据处理

已按你提供的 anti-defensive-writing 原则起草：段落先陈述发现，必要的实验范围与方法限制集中写入 Experimental Setup、Discussion 和附录。中文梳理稿中的编辑提醒没有直接搬进正文。

正文落实了这些关键口径：

1. α 始终表示 primary 权重；静态参考可以是纯 primary。
2. g 是 winner-supervised logistic preference signal，修正由后续映射产生。
3. MKG-W 为原协议声明的确认性验证，DB15K 为次级复现。
4. 新 TEST 区间均从修复后逐查询记录计算，使用原始三元组聚类、10,000 次 percentile bootstrap 和 seed 20260910；旧方法间比较区间不沿用。
5. 正式 DEV 表来自修复后的 expanded-policy P3 held-out 记录，不混入 full-DEV resubstitution。新消融固定原 DEV 锁，仅改变 beta 或 tau，不依据 TEST 重调参。
6. 尚未重新验证的 A100 时间、OpenBG 结果、历史 confidence/radius/coverage 诊断已移除；本轮新增的 DEV 半径诊断来自修正后导出。
7. 数据集表报告实际导出观测所覆盖的 DEV/TEST triples。DB15K 导出 DEV 为 7,922；原预处理审计中的 raw valid 总数 9,904 是另一计数口径。
8. 0.50 是历史限定搜索族的上界，未被校准为风险水平；当前最终半径分别为 .45、.50、.20、.50。新增 DEV 分析不改变 TEST 选择。
9. Global 的参考相对损失为零。ADC 相对 R3 四组受损频率与无条件损失较低，但三组条件损失更高；W-N 相对 Query-soft 也不构成所有风险量的支配。
10. ADC与Query-soft共享相同拟合对象；新对照进一步匹配候选数。Clip-g在四组TEST点估计略高于ADC，因此不宣称tanh独特优势。新比较没有新增置信区间，不据小差异宣称显著。
11. RRF、Relation、两个专家、等权与Global均有PDF数值。专家Oracle只在两专家选择集合内取答案知情最大值，不能上界混合分数。

本文是修复后的研究工作稿，不是无条件投稿认证。仍需按目标期刊整理体例，并妥善处理报告列出的连续数值重复性限制。修改前的正文保留在 `.build/main_before_boundary_rerun.tex`。

## 参考文献核实来源

- M-Hyper：[ACL 2026 正式论文](https://aclanthology.org/2026.acl-long.1289/)
- NativE：[原作者论文](https://arxiv.org/abs/2406.17605)，SIGIR 2024，DOI `10.1145/3626772.3657800`
- AdaMF-MAT：[LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1487/)
- DynaSemble：[ACL 2024](https://aclanthology.org/2024.acl-short.20/)
- MoSE：[EMNLP 2022](https://aclanthology.org/2022.emnlp-main.719/)
- SelectiveNet：[ICML 2019 / PMLR](https://proceedings.mlr.press/v97/geifman19a.html)
- OpenBG：[作者 arXiv 版本](https://arxiv.org/abs/2209.15214)

## 数值来源补充

当前汇总证据位于 `outputs/paper_a_safe_correction/` 下的 `information_boundary_rerun_audit/`、`dynasemble_review_audit/`、`conservative_radius_review/`、`matched_alternatives_v1/`。各轮测试和证据范围分别记录在核验报告；匹配映射新增10项测试通过，前轮保守风险5项测试通过。当前文档检查记录在 `verification.json`，逐页 PNG 位于 `.build/qa/`。
