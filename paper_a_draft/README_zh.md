# Paper A 英文论文初稿

**修复后工作稿（2026-09-10）：** 六个 MKG-W/DB15K 组合与四组 DynaSemble 已完成重导出/重拟合/重评；正文、7 张新表及 `main.pdf` 已更新。当前主结论为四主组合中三组正增益，D-N 的区间跨零。ADC/Query-soft 最终网格权重通过多 gold query 一致性检查；部分连续输出仍有约 1e-7 数值差异，未宣称任意批次下逐位一致。完整关闭口径见 `../docs/reports/paper_a_information_boundary_rerun_review_2026-09-10.md`。

LaTeX 编译所需文件在本目录内；重新出表和来源核验需要仓库中的 `information_boundary_v2` 原始导出。原始 processed 数据仍未复制到本机，但分析新导出和编译无需重新运行基础模型。目录外的旧 zip 尚未更新。

## 阅读与编辑

- `main.pdf`：已编译的全文，含主文、参考文献和附录。
- `main.tex`：论文主文件。
- `references.bib`：7 条已核实的关键参考文献。
- `main.bbl`：本次编译生成的参考文献，可用于归档。
- `figures/method_overview.pdf`：当前唯一保留的方法示意图；旧经验诊断图不再引用。
- `tables/rerun/`：从修复后导出生成的 7 个表格片段；正文另含数据集计数表。
- `rerun_source_manifest.json`：当前表格及重评资产的 SHA-256。
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

在仓库根目录运行以下命令；仅读取新导出、生成表格和文稿，不重训模型或修改 TEST 参数：

```powershell
python scripts/audit_paper_a_boundary_rerun.py --finalize-only
python scripts/build_paper_a_boundary_rerun_tables.py
python paper_a_draft/scripts/compile.py
python paper_a_draft/scripts/verify_draft.py
```

如需重新计算 bootstrap，运行审计脚本时省略 `--finalize-only`。`verify_draft.py` 需要 PyMuPDF 和 Pillow，审计/出表需要 numpy、pandas、torch、scikit-learn；本地使用 `E:/develop/Miniconda3/python.exe`。旧 `build_assets.py` 和 `snapshot_sources.py` 仅处理历史快照，不是当前出表入口。W-N/W-A 分别为 MKG-W 的 M-Hyper+NativE / M-Hyper+AdaMF-MAT，D-N/D-A 对应 DB15K；W-NA/D-NA 为两数据集的 NativE+AdaMF-MAT。

## 写作和证据处理

已按你提供的 anti-defensive-writing 原则起草：段落先陈述发现，必要的实验范围与方法限制集中写入 Experimental Setup、Discussion 和附录。中文梳理稿中的编辑提醒没有直接搬进正文。

正文落实了这些关键口径：

1. α 始终表示 primary 权重；静态参考可以是纯 primary。
2. g 是 winner-supervised logistic preference signal，修正由后续映射产生。
3. MKG-W 为原协议声明的确认性验证，DB15K 为次级复现。
4. 新 TEST 区间均从修复后逐查询记录计算，使用原始三元组聚类、10,000 次 percentile bootstrap 和 seed 20260910；旧方法间比较区间不沿用。
5. 正式 DEV 表来自修复后的 expanded-policy P3 held-out 记录，不混入 full-DEV resubstitution。新消融固定原 DEV 锁，仅改变 beta 或 tau，不依据 TEST 重调参。
6. 尚未重新验证的 A100 时间、OpenBG 结果、confidence/radius/coverage 诊断已从当前正文移除。需要恢复这些主张时再单独补运行。
7. 数据集表报告实际导出观测所覆盖的 DEV/TEST triples。DB15K 导出 DEV 为 7,922；原预处理审计中的 raw valid 总数 9,904 是另一计数口径。

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

当前证据为 `outputs/paper_a_safe_correction/information_boundary_rerun_audit/` 的审计、方法汇总、固定消融及 `manuscript_details.json`。原有“77 项审计”属于历史版本，不用于证明新结果；本次边界/协议/审计单测共 43 项通过，当前文档检查记录在 `verification.json`。
