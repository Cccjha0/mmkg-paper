# A01/A04/E01：创新范围与互补诊断核查（2026-09-12）

最低关闭条件以“有来源的最近邻差异 + 六 pair 的互补诊断 + 收窄多模态专属性”满足。没有重训或新策略选择；保留 C07/C08 的后验证据地位。

## 文献边界与贡献

[DynaSemble §3](https://aclanthology.org/2024.acl-short.20.pdf) 已从候选分数分布生成 query-dependent 权重。[MoSE §§3.5–3.6/Appendix A](https://aclanthology.org/2022.emnlp-main.719.pdf) 包含平均、关系 boosting 和实例元学习集成。论文差异表逐行给出监督、参照、行动、拒绝与实证对象，相关公式/代码定位保存在 `paper_a_nearest_methods_reference.json`。

ADC 研究的对象是围绕 DEV Global 的行动映射、范围与拒绝调整机制；不声称动态集成、分数几何、tanh、clip、收缩或拒绝操作本身是新发明。Shrink-gate 和 Clip-g 也具有参照与拒绝语义；已有同输入/同拟合对象/同预算实验未支持 tanh 的独特优势。MoSE 是机制比较，没有本研究新跑的 MoSE 数值。

ADC 的输入仍为两套候选分数与方向，既不感知显式模态质量，也不接收模态覆盖字段。论文将其定位为在 MMKGC 上研究的一般分数后处理；不同架构的模态处理是研究动机，当前观察不能因果解释强弱差异，也不能声称这些集成现象为多模态独有。

## 数据与结果

所有数字均从 B07/C04 修正并有 SHA-256 的资产重新聚合。DEV 收益连接到 P3 grouped OOF 输出；TEST 连接到锁定策略。完整 query ID/seed/direction/triple 一一匹配，端点 rank/RR、Oracle 恒等式、分组覆盖和贡献分解均通过检查。

| Pair | DEV B 胜率 | DEV Oracle gap | DEV OOF ADC−Global | TEST B 胜率 | TEST Oracle gap | TEST ADC−Global |
|---|---:|---:|---:|---:|---:|---:|
| W-N | 44.25% | 0.038397 | +0.003070 | 43.25% | 0.042181 | +0.002447 |
| W-A | 36.17% | 0.029596 | +0.001204 | 35.19% | 0.031011 | +0.001014 |
| D-N | 28.03% | 0.036234 | -0.000009 | 28.34% | 0.034719 | -0.000081 |
| D-A | 29.11% | 0.031880 | +0.000036 | 29.05% | 0.030434 | +0.000212 |
| W-NA | 29.53% | 0.038671 | +0.000147 | 29.02% | 0.039312 | +0.000139 |
| D-NA | 39.42% | 0.045779 | +0.000031 | 39.41% | 0.044812 | -0.000015 |

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
