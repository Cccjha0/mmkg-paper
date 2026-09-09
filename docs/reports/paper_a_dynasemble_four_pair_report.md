# Paper A：DynaSemble 四主模型组合实验报告

## 当前状态

**待 A100 执行。** 截至 2026-09-09，协议审计、dataset/pair-independent evaluator、四组启动脚本、DEV lock、matched reporting 与汇总程序已经就绪；尚未在本机或其他设备启动本轮 selector training、DEV evaluation 或 TEST evaluation，因此本报告不包含推断值或占位数值。

## 审计结论

1. OpenBG-IMG faithful reproduction 的核心算法实现可以直接泛化：released 4-feature selector、MLP topology、initialization、Adam、margin loss、9,999 strict negatives、batch size 16、单 epoch 和 training-direction behavior 均保持不变。
2. 原 evaluator 将 OpenBG、M-Hyper 和 NativE 写死，且其 `score_expert_block` 调用已落后于当前公共 scorer 的返回接口；现已改成 dataset/pair-independent 参数，同时保留旧 OpenBG 默认 CLI 语义。
3. 四个目标组合的 18 个基础 checkpoint（两个数据集各三个 M-Hyper、NativE、AdaMF-MAT seeds）均存在；本实验只加载其 `best.ckpt`，不会重训 base MMKGC models。
4. 四组均已有 DEV-only Global selection、exact full-ranking DEV/TEST query rows 以及 locked Query-soft/Anchored query rows，可用于逐 query matched reproduction 与统一总表。
5. 已有 Anchored/Query-soft TEST 结果构成既知外部结果，但不会进入 DynaSemble 的配置、selector 训练、early stopping 或 orientation selection。DynaSemble 配置完全由已审计 protocol 冻结。
6. released asymmetric parameterization 能精确到达 fixed secondary endpoint，却不能以有限 learned weight 精确到达 primary endpoint。四组中除 MKG-W / M-Hyper + NativE (`alpha=0.6`) 外，其余三组当前 DEV-locked Global 均为 `alpha=1`（纯 M-Hyper）；本实验为保持 faithful baseline 不增加 primary bypass 或 fallback，最终解释必须把这一结构性限制与一般性的 query adaptation 能力区分开。

## 冻结实现

- 通用 evaluator：`scripts/eval_paper_a_dynasemble.py`
- 兼容实现主体：`scripts/eval_openbg_dynasemble.py`
- 四组 A100 runner：`scripts/run_paper_a_dynasemble_four_pair.ps1`
- 总表构建器：`scripts/build_paper_a_dynasemble_four_pair_summary.py`
- 正式协议：`docs/protocols/paper_a_dynasemble_four_pair_protocol.md`

四组的统一参数化为：secondary 固定权重 1，reliable primary M-Hyper 接收 learned non-negative weight。对 M-Hyper + NativE，这与旧 OpenBG protocol 完全同向；对 M-Hyper + AdaMF-MAT，仅替换 fixed secondary 的模型身份。

## 待执行命令

```powershell
pwsh -File scripts/run_paper_a_dynasemble_four_pair.ps1 -Stage dev -Python python -Device cuda
```

确认四个 pair 目录均生成 `lock.json` 后：

```powershell
pwsh -File scripts/run_paper_a_dynasemble_four_pair.ps1 -Stage test -Python python -Device cuda
```

TEST 阶段结束后 runner 自动生成 `outputs/paper_a_safe_correction/dynasemble/four_pair_summary.csv`。

## 结果表

尚未执行。A100 运行完成后，应以生成的 `four_pair_summary.csv` 中 `scope_type=overall` 的四行为主表，并以 `direction`、`seed`、`seed_direction` 行检查异质性。

| Dataset | Pair | Primary | Global | Query-soft | DynaSemble | Anchored | DynaSemble − Global | Anchored − Global |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W | M-Hyper + NativE | pending | pending | pending | pending | pending | pending | pending |
| MKG-W | M-Hyper + AdaMF-MAT | pending | pending | pending | pending | pending | pending | pending |
| DB15K | M-Hyper + NativE | pending | pending | pending | pending | pending | pending | pending |
| DB15K | M-Hyper + AdaMF-MAT | pending | pending | pending | pending | pending | pending | pending |

## 对核心问题的当前回答

在 A100 结果生成前，不能判断 DynaSemble 是否在四个 strong-primary MMKGC pairs 上表现 pair-sensitive。现有 OpenBG 单 pair 结果只能提供动机，不能替代本轮四 pair 检验。最终结论必须同时查看：

- 四组 pooled `DynaSemble − Global`；
- original-triple clustered bootstrap 95% CI；
- head/tail 是否方向一致；
- 三个 paired seeds 是否存在 collapse 或符号翻转；
- DynaSemble learned-weight zero fraction 与 effective-alpha distribution。

若符号、CI 或 seed/direction 稳定性随 pair 改变，将支持 pair sensitivity；若四组稳定同向且幅度接近，则应否定或弱化该叙事。无论结果如何，不允许根据 TEST 重新设定 DynaSemble 配置。
