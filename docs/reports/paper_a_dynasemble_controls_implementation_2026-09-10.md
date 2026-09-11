# D01–D03 实现交付记录

后续：服务器结果已回传，见 [2026-09-11 结果核验](paper_a_dynasemble_controls_review_2026-09-11.md)。以下保留实现交付当日状态，不代表当前实验尚未运行。

2026-09-10。状态：源码核查、实验入口、锁定/诊断/回传流程和轻量单测已完成；真实实验留待服务器运行，D01/D03 尚未关闭。未在本机进行真实候选评分、selector实验搜索或大规模bootstrap。

## 已发现并处理

1. R0训练时先抽样再构造特征，而上游完整分布统计在gather之前。新R1纠正该路径，R0原文件和模型不覆盖。
2. R0和上游的初始化分布相同，但uniform调用发生在输出层构造前后的顺序不同，同seed参数不同。R1已按上游顺序构造，并直接加载上游类进行参数、前向、loss、梯度一致性单测。
3. 上游默认 YAML 为三模型宽度32，batch因数据集为16或24，不能称现有双模型宽度16/batch16照搬所有配置；它们的来源与取舍已列明。
4. 论文描述DEV训练；检出源码的默认loader/trainer从dataset[0]取训练样本。无法从代码推出作者未记录的外部文件替换。本项目明确坚持DEV正监督，不猜测二者如何在作者运行中对应。
5. 源码记录best_epoch并不等于加载最佳epoch；新的训练轮数只由内部held-out DEV选择，full-DEV refit后锁定。

## 新实验能力

- R1源码核心对齐、R2初始化修复、R3固定一轮Softplus、R3 DEV调参、R4角色交换，与已有R0并列。
- 四主组合；每个新版本3个base seed×3个selector seed，全数保留。每组81条CV轨迹、45个最终selector。
- 独立DEV loader不打开TEST；fold训练负过滤不读取held-out DEV正例；缓存/特征与gold/filter分离；四组DEV锁齐全才能TEST。
- 逐层梯度、更新范数、loss、权重及preactivation分布，完整随机性矩阵；gold rank匹配和完整弱序探针分开报告。
- checkpoint/输入数据/cache/源码/runtime哈希、断点重入检查、失败保留、拒绝无CUDA时自动转CPU。
- 自动生成不含候选cache的回传zip。

协议与指令见 [paper_a_dynasemble_controls.md](G:/mmkg-project-research/docs/protocols/paper_a_dynasemble_controls.md)。源码对照见 `docs/audits/dynasemble_source/`。服务器输出根目录已加入.gitignore。

## 本机验证

新增 `tests/test_dynasemble_controls.py` 以12实体的合成scorer检验完整cache→三折DEV→refit→lock→TEST→summary→bundle链路和篡改拒绝，不读取真实实验资产。与B07/C04及相关既有测试一并运行，55项通过（15.15秒）；PowerShell包装器的Plan阶段也已验证。测试命令：

```powershell
python -m pytest tests/test_dynasemble_controls.py tests/test_paper_a_information_boundary.py tests/test_paper_a_boundary_rerun_audit.py ml/training/tests/test_filtered_ranking_optimization.py ml/training/tests/test_mhyper_reciprocal.py ml/training/tests/test_protocol_isolation.py ml/training/tests/test_anchored_dynamic.py tests/test_paper_a_efficiency_benchmark.py tests/test_dynasemble_generalization.py -q -p no:tmpdir
```

服务器实际训练速度、显存占用与健康性尚无实测，不据合成测试保证真实selector健康或优于Global。若所有预定健康候选仍缺乏有效优化，保留全部结果并继续定位，不能删seed、按TEST挑方向或绕过锁重调参数。
