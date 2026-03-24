# 输入特征缓存流程整改说明

## 1. 目的

本文档用于完成阶段 `4.1 输入特征流程整改` 中的前四项任务：

- 检查当前文本缓存构建脚本保存了哪些内容
- 检查当前图像缓存构建脚本保存了哪些内容
- 设计新的缓存格式，保存原始文本特征
- 设计新的缓存格式，保存原始图像特征

目标是先把现状和新格式设计说清楚，再进入后续代码改造。

## 2. 当前文本缓存脚本保存内容

当前文本缓存脚本为：

- [build_cache_openbg_img_text.py](/E:/learn/R&D/mmkg-project-research/ml/training/scripts/build_cache_openbg_img_text.py)

### 2.1 当前流程

当前文本缓存流程如下：

1. 读取 `entity2text.tsv`
2. 用 `SentenceTransformer` 编码得到原始文本特征 `raw`
3. 随机初始化一个线性层 `proj`
4. 将 `raw` 投影到 `d=256`
5. 将投影后的结果保存为 `text_emb.pt`

### 2.2 当前实际保存文件

当前 `data/cache/openbg_img` 目录中，文本侧相关文件包括：

- `text_emb.pt`
- `has_text.pt`
- `text_proj.pt`
- `text_meta.json`

### 2.3 当前问题

当前文本脚本**没有保存原始文本特征**，只保存了随机投影后的 `text_emb.pt`。

这会带来几个问题：

- 训练阶段无法重新学习文本投影层
- 文本特征质量被一次性随机投影锁定
- 后续若想修改投影结构，必须重新编码全部文本
- 不利于做更公平的模态对照实验

## 3. 当前图像缓存脚本保存内容

当前图像缓存脚本为：

- [build_cache_openbg_img_image.py](/E:/learn/R&D/mmkg-project-research/ml/training/scripts/build_cache_openbg_img_image.py)

### 3.1 当前流程

当前图像缓存流程如下：

1. 扫描 `images_root`
2. 用 `CLIPModel.get_image_features()` 提取原始图像特征
3. 将原始图像特征保存到 `img_emb_raw.pt`
4. 随机初始化一个线性层 `proj`
5. 将原始图像特征投影到 `d=256`
6. 保存投影后的结果 `img_emb.pt`

### 3.2 当前实际保存文件

当前 `data/cache/openbg_img` 目录中，图像侧相关文件包括：

- `img_emb_raw.pt`
- `img_emb.pt`
- `has_img.pt`
- `img_proj.pt`
- `img_meta.json`

### 3.3 当前问题

图像侧已经比文本侧更接近合理方案，因为它至少保留了 `img_emb_raw.pt`。

但仍然存在问题：

- 训练阶段默认仍读取 `img_emb.pt`，而不是 `img_emb_raw.pt`
- 随机投影结果仍被当作冻结输入使用
- `img_proj.pt` 只保留了当时的随机投影权重，并未进入训练

因此，图像缓存虽然比文本缓存更完整，但当前训练链路仍没有真正利用原始图像特征。

## 4. 当前训练链路对缓存的使用方式

当前模型构建逻辑位于：

- [build_model.py](/E:/learn/R&D/mmkg-project-research/ml/training/src/models/build_model.py)

当前做法是：

- 直接读取 `text_emb.pt`
- 直接读取 `img_emb.pt`
- 直接将它们注册为模型 buffer

对应模型实现中：

- [openbg_img_gated_lp.py](/E:/learn/R&D/mmkg-project-research/ml/training/src/models/openbg_img_gated_lp.py)
- [early.py](/E:/learn/R&D/mmkg-project-research/ml/training/src/models/fusion/early.py)

这意味着当前训练时真正参与学习的是：

- `t_adapter`
- `v_adapter`
- fusion 模块
- residual 模块

而不是：

- 文本原始特征到任务空间的投影
- 图像原始特征到任务空间的投影

## 5. 新的缓存格式设计原则

新的缓存格式应满足以下原则：

1. 原始特征必须保存
2. 是否存在对应模态的信息必须单独保存
3. 元信息必须记录编码器名称、维度和来源路径
4. 缓存只负责“特征提取与存储”，不再负责“随机投影到任务维度”
5. 任务维度投影应转移到模型内部，由训练阶段共同优化

## 6. 新的文本缓存格式设计

### 6.1 建议保留文件

建议文本缓存改为保存以下文件：

- `text_feat_raw.pt`
  - 形状：`[num_entities, text_raw_dim]`
  - 内容：文本编码器直接输出的原始文本特征

- `has_text.pt`
  - 形状：`[num_entities]`
  - 内容：实体是否具有文本描述

- `text_meta.json`
  - 内容包括：
    - `text_encoder`
    - `num_entities`
    - `raw_dim`
    - `entity2text`
    - `feature_file`
    - `notes`

### 6.2 建议移除的旧角色

以下内容不应再作为主训练输入：

- `text_emb.pt`
- `text_proj.pt`

可以保留兼容一段时间，但不再作为新流程的核心产物。

### 6.3 设计理由

这样做的好处是：

- 文本编码只需做一次
- 后续可以自由替换投影层结构
- 训练阶段可以联合优化文本投影
- 多模型对比更公平

## 7. 新的图像缓存格式设计

### 7.1 建议保留文件

建议图像缓存统一为以下文件：

- `img_feat_raw.pt`
  - 形状：`[num_entities, img_raw_dim]`
  - 内容：图像编码器直接输出的原始图像特征

- `has_img.pt`
  - 形状：`[num_entities]`
  - 内容：实体是否具有图像

- `img_meta.json`
  - 内容包括：
    - `image_encoder`
    - `num_entities`
    - `raw_dim`
    - `images_root`
    - `entity2text`
    - `feature_file`
    - `notes`

### 7.2 建议降级的旧文件

以下内容不应再作为主训练输入：

- `img_emb.pt`
- `img_proj.pt`

当前已有的 `img_emb_raw.pt` 可以迁移命名为 `img_feat_raw.pt`，以与文本侧统一。

### 7.3 设计理由

这样做的好处是：

- 图像侧与文本侧缓存语义统一
- 模态缓存和任务投影职责分离
- 后续可尝试不同图像投影层，而不必反复提取 CLIP 特征

## 8. 建议的新旧兼容策略

为避免一次性重构导致训练流程中断，建议采用过渡期方案。

### 8.1 过渡期可兼容读取

模型加载时优先级建议如下：

#### 文本侧

1. 优先读取 `text_feat_raw.pt`
2. 若不存在，则回退读取旧版 `text_emb.pt`

#### 图像侧

1. 优先读取 `img_feat_raw.pt`
2. 若不存在，则回退读取旧版 `img_emb_raw.pt`
3. 若也不存在，则最后回退读取旧版 `img_emb.pt`

### 8.2 过渡期文档要求

- 在 `meta.json` 中明确标记当前缓存属于旧版还是新版
- 在训练日志中打印当前使用的是原始特征还是旧版投影特征

## 9. 建议的下一步实现任务

在完成本设计后，后续应继续推进以下事项：

- 在缓存脚本中实际生成 `text_feat_raw.pt`
- 将 `img_emb_raw.pt` 统一迁移或兼容为 `img_feat_raw.pt`
- 在模型内增加文本投影层
- 在模型内增加图像投影层
- 修改 `build_model.py`，优先读取原始特征缓存
- 更新配置和 README

## 10. 当前结论

这四项任务已经可以形成明确结论：

1. 当前文本缓存脚本只保存了随机投影后的特征，没有保存原始文本特征。
2. 当前图像缓存脚本已经保存了原始图像特征，但训练链路没有真正利用它。
3. 新的缓存格式应统一为“保存原始模态特征 + 保存模态存在标记 + 保存元信息”。
4. 任务维度投影应从缓存阶段移入模型训练阶段。

在此基础上，阶段 `4.1` 的后续任务就可以进入代码实现层面。
