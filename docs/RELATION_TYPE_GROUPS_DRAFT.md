# Relation Type Group Draft

## 1. Purpose

This draft defines an initial relation grouping for `6.2 relation type` on OpenBG-IMG.

The goal is not to produce a final ontology. The goal is to create a first-pass split
that is stable enough for grouped evaluation of:

- `Gate-only`
- `Full Model`
- `Residual-only`

under the current unified `paper_split` and `direction=both` protocol.

## 2. Grouping Principle

Current grouping uses three buckets:

- `visual_relations`
  Relations whose target attribute is often directly observable from product images.
- `weak_visual_relations`
  Relations that are mostly determined by metadata, text, packaging specs, brand knowledge,
  product knowledge, or abstract semantics rather than direct visual evidence.
- `ambiguous_material_relations`
  Relations that may sometimes be inferred from images, but visual evidence is noisy,
  indirect, or highly category-dependent. These are separated out in the first round to
  avoid contaminating the core comparison.

## 3. Recommended First-Round Usage

For the first `6.2` experiment, it is recommended to:

1. Compare `visual_relations` vs `weak_visual_relations`.
2. Exclude `ambiguous_material_relations` from the main table.
3. Report `ambiguous_material_relations` separately only if sample size is sufficient.

This keeps the first relation-type conclusion easier to interpret.

## 4. Draft Groups

### 4.1 visual_relations

These relations are expected to be the most image-sensitive.

| Relation ID | Chinese | English |
|---|---|---|
| `rel_0008` | 色泽 | `color / hue` |
| `rel_0043` | 颜色分类 | `color category` |
| `rel_0000` | 图案 | `pattern` |
| `rel_0048` | 设计细节 | `design details` |
| `rel_0006` | 流行元素 | `fashion element` |
| `rel_0020` | 设计元素 | `design element` |
| `rel_0086` | 样式 | `style` |
| `rel_0064` | 细分风格 | `sub-style` |
| `rel_0134` | 基础风格 | `basic style` |
| `rel_0038` | 服装版型 | `clothing fit` |
| `rel_0109` | 版型 | `fit type` |
| `rel_0032` | 领型 | `collar type` |
| `rel_0066` | 袖型 | `sleeve type` |
| `rel_0106` | 袖长 | `sleeve length` |
| `rel_0117` | 衣长 | `clothing length` |
| `rel_0009` | 裤长 | `pants length` |
| `rel_0062` | 裙长 | `skirt length` |
| `rel_0123` | 裙型 | `skirt type` |
| `rel_0027` | 裤款式 | `pants style` |
| `rel_0024` | 旗袍款式 | `cheongsam style` |
| `rel_0065` | 腰型 | `waist type` |
| `rel_0107` | 衣门襟 | `clothing placket` |
| `rel_0101` | 裤门襟 | `pants placket` |
| `rel_0044` | 闭合方式 | `closure type` |
| `rel_0012` | 是否带帽子 | `with hood or not` |
| `rel_0098` | 佩戴方式 | `wearing method` |
| `rel_0121` | 形状 | `shape` |
| `rel_0079` | 模特实拍 | `model photo` |

### 4.2 weak_visual_relations

These relations are expected to be weakly visual, abstract, or mainly metadata-driven.

| Relation ID | Chinese | English |
|---|---|---|
| `rel_0010` | 品牌 | `brand` |
| `rel_0011` | 品牌类型 | `brand type` |
| `rel_0019` | 品牌归属地 | `brand origin` |
| `rel_0130` | 产地 | `place of origin` |
| `rel_0122` | 省份 | `province` |
| `rel_0005` | 地市 | `city` |
| `rel_0096` | 是否进口 | `imported or not` |
| `rel_0115` | 净含量 | `net content` |
| `rel_0118` | 产品重量 | `product weight` |
| `rel_0085` | 容量 | `capacity` |
| `rel_0097` | 规格 | `specification` |
| `rel_0114` | 具体规格 | `detailed specification` |
| `rel_0042` | 包装规格 | `packaging specification` |
| `rel_0104` | 包装方式 | `packaging method` |
| `rel_0087` | 包装 | `packaging` |
| `rel_0004` | 包装数量 | `package quantity` |
| `rel_0068` | 包装体积 | `package volume` |
| `rel_0108` | 保质期 | `shelf life` |
| `rel_0014` | 储存条件 | `storage condition` |
| `rel_0091` | 上市时间 | `release time` |
| `rel_0078` | 上市年份季节 | `release year/season` |
| `rel_0045` | 适用季节 | `suitable season` |
| `rel_0089` | 适合季节 | `suitable season` |
| `rel_0031` | 适用场景 | `usage scenario` |
| `rel_0016` | 关联场景 | `related scenario` |
| `rel_0090` | 功能 | `function` |
| `rel_0037` | 附加功能 | `additional function` |
| `rel_0050` | 功效 | `efficacy` |
| `rel_0052` | 成分 | `ingredients` |
| `rel_0120` | 原料成分 | `raw ingredients` |
| `rel_0129` | 材质成分 | `material composition` |
| `rel_0061` | 是否为有机食品 | `organic or not` |
| `rel_0022` | 是否精酿 | `craft brewed or not` |
| `rel_0021` | 食品工艺 | `food processing method` |
| `rel_0070` | 口味 | `flavor` |
| `rel_0124` | 食品口味 | `food flavor` |
| `rel_0067` | 香味 | `fragrance` |
| `rel_0095` | 电源方式 | `power supply` |
| `rel_0111` | 功率 | `power` |

### 4.3 ambiguous_material_relations

These relations are intentionally excluded from the first main comparison because their
visual observability is category-sensitive and unreliable.

| Relation ID | Chinese | English |
|---|---|---|
| `rel_0082` | 材质 | `material` |
| `rel_0100` | 材质 | `material` |
| `rel_0003` | 面料 | `fabric` |
| `rel_0073` | 面料材质 | `fabric material` |
| `rel_0069` | 帮面材质 | `upper material` |
| `rel_0029` | 鞋面材质 | `shoe upper material` |
| `rel_0102` | 鞋底材质 | `sole material` |
| `rel_0035` | 里料材质 | `lining material` |
| `rel_0059` | 内胆材质 | `lining material` |
| `rel_0074` | 里料 | `lining` |
| `rel_0119` | 填充料 | `filling material` |
| `rel_0092` | 薄厚 | `thickness` |
| `rel_0083` | 质地 | `texture` |
| `rel_0046` | 服饰工艺 | `clothing craftsmanship` |
| `rel_0099` | 材质工艺 | `material process` |

## 5. Notes

- `visual_relations` should be interpreted as "more likely visually grounded", not
  "always perfectly recoverable from images".
- `weak_visual_relations` does not mean images are useless. It means image evidence is
  usually indirect or secondary.
- `ambiguous_material_relations` are likely worth revisiting in a second round after the
  main relation-type result is established.

## 6. Next Step

The next implementation step is to create a relation-type evaluation script that:

1. loads this relation grouping,
2. slices the `paper_split` test triples by relation id,
3. runs grouped filtered-ranking evaluation for selected models,
4. outputs a markdown summary and machine-readable JSON.
