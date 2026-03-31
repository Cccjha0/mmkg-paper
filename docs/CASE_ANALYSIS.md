# Case Analysis

## 1. Purpose

This document extracts concrete success and failure cases from the completed paper-stage models.

Current focus:

- success cases where `Full Model` consistently beats `Residual-only`
- failure cases where `Residual-only` consistently beats `Full Model`
- case-level links to image availability, relation group, gate behavior, and residual/fused balance

## 2. Setup

- Outputs root: `ml/artifacts/outputs`
- Grouping source: `docs/relation_type_groups_draft.json`
- Compared models: `Full Model, Residual-only`
- Common seeds: `[1, 2, 3]`
- Candidate query directions: `both`
- Selected success cases: `6`
- Selected failure cases: `6`

## 3. Success Cases

### Success 1

- Query: `(ent_015357, rel_0010, ent_005039)`
- Text: `中老年女装` --`品牌`--> `梵忆轩`
- Direction: `head` | Relation group: `weak_visual_relations`
- Target: `ent_015357` / `中老年女装` | `target_has_img=True`
- Cue hint: mixed cue regime
- Full Model mean rank / MRR: `1.33` / `0.8333`
- Residual-only mean rank / MRR: `469.67` / `0.0022`
- Delta MRR (Full - Residual): `0.8312`
- Full gate mean: `0.3567`
- Full effective residual / fused: `0.3608` / `5.9157`
- Full residual-to-fused ratio: `0.0610`

Full Model filtered top-5:
- `ent_015357` / `中老年女装` | score `2.9580` | GOLD
- `ent_007820` / `春娟黄芪精粹活能精华液` | score `1.8025`
- `ent_015279` / `中老年女装` | score `1.7806`
- `ent_009814` / `洋河蓝色经典天之蓝52度` | score `1.6476`
- `ent_012106` / `阿芙玫瑰花瓣精华水` | score `1.5558`

Residual-only filtered top-5:
- `ent_024300` / `雪纺衫` | score `0.4408`
- `ent_027127` / `T恤` | score `0.4238`
- `ent_006145` / `连衣裙` | score `0.4192`
- `ent_016113` / `连衣裙` | score `0.4136`
- `ent_026963` / `连衣裙` | score `0.4106`

### Success 2

- Query: `(ent_008826, rel_0098, ent_026731)`
- Text: `时尚凉鞋` --`佩戴方式`--> `魔术贴`
- Direction: `head` | Relation group: `visual_relations`
- Target: `ent_008826` / `时尚凉鞋` | `target_has_img=True`
- Cue hint: likely multimodal-favorable: head target has image and relation is visually grounded
- Full Model mean rank / MRR: `7.33` / `0.4048`
- Residual-only mean rank / MRR: `54.00` / `0.0279`
- Delta MRR (Full - Residual): `0.3768`
- Full gate mean: `0.3687`
- Full effective residual / fused: `0.4245` / `6.1740`
- Full residual-to-fused ratio: `0.0688`

Full Model filtered top-5:
- `ent_008826` / `时尚凉鞋` | score `2.7736` | GOLD
- `ent_021835` / `时尚凉鞋` | score `2.6401`
- `ent_010878` / `时尚凉鞋` | score `2.5254`
- `ent_006889` / `时尚凉鞋` | score `2.5233`
- `ent_025125` / `儿童沙滩凉鞋` | score `2.5012`

Residual-only filtered top-5:
- `ent_003254` / `成人` | score `0.2686`
- `ent_017568` / `休闲鞋` | score `0.2094`
- `ent_026845` / `长袖针织衫` | score `0.2052`
- `ent_012965` / `短袖T恤` | score `0.2027`
- `ent_008349` / `女童t恤` | score `0.2023`

### Success 3

- Query: `(ent_013942, rel_0005, ent_001388)`
- Text: `时尚凉鞋` --`地市`--> `温州市`
- Direction: `head` | Relation group: `weak_visual_relations`
- Target: `ent_013942` / `时尚凉鞋` | `target_has_img=True`
- Cue hint: mixed cue regime
- Full Model mean rank / MRR: `5.33` / `0.2194`
- Residual-only mean rank / MRR: `254.00` / `0.0040`
- Delta MRR (Full - Residual): `0.2155`
- Full gate mean: `0.3549`
- Full effective residual / fused: `0.3167` / `6.0697`
- Full residual-to-fused ratio: `0.0522`

Full Model filtered top-5:
- `ent_009089` / `时尚凉鞋` | score `3.0668`
- `ent_022808` / `时尚凉鞋` | score `3.0212`
- `ent_021875` / `时尚凉鞋` | score `2.9278`
- `ent_025980` / `时尚凉鞋` | score `2.8866`
- `ent_014482` / `时尚凉鞋` | score `2.8843`

Residual-only filtered top-5:
- `ent_027616` / `针织运动长裤` | score `0.1675`
- `ent_010987` / `童鞋` | score `0.1595`
- `ent_026845` / `长袖针织衫` | score `0.1588`
- `ent_012964` / `休闲鞋` | score `0.1579`
- `ent_012424` / `雪地靴` | score `0.1568`

### Success 4

- Query: `(ent_025297, rel_0062, ent_007273)`
- Text: `短裙` --`裙长`--> `短裙`
- Direction: `head` | Relation group: `visual_relations`
- Target: `ent_025297` / `短裙` | `target_has_img=True`
- Cue hint: likely multimodal-favorable: head target has image and relation is visually grounded
- Full Model mean rank / MRR: `14.00` / `0.1483`
- Residual-only mean rank / MRR: `486.00` / `0.0021`
- Delta MRR (Full - Residual): `0.1462`
- Full gate mean: `0.3116`
- Full effective residual / fused: `0.4719` / `6.0926`
- Full residual-to-fused ratio: `0.0775`

Full Model filtered top-5:
- `ent_003680` / `半身裙` | score `3.1542`
- `ent_026342` / `短裙` | score `3.0631`
- `ent_022922` / `女士半裙` | score `3.0075`
- `ent_017852` / `短裙` | score `2.9946`
- `ent_002871` / `女士半裙` | score `2.7792`

Residual-only filtered top-5:
- `ent_024300` / `雪纺衫` | score `0.7947`
- `ent_006145` / `连衣裙` | score `0.7814`
- `ent_027127` / `T恤` | score `0.7740`
- `ent_016113` / `连衣裙` | score `0.7717`
- `ent_019432` / `连衣裙` | score `0.7603`

### Success 5

- Query: `(ent_002283, rel_0022, ent_025396)`
- Text: `单鞋` --`是否精酿`--> `胶粘鞋`
- Direction: `head` | Relation group: `weak_visual_relations`
- Target: `ent_002283` / `单鞋` | `target_has_img=True`
- Cue hint: mixed cue regime
- Full Model mean rank / MRR: `16.00` / `0.1020`
- Residual-only mean rank / MRR: `132.00` / `0.0100`
- Delta MRR (Full - Residual): `0.0919`
- Full gate mean: `0.3622`
- Full effective residual / fused: `0.3179` / `6.2010`
- Full residual-to-fused ratio: `0.0513`

Full Model filtered top-5:
- `ent_020691` / `单鞋` | score `3.2202`
- `ent_008314` / `单鞋` | score `3.0295`
- `ent_012399` / `单鞋` | score `2.9917`
- `ent_017302` / `单鞋` | score `2.9884`
- `ent_002283` / `单鞋` | score `2.9667` | GOLD

Residual-only filtered top-5:
- `ent_011400` / `男士短袖T恤` | score `0.5073`
- `ent_006979` / `梭织短裤` | score `0.4932`
- `ent_019481` / `长袖针织上衣` | score `0.4895`
- `ent_025091` / `男士牛仔裤` | score `0.4861`
- `ent_014520` / `短袖t恤` | score `0.4827`

### Success 6

- Query: `(ent_003729, rel_0064, ent_010666)`
- Text: `短袖衬衫` --`细分风格`--> `潮`
- Direction: `head` | Relation group: `visual_relations`
- Target: `ent_003729` / `短袖衬衫` | `target_has_img=True`
- Cue hint: likely multimodal-favorable: head target has image and relation is visually grounded
- Full Model mean rank / MRR: `17.67` / `0.0593`
- Residual-only mean rank / MRR: `477.33` / `0.0023`
- Delta MRR (Full - Residual): `0.0570`
- Full gate mean: `0.3172`
- Full effective residual / fused: `0.3966` / `5.8178`
- Full residual-to-fused ratio: `0.0682`

Full Model filtered top-5:
- `ent_006799` / `短袖针织上衣` | score `3.0937`
- `ent_013905` / `男士T恤` | score `2.9833`
- `ent_024740` / `男士衬衫` | score `2.9044`
- `ent_007638` / `长袖圆领T恤` | score `2.7618`
- `ent_003604` / `男士T恤` | score `2.7364`

Residual-only filtered top-5:
- `ent_011400` / `男士短袖T恤` | score `0.8525`
- `ent_019481` / `长袖针织上衣` | score `0.7836`
- `ent_000836` / `T恤` | score `0.7676`
- `ent_011394` / `T恤衫` | score `0.7587`
- `ent_010268` / `连衣裙` | score `0.7559`

## 4. Failure Cases

### Failure 1

- Query: `(ent_018985, rel_0031, ent_005725)`
- Text: `POLO衫` --`适用场景`--> `休闲`
- Direction: `tail` | Relation group: `weak_visual_relations`
- Target: `ent_005725` / `休闲` | `target_has_img=False`
- Cue hint: likely structure-favorable under current split: tail target is typically no-image
- Full Model mean rank / MRR: `1686.33` / `0.0006`
- Residual-only mean rank / MRR: `1.00` / `1.0000`
- Delta MRR (Full - Residual): `-0.9994`
- Full gate mean: `0.6201`
- Full effective residual / fused: `1.1239` / `6.5904`
- Full residual-to-fused ratio: `0.1705`

Full Model filtered top-5:
- `ent_017944` / `居家` | score `-0.5483`
- `ent_022780` / `演出` | score `-0.5495`
- `ent_027503` / `演出` | score `-0.6225`
- `ent_005859` / `家人` | score `-0.8116`
- `ent_001531` / `Great Family/歌瑞家` | score `-0.8384`

Residual-only filtered top-5:
- `ent_005725` / `休闲` | score `0.5219` | GOLD
- `ent_009093` / `日常` | score `0.3810`
- `ent_015833` / `设计` | score `0.3787`
- `ent_017656` / `运动` | score `0.3701`
- `ent_011282` / `商务` | score `0.2388`

### Failure 2

- Query: `(ent_019724, rel_0082, ent_016246)`
- Text: `背心` --`材质`--> `棉`
- Direction: `tail` | Relation group: `ambiguous_material_relations`
- Target: `ent_016246` / `棉` | `target_has_img=False`
- Cue hint: likely structure-favorable under current split: tail target is typically no-image
- Full Model mean rank / MRR: `381.33` / `0.0031`
- Residual-only mean rank / MRR: `1.00` / `1.0000`
- Delta MRR (Full - Residual): `-0.9969`
- Full gate mean: `0.5895`
- Full effective residual / fused: `1.0033` / `6.3548`
- Full residual-to-fused ratio: `0.1579`

Full Model filtered top-5:
- `ent_017502` / `粘纤` | score `1.2632`
- `ent_017130` / `莫代尔` | score `1.0025`
- `ent_014259` / `网纱` | score `0.9871`
- `ent_018413` / `化纤` | score `0.8658`
- `ent_004385` / `粘胶纤维(粘纤)` | score `0.7133`

Residual-only filtered top-5:
- `ent_016246` / `棉` | score `0.6396` | GOLD
- `ent_000799` / `涤纶` | score `0.5975`
- `ent_014086` / `锦纶` | score `0.5524`
- `ent_018835` / `氨纶` | score `0.5290`
- `ent_014842` / `聚酯` | score `0.4859`

### Failure 3

- Query: `(ent_024463, rel_0009, ent_014175)`
- Text: `套头卫衣` --`裤长`--> `长裤`
- Direction: `tail` | Relation group: `visual_relations`
- Target: `ent_014175` / `长裤` | `target_has_img=False`
- Cue hint: likely structure-favorable under current split: tail target is typically no-image
- Full Model mean rank / MRR: `764.67` / `0.0032`
- Residual-only mean rank / MRR: `1.00` / `1.0000`
- Delta MRR (Full - Residual): `-0.9968`
- Full gate mean: `0.4047`
- Full effective residual / fused: `1.2199` / `6.0386`
- Full residual-to-fused ratio: `0.2020`

Full Model filtered top-5:
- `ent_007492` / `普通装` | score `0.4513`
- `ent_007314` / `否` | score `0.1894`
- `ent_017802` / `蝙蝠袖` | score `0.1762`
- `ent_014799` / `贴布` | score `0.1532`
- `ent_027378` / `套头` | score `0.1361`

Residual-only filtered top-5:
- `ent_014175` / `长裤` | score `0.3970` | GOLD
- `ent_001501` / `短裤` | score `0.3286`
- `ent_021652` / `五分裤` | score `0.3141`
- `ent_011539` / `九分裤` | score `0.3110`
- `ent_006249` / `七分裤` | score `0.2567`

### Failure 4

- Query: `(ent_015204, rel_0115, ent_007492)`
- Text: `咸蛋黄麦芽饼` --`净含量`--> `普通装`
- Direction: `tail` | Relation group: `weak_visual_relations`
- Target: `ent_007492` / `普通装` | `target_has_img=False`
- Cue hint: likely structure-favorable under current split: tail target is typically no-image
- Full Model mean rank / MRR: `236.00` / `0.0044`
- Residual-only mean rank / MRR: `1.00` / `1.0000`
- Delta MRR (Full - Residual): `-0.9956`
- Full gate mean: `0.5918`
- Full effective residual / fused: `1.2111` / `6.4273`
- Full residual-to-fused ratio: `0.1884`

Full Model filtered top-5:
- `ent_012692` / `100克` | score `2.1064`
- `ent_020126` / `150克` | score `1.8552`
- `ent_021006` / `1000克` | score `1.7865`
- `ent_008213` / `130克` | score `1.7738`
- `ent_007780` / `70克` | score `1.6764`

Residual-only filtered top-5:
- `ent_007492` / `普通装` | score `0.1126` | GOLD
- `ent_007242` / `青年` | score `0.0597`
- `ent_003254` / `成人` | score `0.0525`
- `ent_015636` / `女` | score `0.0469`
- `ent_012566` / `短袖` | score `0.0461`

### Failure 5

- Query: `(ent_017155, rel_0008, ent_017020)`
- Text: `文胸` --`色泽`--> `黑色`
- Direction: `tail` | Relation group: `visual_relations`
- Target: `ent_017020` / `黑色` | `target_has_img=False`
- Cue hint: likely structure-favorable under current split: tail target is typically no-image
- Full Model mean rank / MRR: `278.33` / `0.0056`
- Residual-only mean rank / MRR: `1.00` / `1.0000`
- Delta MRR (Full - Residual): `-0.9944`
- Full gate mean: `0.6305`
- Full effective residual / fused: `1.1605` / `6.4278`
- Full residual-to-fused ratio: `0.1805`

Full Model filtered top-5:
- `ent_026557` / `肤色` | score `1.5481`
- `ent_001670` / `灰紫` | score `1.1578`
- `ent_015360` / `粉色` | score `0.8618`
- `ent_020121` / `紫灰` | score `0.7856`
- `ent_020404` / `象牙色` | score `0.7174`

Residual-only filtered top-5:
- `ent_017020` / `黑色` | score `0.5219` | GOLD
- `ent_001693` / `白色` | score `0.4799`
- `ent_005478` / `蓝色` | score `0.4487`
- `ent_002030` / `灰色` | score `0.4278`
- `ent_018850` / `红色` | score `0.4225`

### Failure 6

- Query: `(ent_001152, rel_0020, ent_000899)`
- Text: `真丝香云纱上衣` --`设计元素`--> `纯色`
- Direction: `tail` | Relation group: `visual_relations`
- Target: `ent_000899` / `纯色` | `target_has_img=False`
- Cue hint: likely structure-favorable under current split: tail target is typically no-image
- Full Model mean rank / MRR: `156.33` / `0.0086`
- Residual-only mean rank / MRR: `1.00` / `1.0000`
- Delta MRR (Full - Residual): `-0.9914`
- Full gate mean: `0.6408`
- Full effective residual / fused: `1.2198` / `6.4377`
- Full residual-to-fused ratio: `0.1895`

Full Model filtered top-5:
- `ent_004138` / `连帽` | score `-0.3103`
- `ent_014344` / `长袖` | score `-0.6774`
- `ent_018433` / `H型` | score `-0.6821`
- `ent_007931` / `露背` | score `-0.7311`
- `ent_025115` / `线帽` | score `-0.7332`

Residual-only filtered top-5:
- `ent_000899` / `纯色` | score `0.4547` | GOLD
- `ent_016658` / `字母` | score `0.4183`
- `ent_002284` / `印花` | score `0.4131`
- `ent_017844` / `条纹` | score `0.3829`
- `ent_020092` / `拼接` | score `0.3391`

## 5. Takeaways

- Success cases should be read as sample-level evidence for when multimodal fusion adds useful signal beyond pure residual compensation.
- Failure cases should be read as sample-level evidence for when structure compensation remains the safer and stronger route.
- These cases complement the completed `6` and `7` stages by grounding the gain-boundary narrative in concrete triples.
