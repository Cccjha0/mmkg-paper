"""
MM-1: Early Fusion 多模态链接预测模型
- 图像编码: CLIP-ViT (缓存 img_emb)
- 文本编码: CLIP-text (缓存 text_emb)
- 融合: concat + 线性投影到 d 维
- 后端: 与 gated 一致，使用 ComplEx (KGE)
- Trainer API: forward(pos_triples, neg_triples) -> loss; score(triples) -> scores
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.src.models.decoders.complex import ComplEx


class EarlyFusion(nn.Module):
    """Early Fusion: concat(text, image) -> linear projection -> d 维实体表示"""

    def __init__(self, d: int, use_layernorm: bool = True):
        super().__init__()
        self.d = d
        self.use_ln = use_layernorm
        if use_layernorm:
            self.ln_t = nn.LayerNorm(d)
            self.ln_v = nn.LayerNorm(d)
        self.proj = nn.Linear(2 * d, d)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        t: [B, d], v: [B, d]
        returns: z [B, d]
        """
        if self.use_ln:
            t = self.ln_t(t)
            v = self.ln_v(v)
        z = self.proj(torch.cat([t, v], dim=-1))
        return z


class OpenBGImgEarlyLP(nn.Module):
    """
    OpenBG-IMG 链接预测 — MM-1 Early Fusion 版本
    后端与 gated 一致：ComplEx。接口与 TrainerYAML 兼容。
    """

    def __init__(
        self,
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        has_img: torch.Tensor,
        num_relations: int,
        d: int = 256,
        use_layernorm: bool = True,
        neg_ratio: int = 10,
        adv_temperature: float = 1.0,
        img_dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.num_relations = num_relations
        self.neg_ratio = neg_ratio
        self.adv_temperature = adv_temperature
        self.img_dropout = float(img_dropout)
        text_in_dim = int(text_feat.shape[1])
        img_in_dim = int(img_feat.shape[1])
        num_entities = text_feat.shape[0]

        self.register_buffer("text_feat", text_feat)
        self.register_buffer("img_feat", img_feat)
        self.register_buffer("has_img", has_img)

        self.text_proj = nn.Identity() if text_in_dim == d else nn.Linear(text_in_dim, d)
        self.img_proj = nn.Identity() if img_in_dim == d else nn.Linear(img_in_dim, d)

        self.v_missing = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.v_missing, mean=0.0, std=0.02)

        self.fusion = EarlyFusion(d=d, use_layernorm=use_layernorm)
        self.decoder = ComplEx(num_relations=num_relations, d=d)

    def _entity_text(self, eids: torch.Tensor) -> torch.Tensor:
        return self.text_proj(self.text_feat[eids])

    def _entity_image(self, eids: torch.Tensor) -> torch.Tensor:
        v = self.img_proj(self.img_feat[eids])
        has_img = self.has_img[eids]
        mask = has_img.unsqueeze(-1)
        v = torch.where(mask, v, self.v_missing.unsqueeze(0).expand_as(v))
        if self.training and self.img_dropout > 0:
            drop_mask = (torch.rand(eids.size(0), device=eids.device) < self.img_dropout) & has_img
            if drop_mask.any():
                v = v.clone()
                v[drop_mask] = self.v_missing
        return v

    def _fused(self, eids: torch.LongTensor) -> torch.Tensor:
        t = self._entity_text(eids)
        v = self._entity_image(eids)
        return self.fusion(t, v)

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        h = triples[:, 0]
        r = triples[:, 1]
        t = triples[:, 2]
        zh = self._fused(h)
        zt = self._fused(t)
        return self.decoder.score(zh, r, zt)

    @torch.no_grad()
    def score_eval(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score(triples)

    def self_adversarial_loss(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
        B = pos_logits.size(0)
        neg = neg_logits.view(B, self.neg_ratio)
        pos_loss = F.softplus(-pos_logits)
        with torch.no_grad():
            w = F.softmax(self.adv_temperature * neg, dim=1)
        neg_loss = (w * F.softplus(neg)).sum(dim=1)
        return (pos_loss + neg_loss).mean()

    def forward(self, pos_triples: torch.LongTensor, neg_triples: torch.LongTensor) -> torch.Tensor:
        pos_scores = self.score(pos_triples)
        neg_scores = self.score(neg_triples)
        return self.self_adversarial_loss(pos_scores, neg_scores)
