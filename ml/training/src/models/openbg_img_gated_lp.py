import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.src.models.decoders.complex import ComplEx
from ml.training.src.models.fusion.gated import RelAwareGatedFusion


class BaseOpenBGImgLP(nn.Module):
    def __init__(
        self,
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        has_img: torch.Tensor,
        num_relations: int,
        d: int = 256,
        neg_ratio: int = 10,
        adv_temperature: float = 1.0,
        img_dropout: float = 0.0,
    ):
        super().__init__()
        self.d = int(d)
        self.num_relations = int(num_relations)
        self.neg_ratio = int(neg_ratio)
        self.adv_temperature = float(adv_temperature)
        self.img_dropout = float(img_dropout)

        text_in_dim = int(text_feat.shape[1])
        img_in_dim = int(img_feat.shape[1])

        self.register_buffer("text_feat", text_feat)
        self.register_buffer("img_feat", img_feat)
        self.register_buffer("has_img", has_img)

        self.text_proj = nn.Identity() if text_in_dim == d else nn.Linear(text_in_dim, d)
        self.img_proj = nn.Identity() if img_in_dim == d else nn.Linear(img_in_dim, d)

        self.v_missing = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.v_missing, mean=0.0, std=0.02)

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

    def _entity_with_relation(self, eids: torch.LongTensor, rids: torch.LongTensor):
        raise NotImplementedError

    def _score_with_aux(self, triples: torch.LongTensor):
        h = triples[:, 0]
        r = triples[:, 1]
        t = triples[:, 2]
        zh, aux_h = self._entity_with_relation(h, r)
        zt, aux_t = self._entity_with_relation(t, r)
        return self.decoder.score(zh, r, zt), aux_h, aux_t

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        scores, _, _ = self._score_with_aux(triples)
        return scores

    @torch.no_grad()
    def score_eval(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score(triples)

    def self_adversarial_loss(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
        bsz = pos_logits.size(0)
        neg = neg_logits.view(bsz, self.neg_ratio)
        pos_loss = F.softplus(-pos_logits)
        with torch.no_grad():
            w = F.softmax(self.adv_temperature * neg, dim=1)
        neg_loss = (w * F.softplus(neg)).sum(dim=1)
        return (pos_loss + neg_loss).mean()

    def extra_loss(self, pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t, device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device)

    def forward(self, pos_triples: torch.LongTensor, neg_triples: torch.LongTensor) -> torch.Tensor:
        pos_scores, pos_aux_h, pos_aux_t = self._score_with_aux(pos_triples)
        neg_scores, neg_aux_h, neg_aux_t = self._score_with_aux(neg_triples)
        main_loss = self.self_adversarial_loss(pos_scores, neg_scores)
        aux_loss = self.extra_loss(pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t, pos_scores.device)
        return main_loss + aux_loss


class OpenBGImgGateOnlyLP(BaseOpenBGImgLP):
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
        gate_reg_weight: float = 1e-3,
        gate_reg_target: float = 0.5,
    ):
        super().__init__(
            text_feat=text_feat,
            img_feat=img_feat,
            has_img=has_img,
            num_relations=num_relations,
            d=d,
            neg_ratio=neg_ratio,
            adv_temperature=adv_temperature,
            img_dropout=img_dropout,
        )
        self.use_fusion = True
        self.use_residual = False
        self.gate_reg_weight = float(gate_reg_weight)
        self.gate_reg_target = float(gate_reg_target)
        self.fusion = RelAwareGatedFusion(d=d, num_relations=num_relations, use_layernorm=use_layernorm)
        self.t_adapter = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.v_adapter = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))

    def _entity_with_relation(self, eids: torch.LongTensor, rids: torch.LongTensor):
        t = self.t_adapter(self._entity_text(eids))
        v = self.v_adapter(self._entity_image(eids))
        z, g = self.fusion(t, v, rids)
        return z, g

    @torch.no_grad()
    def gate_for_entities(self, eids: torch.LongTensor) -> torch.Tensor:
        rids = torch.randint(0, self.num_relations, (eids.size(0),), device=eids.device)
        _, g = self._entity_with_relation(eids, rids)
        return g.mean(dim=-1)

    def extra_loss(self, pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t, device: torch.device) -> torch.Tensor:
        if self.gate_reg_weight <= 0:
            return torch.zeros((), device=device)
        g_all = torch.cat([pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t], dim=0)
        g_mean = g_all.mean()
        return self.gate_reg_weight * (g_mean - self.gate_reg_target).pow(2)


class OpenBGImgTextOnlyLP(BaseOpenBGImgLP):
    def __init__(
        self,
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        has_img: torch.Tensor,
        num_relations: int,
        d: int = 256,
        neg_ratio: int = 10,
        adv_temperature: float = 1.0,
        img_dropout: float = 0.0,
    ):
        super().__init__(
            text_feat=text_feat,
            img_feat=img_feat,
            has_img=has_img,
            num_relations=num_relations,
            d=d,
            neg_ratio=neg_ratio,
            adv_temperature=adv_temperature,
            img_dropout=img_dropout,
        )
        self.use_fusion = False
        self.use_residual = False
        self.t_adapter = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))

    def _entity_with_relation(self, eids: torch.LongTensor, rids: torch.LongTensor):
        del rids
        return self.t_adapter(self._entity_text(eids)), None

    def extra_loss(self, pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t, device: torch.device) -> torch.Tensor:
        del pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t
        return torch.zeros((), device=device)


class OpenBGImgResidualOnlyLP(BaseOpenBGImgLP):
    def __init__(
        self,
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        has_img: torch.Tensor,
        num_relations: int,
        d: int = 256,
        neg_ratio: int = 10,
        adv_temperature: float = 1.0,
        img_dropout: float = 0.0,
    ):
        super().__init__(
            text_feat=text_feat,
            img_feat=img_feat,
            has_img=has_img,
            num_relations=num_relations,
            d=d,
            neg_ratio=neg_ratio,
            adv_temperature=adv_temperature,
            img_dropout=img_dropout,
        )
        self.use_fusion = False
        self.use_residual = True
        num_entities = text_feat.shape[0]
        self.entity_residual = nn.Embedding(num_entities, d)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.xavier_uniform_(self.entity_residual.weight)

    def _entity_with_relation(self, eids: torch.LongTensor, rids: torch.LongTensor):
        del rids
        res = self.entity_residual(eids)
        scale = F.softplus(self.residual_scale)
        return scale * res, None

    def extra_loss(self, pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t, device: torch.device) -> torch.Tensor:
        del pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t, device
        l2 = 1e-6 * self.entity_residual.weight.pow(2).mean()
        scale = F.softplus(self.residual_scale)
        scale_l2 = 1e-4 * scale.pow(2)
        return l2 + scale_l2


class OpenBGImgGateResidualLP(OpenBGImgGateOnlyLP):
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
        use_normalized_mix: bool = False,
        gate_reg_weight: float = 1e-3,
        gate_reg_target: float = 0.5,
    ):
        super().__init__(
            text_feat=text_feat,
            img_feat=img_feat,
            has_img=has_img,
            num_relations=num_relations,
            d=d,
            use_layernorm=use_layernorm,
            neg_ratio=neg_ratio,
            adv_temperature=adv_temperature,
            img_dropout=img_dropout,
            gate_reg_weight=gate_reg_weight,
            gate_reg_target=gate_reg_target,
        )
        self.use_residual = True
        self.enable_residual = True
        self.use_normalized_mix = bool(use_normalized_mix)
        num_entities = text_feat.shape[0]
        self.entity_residual = nn.Embedding(num_entities, d)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.xavier_uniform_(self.entity_residual.weight)
        self.mix_fusion_raw = nn.Parameter(torch.tensor(-3.0))
        self.mix_residual_raw = nn.Parameter(torch.tensor(0.0))

    def _entity_with_relation(self, eids: torch.LongTensor, rids: torch.LongTensor):
        z_fused, g = super()._entity_with_relation(eids, rids)
        if self.enable_residual:
            res = self.entity_residual(eids)
            scale = F.softplus(self.residual_scale)
            z_res = scale * res
        else:
            z_res = torch.zeros_like(z_fused)
        if self.use_normalized_mix:
            a = F.softplus(self.mix_fusion_raw)
            b = F.softplus(self.mix_residual_raw)
            denom = a + b + 1e-12
            z = (a / denom) * z_fused + (b / denom) * z_res
        else:
            z = z_fused + z_res
        return z, g

    def extra_loss(self, pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t, device: torch.device) -> torch.Tensor:
        gate_reg = super().extra_loss(pos_aux_h, pos_aux_t, neg_aux_h, neg_aux_t, device)
        l2 = 1e-6 * self.entity_residual.weight.pow(2).mean()
        scale = F.softplus(self.residual_scale)
        scale_l2 = 1e-4 * scale.pow(2)
        return gate_reg + l2 + scale_l2


class OpenBGImgGatedLP(OpenBGImgGateResidualLP):
    """
    Legacy compatibility wrapper.

    This keeps the previous build path working while the codebase migrates to
    explicit model classes. New code should prefer:
    - OpenBGImgGateOnlyLP
    - OpenBGImgResidualOnlyLP
    - OpenBGImgGateResidualLP
    """

    def __new__(
        cls,
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        has_img: torch.Tensor,
        num_relations: int,
        d: int = 256,
        use_layernorm: bool = True,
        neg_ratio: int = 10,
        adv_temperature: float = 1.0,
        img_dropout: float = 0.0,
        use_fusion: bool = True,
        use_residual: bool = True,
        use_normalized_mix: bool = False,
        gate_reg_weight: float = 1e-3,
        gate_reg_target: float = 0.5,
    ):
        if use_fusion and use_residual:
            return OpenBGImgGateResidualLP(
                text_feat=text_feat,
                img_feat=img_feat,
                has_img=has_img,
                num_relations=num_relations,
                d=d,
                use_layernorm=use_layernorm,
                neg_ratio=neg_ratio,
                adv_temperature=adv_temperature,
                img_dropout=img_dropout,
                use_normalized_mix=use_normalized_mix,
                gate_reg_weight=gate_reg_weight,
                gate_reg_target=gate_reg_target,
            )
        if use_fusion and not use_residual:
            return OpenBGImgGateOnlyLP(
                text_feat=text_feat,
                img_feat=img_feat,
                has_img=has_img,
                num_relations=num_relations,
                d=d,
                use_layernorm=use_layernorm,
                neg_ratio=neg_ratio,
                adv_temperature=adv_temperature,
                img_dropout=img_dropout,
                gate_reg_weight=gate_reg_weight,
                gate_reg_target=gate_reg_target,
            )
        if (not use_fusion) and use_residual:
            return OpenBGImgResidualOnlyLP(
                text_feat=text_feat,
                img_feat=img_feat,
                has_img=has_img,
                num_relations=num_relations,
                d=d,
                neg_ratio=neg_ratio,
                adv_temperature=adv_temperature,
                img_dropout=img_dropout,
            )
        raise ValueError("At least one of use_fusion/use_residual must be True.")
