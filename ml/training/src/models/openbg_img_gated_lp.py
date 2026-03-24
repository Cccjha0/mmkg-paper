import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.src.models.fusion.gated import RelAwareGatedFusion
from ml.training.src.models.decoders.complex import ComplEx


class OpenBGImgGatedLP(nn.Module):
    def __init__(self, text_feat: torch.Tensor, img_feat: torch.Tensor, has_img: torch.Tensor,
                 num_relations: int, d: int = 256, use_layernorm: bool = True,
                 neg_ratio: int = 10, adv_temperature: float = 1.0, img_dropout: float = 0.0,
                 use_fusion: bool = True, use_residual: bool = True,
                 use_normalized_mix: bool = False,
                 gate_reg_weight: float = 1e-3,
                 gate_reg_target: float = 0.5):
        super().__init__()
        self.d = d
        self.num_relations = num_relations
        self.neg_ratio = neg_ratio
        self.adv_temperature = adv_temperature
        self.img_dropout = float(img_dropout)
        self.use_fusion = bool(use_fusion)
        self.use_residual = bool(use_residual)
        self.use_normalized_mix = bool(use_normalized_mix)
        self.gate_reg_weight = float(gate_reg_weight)
        self.gate_reg_target = float(gate_reg_target)
        if not self.use_fusion and not self.use_residual:
            raise ValueError("At least one of use_fusion/use_residual must be True.")
        num_entities = text_feat.shape[0]
        text_in_dim = int(text_feat.shape[1])
        img_in_dim = int(img_feat.shape[1])

        # register cached modality features as buffers (not trainable)
        self.register_buffer("text_feat", text_feat)  # [N,text_in_dim]
        self.register_buffer("img_feat", img_feat)    # [N,img_in_dim]
        self.register_buffer("has_img", has_img)    # [N]

        self.text_proj = nn.Identity() if text_in_dim == d else nn.Linear(text_in_dim, d)
        self.img_proj = nn.Identity() if img_in_dim == d else nn.Linear(img_in_dim, d)

        # missing image token (trainable)
        self.v_missing = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.v_missing, mean=0.0, std=0.02)

        self.entity_residual = nn.Embedding(num_entities, d)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))  # softplus(-2)≈0.127
        nn.init.xavier_uniform_(self.entity_residual.weight)
        # learnable branch weights (used when use_normalized_mix=True)
        # Start with residual-dominant mixing, then let training adjust.
        # softplus(-3) << softplus(0), so initial fusion weight is small.
        self.mix_fusion_raw = nn.Parameter(torch.tensor(-3.0))
        self.mix_residual_raw = nn.Parameter(torch.tensor(0.0))

        self.fusion = RelAwareGatedFusion(d=d, num_relations=num_relations, use_layernorm=use_layernorm)
        self.decoder = ComplEx(num_relations=num_relations, d=d)
        self.t_adapter = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.v_adapter = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))

    def _entity_text(self, eids: torch.Tensor) -> torch.Tensor:
        return self.text_proj(self.text_feat[eids])  # [B,d]

    def _entity_image(self, eids: torch.Tensor) -> torch.Tensor:
        v = self.img_proj(self.img_feat[eids])  # [B,d] or projected from raw cache
        has_img = self.has_img[eids]  # [B] bool
        mask = has_img.unsqueeze(-1)  # [B,1] bool
        # if missing -> v_missing
        v = torch.where(mask, v, self.v_missing.unsqueeze(0).expand_as(v))

        # modality dropout: only drop image modality for entities that actually have images
        if self.training and self.img_dropout > 0:
            drop_mask = (torch.rand(eids.size(0), device=eids.device) < self.img_dropout) & has_img
            if drop_mask.any():
                v = v.clone()
                v[drop_mask] = self.v_missing
        return v

    def _fused_with_r(self, eids: torch.LongTensor, rids: torch.LongTensor):
        t = self._entity_text(eids)
        v = self._entity_image(eids)
        if self.use_fusion:
            t = self.t_adapter(t)
            v = self.v_adapter(v)
            z_fused, g = self.fusion(t, v, rids)
        else:
            z_fused = torch.zeros_like(t)
            g = torch.zeros_like(t)

        if self.use_residual:
            res = self.entity_residual(eids)
            scale = F.softplus(self.residual_scale)  # >0
            z_res = scale * res
            if self.use_fusion and self.use_normalized_mix:
                a = F.softplus(self.mix_fusion_raw)
                b = F.softplus(self.mix_residual_raw)
                denom = a + b + 1e-12
                z = (a / denom) * z_fused + (b / denom) * z_res
            else:
                z = z_fused + z_res
        else:
            z = z_fused
        return z, g

    def _score_with_g(self, triples: torch.LongTensor):
        """
        triples: [B,3] on device
        returns scores: [B], g_h: [B,d], g_t: [B,d]
        (Training needs gradients -> DO NOT use torch.no_grad here)
        """
        h = triples[:, 0]
        r = triples[:, 1]
        t = triples[:, 2]
        zh, gh = self._fused_with_r(h, r)
        zt, gt = self._fused_with_r(t, r)
        return self.decoder.score(zh, r, zt), gh, gt

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        scores, _, _ = self._score_with_g(triples)
        return scores

    @torch.no_grad()
    def gate_for_entities(self, eids: torch.LongTensor) -> torch.Tensor:
        """
        eids: [B] entity ids on device
        return: gate g [B] in [0,1] under random relation ids
        """
        if not self.use_fusion:
            return torch.zeros(eids.size(0), device=eids.device)
        rids = torch.randint(0, self.num_relations, (eids.size(0),), device=eids.device)
        t = self._entity_text(eids)
        v = self._entity_image(eids)
        t = self.t_adapter(t)
        v = self.v_adapter(v)
        _, g = self.fusion(t, v, rids)  # [B,d]
        return g.mean(dim=-1)   # [B]

    @torch.no_grad()
    def score_eval(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score(triples)

    def self_adversarial_loss(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
        """
        pos_logits: [B]
        neg_logits: [B * neg_ratio]
        """
        B = pos_logits.size(0)
        neg = neg_logits.view(B, self.neg_ratio)  # [B,K]

        # -log(sigmoid(pos))
        pos_loss = F.softplus(-pos_logits)

        # softmax(alpha * neg) with detached weights
        with torch.no_grad():
            w = F.softmax(self.adv_temperature * neg, dim=1)  # [B,K]

        # sum_i w_i * -log(sigmoid(-neg_i))
        neg_loss = (w * F.softplus(neg)).sum(dim=1)

        return (pos_loss + neg_loss).mean()

    def forward(self, pos_triples: torch.LongTensor, neg_triples: torch.LongTensor) -> torch.Tensor:
        """
        pos_triples: [B,3], neg_triples: [B*neg_ratio,3]
        """
        pos_scores, pos_gh, pos_gt = self._score_with_g(pos_triples)
        neg_scores, neg_gh, neg_gt = self._score_with_g(neg_triples)

        main_loss = self.self_adversarial_loss(pos_scores, neg_scores)
        gate_reg = torch.zeros((), device=pos_scores.device)
        if self.use_fusion and self.gate_reg_weight > 0:
            g_all = torch.cat([pos_gh, pos_gt, neg_gh, neg_gt], dim=0)  # [*,d]
            g_mean = g_all.mean()
            gate_reg = self.gate_reg_weight * (g_mean - self.gate_reg_target).pow(2)

        if self.use_residual:
            l2 = 1e-6 * self.entity_residual.weight.pow(2).mean()
            scale = F.softplus(self.residual_scale)
            scale_l2 = 1e-4 * scale.pow(2)
            loss = main_loss + l2 + scale_l2 + gate_reg
        else:
            loss = main_loss + gate_reg
        return loss
