import torch


def _load_cache_tensor(cache_dir: str, candidates: list[str]) -> tuple[torch.Tensor, str]:
    for name in candidates:
        path = f"{cache_dir}/{name}"
        try:
            return torch.load(path, map_location="cpu"), name
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No cache file found in {cache_dir} for candidates: {candidates}")


def _load_openbg_img_features(cache_dir: str, cache_format: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_format = (cache_format or "legacy").lower()

    if cache_format == "legacy":
        text_feat, _ = _load_cache_tensor(cache_dir, ["text_emb.pt"])
        img_feat, _ = _load_cache_tensor(cache_dir, ["img_emb.pt"])
    elif cache_format == "raw":
        text_feat, _ = _load_cache_tensor(cache_dir, ["text_feat_raw.pt"])
        img_feat, _ = _load_cache_tensor(cache_dir, ["img_feat_raw.pt", "img_emb_raw.pt"])
    elif cache_format == "auto":
        try:
            text_feat, text_name = _load_cache_tensor(cache_dir, ["text_feat_raw.pt"])
            img_feat, img_name = _load_cache_tensor(cache_dir, ["img_feat_raw.pt", "img_emb_raw.pt"])
            print(f"[BuildModel] using raw caches: text={text_name}, image={img_name}")
        except FileNotFoundError:
            text_feat, _ = _load_cache_tensor(cache_dir, ["text_emb.pt"])
            img_feat, _ = _load_cache_tensor(cache_dir, ["img_emb.pt"])
            print("[BuildModel] raw caches unavailable, falling back to legacy projected caches")
    else:
        raise ValueError(f"Unsupported dataset.cache_format: {cache_format}")

    has_img, _ = _load_cache_tensor(cache_dir, ["has_img.pt"])
    return text_feat.float(), img_feat.float(), has_img


def build_model(cfg: dict):
    model_name = cfg["model"]["name"]

    if model_name == "openbg_img_tucker":
        from ml.training.src.models.structure_baselines import StructureTuckERLP

        cache_dir = cfg["dataset"]["cache_dir"]
        cache_format = cfg["dataset"].get("cache_format", "auto")
        d = cfg["embedding"]["d"]
        tr = cfg["training"]
        num_relations = cfg["model"]["num_relations"]
        neg_ratio = tr.get("neg_ratio", 10)
        adv_temperature = tr.get("adv_temperature", 1.0)
        entity_l2_weight = tr.get("entity_l2_weight", 1e-6)
        relation_l2_weight = tr.get("relation_l2_weight", 1e-6)
        core_l2_weight = tr.get("core_l2_weight", 1e-6)

        text_feat, _, _ = _load_openbg_img_features(cache_dir, cache_format)
        num_entities = text_feat.shape[0]

        print("[BuildModel] building explicit model: TuckER")
        model = StructureTuckERLP(
            num_entities=num_entities,
            num_relations=num_relations,
            d=d,
            neg_ratio=neg_ratio,
            adv_temperature=adv_temperature,
            entity_l2_weight=entity_l2_weight,
            relation_l2_weight=relation_l2_weight,
            core_l2_weight=core_l2_weight,
        )
        return model, num_entities

    if model_name == "openbg_img_complex":
        from ml.training.src.models.structure_baselines import StructureComplExLP

        cache_dir = cfg["dataset"]["cache_dir"]
        cache_format = cfg["dataset"].get("cache_format", "auto")
        d = cfg["embedding"]["d"]
        tr = cfg["training"]
        num_relations = cfg["model"]["num_relations"]
        neg_ratio = tr.get("neg_ratio", 10)
        adv_temperature = tr.get("adv_temperature", 1.0)
        entity_l2_weight = tr.get("entity_l2_weight", 1e-6)

        text_feat, _, _ = _load_openbg_img_features(cache_dir, cache_format)
        num_entities = text_feat.shape[0]

        print("[BuildModel] building explicit model: ComplEx")
        model = StructureComplExLP(
            num_entities=num_entities,
            num_relations=num_relations,
            d=d,
            neg_ratio=neg_ratio,
            adv_temperature=adv_temperature,
            entity_l2_weight=entity_l2_weight,
        )
        return model, num_entities

    if model_name in {
        "openbg_img_text_only",
        "openbg_img_gated",
        "openbg_img_gate_only",
        "openbg_img_residual_only",
        "openbg_img_gate_residual",
    }:
        from ml.training.src.models.openbg_img_gated_lp import (
            OpenBGImgGateOnlyLP,
            OpenBGImgGateResidualLP,
            OpenBGImgGatedLP,
            OpenBGImgResidualOnlyLP,
            OpenBGImgTextOnlyLP,
        )
        cache_dir = cfg["dataset"]["cache_dir"]
        cache_format = cfg["dataset"].get("cache_format", "legacy")
        d = cfg["embedding"]["d"]
        tr = cfg["training"]
        num_relations = cfg["model"]["num_relations"]
        use_layernorm = cfg["model"].get("use_layernorm", True)
        use_fusion = cfg["model"].get("use_fusion", True)
        use_residual = cfg["model"].get("use_residual", True)
        use_normalized_mix = cfg["model"].get("use_normalized_mix", False)
        neg_ratio = tr.get("neg_ratio", 10)
        adv_temperature = tr.get("adv_temperature", 1.0)
        img_dropout = tr.get("img_dropout", 0.0)
        gate_reg_weight = tr.get("gate_reg_weight", 1e-3)
        gate_reg_target = tr.get("gate_reg_target", 0.5)
        residual_scale_init = tr.get("residual_scale_init", -2.0)
        residual_l2_weight = tr.get("residual_l2_weight", 1e-6)
        residual_scale_l2_weight = tr.get("residual_scale_l2_weight", 1e-4)

        text_feat, img_feat, has_img = _load_openbg_img_features(cache_dir, cache_format)

        if model_name == "openbg_img_text_only":
            print("[BuildModel] building explicit model: Text-Only")
            model = OpenBGImgTextOnlyLP(
                text_feat=text_feat,
                img_feat=img_feat,
                has_img=has_img,
                num_relations=num_relations,
                d=d,
                neg_ratio=neg_ratio,
                adv_temperature=adv_temperature,
                img_dropout=img_dropout,
            )
        elif model_name == "openbg_img_gate_only":
            print("[BuildModel] building explicit model: Gate-Only")
            model = OpenBGImgGateOnlyLP(
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
        elif model_name == "openbg_img_residual_only":
            print("[BuildModel] building explicit model: Residual-Only")
            model = OpenBGImgResidualOnlyLP(
                text_feat=text_feat,
                img_feat=img_feat,
                has_img=has_img,
                num_relations=num_relations,
                d=d,
                neg_ratio=neg_ratio,
                adv_temperature=adv_temperature,
                img_dropout=img_dropout,
                residual_scale_init=residual_scale_init,
                residual_l2_weight=residual_l2_weight,
                residual_scale_l2_weight=residual_scale_l2_weight,
            )
        elif model_name == "openbg_img_gate_residual":
            print("[BuildModel] building explicit model: Gate+Residual")
            model = OpenBGImgGateResidualLP(
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
                residual_scale_init=residual_scale_init,
                residual_l2_weight=residual_l2_weight,
                residual_scale_l2_weight=residual_scale_l2_weight,
            )
        else:
            print(
                "[BuildModel] building legacy openbg_img_gated via compatibility path: "
                f"use_fusion={use_fusion}, use_residual={use_residual}, "
                f"use_normalized_mix={use_normalized_mix}"
            )
            model = OpenBGImgGatedLP(
                text_feat=text_feat,
                img_feat=img_feat,
                has_img=has_img,
                num_relations=num_relations,
                d=d,
                use_layernorm=use_layernorm,
                neg_ratio=neg_ratio,
                adv_temperature=adv_temperature,
                img_dropout=img_dropout,
                use_fusion=use_fusion,
                use_residual=use_residual,
                use_normalized_mix=use_normalized_mix,
                gate_reg_weight=gate_reg_weight,
                gate_reg_target=gate_reg_target,
                residual_scale_init=residual_scale_init,
                residual_l2_weight=residual_l2_weight,
                residual_scale_l2_weight=residual_scale_l2_weight,
            )
        num_entities = text_feat.shape[0]
        return model, num_entities

    if model_name == "openbg_img_early":
        from ml.training.src.models.fusion.early import OpenBGImgEarlyLP

        cache_dir = cfg["dataset"]["cache_dir"]
        cache_format = cfg["dataset"].get("cache_format", "legacy")
        d = cfg["embedding"]["d"]
        tr = cfg["training"]
        num_relations = cfg["model"]["num_relations"]
        use_layernorm = cfg["model"].get("use_layernorm", True)
        neg_ratio = tr.get("neg_ratio", 10)
        adv_temperature = tr.get("adv_temperature", 1.0)
        img_dropout = tr.get("img_dropout", 0.0)

        text_feat, img_feat, has_img = _load_openbg_img_features(cache_dir, cache_format)

        print("[BuildModel] building explicit model: Early Fusion")
        model = OpenBGImgEarlyLP(
            text_feat=text_feat,
            img_feat=img_feat,
            has_img=has_img,
            num_relations=num_relations,
            d=d,
            use_layernorm=use_layernorm,
            neg_ratio=neg_ratio,
            adv_temperature=adv_temperature,
            img_dropout=img_dropout,
        )
        num_entities = text_feat.shape[0]
        return model, num_entities

    if model_name == "text_complex":
        from ml.training.src.models.text.text_complex import TextComplEx

        mcfg = cfg["model"]
        bert_cache_path = mcfg["bert_cache_path"]
        num_relations = mcfg.get("num_relations", 511)

        print(f"[BuildModel] loading BERT embeddings from: {bert_cache_path}")
        temp_emb = torch.load(bert_cache_path, map_location="cpu")
        num_entities = temp_emb.shape[0]
        del temp_emb

        model = TextComplEx(
            config=cfg,
            num_entities=num_entities,
            num_relations=num_relations,
            bert_emb_path=bert_cache_path,
        )
        return model, num_entities

    raise ValueError(f"Unknown model.name: {model_name}")
