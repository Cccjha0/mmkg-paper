import torch

from ml.training.src.data.tsv_reader import read_allow_2or3


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

    if model_name == "openbg_img_gated":
        from ml.training.src.models.openbg_img_gated_lp import OpenBGImgGatedLP

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

        text_feat, img_feat, has_img = _load_openbg_img_features(cache_dir, cache_format)

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

    if model_name == "text_rgcn":
        from ml.training.src.models.text.text_rgcn import TextRGCN

        mcfg = cfg["model"]
        train_path = cfg["dataset"]["train"]
        cache_dir = cfg["dataset"].get("cache_dir", "")
        ent_emb_file = mcfg.get("entity_emb_file", "entity_bert_emb.pt")
        init_ent_emb = None
        num_entities = mcfg.get("num_entities")
        if cache_dir:
            emb_path = f"{cache_dir}/{ent_emb_file}"
            try:
                init_ent_emb = torch.load(emb_path, map_location="cpu").float()
                print(f"[BuildModel] loaded text entity embeddings from: {emb_path}")
                if num_entities is not None and int(num_entities) != int(init_ent_emb.shape[0]):
                    print(
                        "[BuildModel] WARN: num_entities in config does not match text cache; "
                        f"using cache size {init_ent_emb.shape[0]} instead of {num_entities}"
                    )
                num_entities = int(init_ent_emb.shape[0])
            except FileNotFoundError:
                print(f"[BuildModel] WARN: text cache not found, random init: {emb_path}")
        if num_entities is None:
            raise RuntimeError("text_rgcn requires model.num_entities or a valid entity embedding cache.")

        train_triples, _, bad_train = read_allow_2or3(train_path)
        if bad_train:
            print(f"[BuildModel] WARN: malformed train lines skipped for text_rgcn: {bad_train}")
        if len(train_triples) == 0:
            raise RuntimeError("text_rgcn requires non-empty 3-column train triples to build the graph.")

        src = torch.tensor([h for h, _, _ in train_triples], dtype=torch.long)
        rel = torch.tensor([r for _, r, _ in train_triples], dtype=torch.long)
        dst = torch.tensor([t for _, _, t in train_triples], dtype=torch.long)
        edge_src = torch.cat([src, dst], dim=0)
        edge_dst = torch.cat([dst, src], dim=0)
        edge_type = torch.cat([rel, rel + mcfg["num_relations"]], dim=0)

        model = TextRGCN(
            num_entities=num_entities,
            num_relations=mcfg["num_relations"],
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_type=edge_type,
            text_emb_dim=mcfg.get("text_emb_dim", 256),
            hidden_dim=mcfg.get("hidden_dim", 64),
            num_layers=mcfg.get("num_layers", 2),
            num_bases=mcfg.get("num_bases", 8),
            sample_neighbors=mcfg.get("sample_neighbors", 10),
            eval_on_cpu=mcfg.get("eval_on_cpu", False),
            dropout=mcfg.get("dropout", 0.1),
            init_ent_emb=init_ent_emb,
        )
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
