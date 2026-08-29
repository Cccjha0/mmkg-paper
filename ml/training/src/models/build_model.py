import torch

from ml.training.src.data.dataset_spec import DatasetBundle, MMKG_GENERAL_V1, OPENBG_LEGACY_V1
from ml.training.src.data.feature_bundle import load_openbg_feature_bundle


def _load_cache_tensor(cache_dir: str, candidates: list[str]) -> tuple[torch.Tensor, str]:
    for name in candidates:
        path = f"{cache_dir}/{name}"
        try:
            return torch.load(path, map_location="cpu"), name
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No cache file found in {cache_dir} for candidates: {candidates}")


def _load_openbg_img_features(cache_dir: str, cache_format: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bundle = load_openbg_feature_bundle(cache_dir, cache_format)
    return bundle.text_features, bundle.image_features, bundle.has_img


def build_model(cfg: dict, dataset_bundle: DatasetBundle | None = None):
    requested_model_name = cfg["model"]["name"]
    aliases = {
        "mmkg_mhyper": "openbg_img_mhyper",
        "mmkg_adamf_mat": "openbg_img_adamf_mat",
        "mmkg_native": "openbg_img_native",
        "mmkg_apkgc": "openbg_img_apkgc",
        "mmkg_tucker": "openbg_img_tucker",
        "mmkg_complex": "openbg_img_complex",
        "mmkg_text_only": "openbg_img_text_only",
        "mmkg_gate_only": "openbg_img_gate_only",
        "mmkg_residual_only": "openbg_img_residual_only",
        "mmkg_gate_residual": "openbg_img_gate_residual",
    }
    model_name = aliases.get(requested_model_name, requested_model_name)
    protocol_version = cfg.get("protocol", {}).get("version", OPENBG_LEGACY_V1)
    if requested_model_name.startswith("mmkg_") and protocol_version != MMKG_GENERAL_V1:
        raise ValueError("Generic mmkg_* model names require protocol.version=mmkg_general_v1")
    if dataset_bundle is None and protocol_version == MMKG_GENERAL_V1:
        from ml.training.src.data.dataset_loader import load_dataset_bundle

        dataset_bundle = load_dataset_bundle(cfg)

    def feature_inputs(cache_format: str):
        if dataset_bundle is not None:
            features = dataset_bundle.features
            return features.text_features, features.image_features, features.has_img
        cache_dir = cfg["dataset"].get("cache_dir")
        if not cache_dir:
            raise ValueError("Legacy model construction requires dataset.cache_dir.")
        return _load_openbg_img_features(cache_dir, cache_format)

    def entity_count(cache_format: str = "auto") -> int:
        if dataset_bundle is not None:
            return dataset_bundle.num_entities
        text_feat, _, _ = feature_inputs(cache_format)
        return int(text_feat.shape[0])

    if dataset_bundle is not None:
        configured_relations = int(cfg["model"]["num_relations"])
        if configured_relations != dataset_bundle.num_relations:
            raise ValueError(
                "model.num_relations does not match the dataset manifest: "
                f"{configured_relations} != {dataset_bundle.num_relations}"
            )
    general_has_text = (
        dataset_bundle.features.has_text
        if dataset_bundle is not None and protocol_version == MMKG_GENERAL_V1
        else None
    )

    if model_name == "openbg_img_mhyper":
        from ml.training.src.models.recent_baselines.mhyper import OpenBGMHyper

        cache_dir = cfg["dataset"].get("cache_dir", "")
        cache_format = cfg["dataset"].get("cache_format", "raw")
        text_feat, img_feat, has_img = feature_inputs(cache_format)
        mcfg = cfg["model"]
        tr = cfg["training"]
        num_entities = text_feat.shape[0]
        print("[BuildModel] building recent baseline: M-Hyper")
        model = OpenBGMHyper(
            text_feat=text_feat,
            img_feat=img_feat,
            has_img=has_img,
            has_text=general_has_text,
            num_entities=num_entities,
            num_relations=mcfg["num_relations"],
            rank=mcfg.get("rank", 128),
            init_size=mcfg.get("init_size", 0.001),
            noise_preserve_ratio=mcfg.get("noise_preserve_ratio", 0.2),
            wn3_weight=tr.get("wn3_weight", 0.005),
            pca_init=mcfg.get("pca_init", True),
            pca_fit_scope=mcfg.get("pca_fit_scope", "train_entities"),
            pca_random_state=cfg.get("system", {}).get("seed"),
            faithful_upstream_reconstruction=mcfg.get("faithful_upstream_reconstruction", True),
        )
        return model, num_entities

    if model_name == "openbg_img_adamf_mat":
        from ml.training.src.models.recent_baselines.adamf_mat import OpenBGAdaMFMAT

        cache_dir = cfg["dataset"].get("cache_dir", "")
        cache_format = cfg["dataset"].get("cache_format", "raw")
        text_feat, img_feat, has_img = feature_inputs(cache_format)
        mcfg = cfg["model"]
        num_entities = text_feat.shape[0]
        print("[BuildModel] building recent baseline: AdaMF-MAT")
        model = OpenBGAdaMFMAT(
            text_feat=text_feat,
            img_feat=img_feat,
            has_img=has_img,
            has_text=general_has_text,
            num_entities=num_entities,
            num_relations=mcfg["num_relations"],
            d=mcfg.get("dim", 128),
            margin=mcfg.get("margin", 6.0),
            epsilon=mcfg.get("epsilon", 2.0),
        )
        return model, num_entities

    if model_name == "openbg_img_native":
        from ml.training.src.models.recent_baselines.native import OpenBGNativE

        cache_dir = cfg["dataset"].get("cache_dir", "")
        cache_format = cfg["dataset"].get("cache_format", "raw")
        text_feat, img_feat, has_img = feature_inputs(cache_format)
        mcfg = cfg["model"]
        num_entities = text_feat.shape[0]
        print("[BuildModel] building recent baseline: NativE")
        model = OpenBGNativE(
            text_feat=text_feat,
            img_feat=img_feat,
            has_img=has_img,
            has_text=general_has_text,
            num_entities=num_entities,
            num_relations=mcfg["num_relations"],
            d=mcfg.get("dim", 128),
            margin=mcfg.get("margin", 6.0),
            epsilon=mcfg.get("epsilon", 2.0),
        )
        return model, num_entities

    if model_name == "openbg_img_apkgc":
        from ml.training.src.models.recent_baselines.apkgc import OpenBGAPKGC

        cache_dir = cfg["dataset"].get("cache_dir", "")
        cache_format = cfg["dataset"].get("cache_format", "raw")
        text_feat, img_feat, has_img = feature_inputs(cache_format)
        mcfg = cfg["model"]
        tr = cfg["training"]
        num_entities = text_feat.shape[0]
        print("[BuildModel] building recent baseline: APKGC")
        model = OpenBGAPKGC(
            text_feat=text_feat,
            img_feat=img_feat,
            has_img=has_img,
            has_text=general_has_text,
            num_entities=num_entities,
            num_relations=mcfg["num_relations"],
            d=mcfg.get("dim", 128),
            margin=mcfg.get("margin", 6.0),
            epsilon=mcfg.get("epsilon", 2.0),
            num_hidden_layers=mcfg.get("num_hidden_layers", 1),
            num_attention_heads=mcfg.get("num_attention_heads", 1),
            joint_way=mcfg.get("joint_way", "Mformer_hd_mean"),
            num_proj=mcfg.get("num_proj", 1),
            add_noise=mcfg.get("add_noise", False),
            noise_update=mcfg.get("noise_update", "epoch"),
            noise_ratio=mcfg.get("noise_ratio", 0.2),
            mask_ratio=mcfg.get("mask_ratio", 0.7),
            adv_temperature=tr.get("adv_temperature", 2.0),
            attention_dropout=mcfg.get("attention_dropout", 0.1),
            use_intermediate=mcfg.get("use_intermediate", False),
            intermediate_size=mcfg.get("intermediate_size"),
        )
        return model, num_entities

    if model_name == "openbg_img_tucker":
        from ml.training.src.models.structure_baselines import StructureTuckERLP

        cache_dir = cfg["dataset"].get("cache_dir", "")
        cache_format = cfg["dataset"].get("cache_format", "auto")
        d = cfg["embedding"]["d"]
        tr = cfg["training"]
        num_relations = cfg["model"]["num_relations"]
        neg_ratio = tr.get("neg_ratio", 10)
        adv_temperature = tr.get("adv_temperature", 1.0)
        entity_l2_weight = tr.get("entity_l2_weight", 1e-6)
        relation_l2_weight = tr.get("relation_l2_weight", 1e-6)
        core_l2_weight = tr.get("core_l2_weight", 1e-6)

        num_entities = entity_count(cache_format)

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

        cache_dir = cfg["dataset"].get("cache_dir", "")
        cache_format = cfg["dataset"].get("cache_format", "auto")
        d = cfg["embedding"]["d"]
        tr = cfg["training"]
        num_relations = cfg["model"]["num_relations"]
        neg_ratio = tr.get("neg_ratio", 10)
        adv_temperature = tr.get("adv_temperature", 1.0)
        entity_l2_weight = tr.get("entity_l2_weight", 1e-6)

        num_entities = entity_count(cache_format)

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
        cache_dir = cfg["dataset"].get("cache_dir", "")
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

        text_feat, img_feat, has_img = feature_inputs(cache_format)
        has_text = general_has_text

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
                has_text=has_text,
                protocol_version=protocol_version,
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
                has_text=has_text,
                protocol_version=protocol_version,
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
                has_text=has_text,
                protocol_version=protocol_version,
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
                has_text=has_text,
                protocol_version=protocol_version,
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
                has_text=has_text,
                protocol_version=protocol_version,
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

    raise ValueError(f"Unknown model.name: {requested_model_name}")
