from __future__ import annotations

from ml.training.src.data.dataset_spec import DatasetBundle, OPENBG_LEGACY_V1
from ml.training.src.data.feature_bundle import load_openbg_feature_bundle
from ml.training.src.data.tsv_reader import read_allow_2or3


def load_openbg_legacy(cfg: dict) -> DatasetBundle:
    dataset_cfg = cfg["dataset"]
    features = load_openbg_feature_bundle(
        dataset_cfg["cache_dir"], dataset_cfg.get("cache_format", "legacy")
    )
    train, _, bad_train = read_allow_2or3(dataset_cfg["train"])
    valid, _, bad_valid = read_allow_2or3(dataset_cfg["dev"])
    test, test_queries, bad_test = read_allow_2or3(dataset_cfg["test"])
    if bad_train or bad_valid or bad_test:
        print(
            "[WARN] malformed OpenBG lines skipped: "
            f"train={bad_train}, dev={bad_valid}, test={bad_test}"
        )
    if not train or not valid:
        raise RuntimeError("OpenBG legacy train/dev splits must contain labeled triples.")
    if not test and test_queries:
        # Training keeps supporting the historical query-only inference file.
        test = []
    num_entities = features.num_entities
    num_relations = int(cfg["model"]["num_relations"])
    bundle = DatasetBundle(
        name=dataset_cfg.get("name", "openbg_img"),
        protocol_version=OPENBG_LEGACY_V1,
        train_triples=train,
        valid_triples=valid,
        test_triples=test,
        num_entities=num_entities,
        num_relations=num_relations,
        entity2id={f"ent_{idx:06d}": idx for idx in range(num_entities)},
        relation2id={f"rel_{idx:04d}": idx for idx in range(num_relations)},
        features=features,
        manifest={
            "protocol": OPENBG_LEGACY_V1,
            "split": "paper_split",
            "query_only_test": bool(test_queries and not test),
            "query_only_test_count": len(test_queries) if not test else 0,
        },
    )
    bundle.validate()
    return bundle
