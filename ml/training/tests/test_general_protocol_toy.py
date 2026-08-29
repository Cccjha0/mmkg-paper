from __future__ import annotations

import torch

from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.data.dataset_spec import MMKG_GENERAL_V1
from ml.training.src.eval.filtered_ranking import filtered_ranking_eval
from ml.training.src.models.openbg_img_gated_lp import OpenBGImgGateResidualLP
from ml.training.src.models.recent_baselines.mhyper import OpenBGMHyper


def test_four_modality_states_forward_backward_and_filtered_ranking() -> None:
    # Entities 0..3 cover T1V1, T1V0, T0V1, T0V0 respectively.
    has_text = torch.tensor([True, True, False, False])
    has_img = torch.tensor([True, False, True, False])
    model = OpenBGImgGateResidualLP(
        text_feat=torch.randn(4, 3),
        img_feat=torch.randn(4, 5),
        has_text=has_text,
        has_img=has_img,
        protocol_version=MMKG_GENERAL_V1,
        num_relations=1,
        d=4,
    )
    triples = torch.tensor([(0, 0, 1), (1, 0, 2), (2, 0, 3), (3, 0, 0)], dtype=torch.long)
    score = model.score(triples)
    assert score.shape == (4,)
    (-score.mean()).backward()
    assert model.t_missing.grad is not None
    assert model.v_missing.grad is not None

    true_tails, true_heads = build_true_facts([tuple(row) for row in triples.tolist()])
    metrics = filtered_ranking_eval(
        model=model,
        triples=triples,
        true_tails=true_tails,
        true_heads=true_heads,
        num_entities=4,
        chunk_size=2,
        query_batch_size=2,
        device="cpu",
        direction="both",
        entity_has_text=has_text,
        entity_has_img=has_img,
    )
    assert metrics["tail_T1V1_count"] == 1
    assert metrics["tail_T1V0_count"] == 1
    assert metrics["tail_T0V1_count"] == 1
    assert metrics["tail_T0V0_count"] == 1
    assert metrics["head_T1V1_count"] == 1
    assert metrics["head_T1V0_count"] == 1
    assert metrics["head_T0V1_count"] == 1
    assert metrics["head_T0V0_count"] == 1


def test_general_mhyper_masks_both_projected_and_independent_missing_modalities() -> None:
    has_text = torch.tensor([True, True, False, False])
    has_img = torch.tensor([True, False, True, False])
    model = OpenBGMHyper(
        text_feat=torch.randn(4, 3),
        img_feat=torch.randn(4, 3),
        has_text=has_text,
        has_img=has_img,
        num_entities=4,
        num_relations=1,
        rank=1,
        pca_init=False,
    )
    structure, image, text, _ = model._clean_modalities()
    assert structure.shape == (4, 2)
    assert torch.count_nonzero(image[~has_img]).item() == 0
    assert torch.count_nonzero(text[~has_text]).item() == 0
