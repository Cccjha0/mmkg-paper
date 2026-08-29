from unittest.mock import patch

import torch

from ml.training.src.eval import filtered_ranking as ranking


class ExactDirectionalModel(torch.nn.Module):
    """Deterministic scores with ties, so strict-greater ranking is exercised."""

    def score_tail(self, triples):
        return (triples[:, 2] % 4).float()

    def score_head(self, triples):
        return (3 - triples[:, 0] % 4).float()


TRIPLES = torch.tensor(
    [
        [0, 0, 2],
        [1, 0, 4],
        [3, 1, 1],
    ],
    dtype=torch.long,
)
TRUE_TAILS = {
    (0, 0): {2, 3, 6},
    (1, 0): {0, 4},
    (3, 1): {1, 2, 5},
}
TRUE_HEADS = {
    (0, 2): {0, 2, 6},
    (0, 4): {1, 4},
    (1, 1): {0, 3, 5},
}


def _reference_ranks(direction):
    model = ExactDirectionalModel()
    ranks = []
    for h, r, t in TRIPLES.tolist():
        target_triple = torch.tensor([[h, r, t]], dtype=torch.long)
        if direction == "tail":
            target_score = model.score_tail(target_triple)[0]
            filtered = TRUE_TAILS.get((h, r), set()) - {t}
        else:
            target_score = model.score_head(target_triple)[0]
            filtered = TRUE_HEADS.get((r, t), set()) - {h}

        greater = 0
        for entity in range(7):
            if entity in filtered:
                continue
            candidate = [h, r, entity] if direction == "tail" else [entity, r, t]
            candidate_tensor = torch.tensor([candidate], dtype=torch.long)
            score = (
                model.score_tail(candidate_tensor)[0]
                if direction == "tail"
                else model.score_head(candidate_tensor)[0]
            )
            greater += int(score > target_score)
        ranks.append(greater + 1)
    return torch.tensor(ranks, dtype=torch.long)


def _capture_ranks(direction, query_batch_size, chunk_size, dense_mask_limit):
    captured = []
    original_metrics = ranking._metrics_from_ranks

    def capture(ranks, ks=(1, 3, 10)):
        captured.append(ranks.detach().cpu().clone())
        return original_metrics(ranks, ks=ks)

    with (
        patch.object(ranking, "_DENSE_FILTER_MASK_MAX_BYTES", dense_mask_limit),
        patch.object(ranking, "_metrics_from_ranks", side_effect=capture),
    ):
        ranking.filtered_ranking_eval(
            model=ExactDirectionalModel(),
            triples=TRIPLES,
            true_tails=TRUE_TAILS,
            true_heads=TRUE_HEADS,
            num_entities=7,
            chunk_size=chunk_size,
            query_batch_size=query_batch_size,
            device="cpu",
            direction=direction,
        )
    assert len(captured) == 1
    return captured[0]


def test_dense_and_sparse_filter_paths_match_reference_ranks_exactly():
    for direction in ("tail", "head"):
        expected = _reference_ranks(direction)
        for query_batch_size in (1, 2, 4):
            for chunk_size in (1, 3, 7):
                dense = _capture_ranks(direction, query_batch_size, chunk_size, dense_mask_limit=10_000)
                sparse = _capture_ranks(direction, query_batch_size, chunk_size, dense_mask_limit=0)
                assert torch.equal(dense, expected)
                assert torch.equal(sparse, expected)


def test_dense_and_sparse_paths_preserve_bidirectional_subgroup_metrics():
    kwargs = {
        "model": ExactDirectionalModel(),
        "triples": TRIPLES,
        "true_tails": TRUE_TAILS,
        "true_heads": TRUE_HEADS,
        "num_entities": 7,
        "chunk_size": 3,
        "query_batch_size": 2,
        "device": "cpu",
        "direction": "both",
        "entity_has_img": torch.tensor([True, False, True, False, True, False, True]),
    }
    with patch.object(ranking, "_DENSE_FILTER_MASK_MAX_BYTES", 10_000):
        dense_metrics = ranking.filtered_ranking_eval(**kwargs)
    with patch.object(ranking, "_DENSE_FILTER_MASK_MAX_BYTES", 0):
        sparse_metrics = ranking.filtered_ranking_eval(**kwargs)
    assert dense_metrics == sparse_metrics
    assert dense_metrics["has_img_mrr"] == 0.5 * (
        dense_metrics["tail_has_img_mrr"] + dense_metrics["head_has_img_mrr"]
    )
    assert "direction_balanced_has_img_mrr" not in dense_metrics
