"""Versioned contract for deployable features and evaluation-only truth."""

SCORE_INFORMATION_CONTRACT = "unfiltered_features_and_normalization_v2"


def require_score_information_contract(payload: dict) -> None:
    if payload.get("score_information_contract") != SCORE_INFORMATION_CONTRACT:
        raise ValueError(
            "Legacy or missing score information contract: re-export unfiltered "
            "DEV/TEST scores and refit/relock policies before evaluation."
        )


def require_unfiltered_rows(rows) -> None:
    for row in rows:
        require_score_information_contract(row)
