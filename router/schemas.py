from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QueryEvalRecord:
    query_id: str
    split: str
    direction: str
    relation_id: int
    relation_name: str
    head_id: int
    tail_id: int
    target_entity_id: int
    target_position: str
    target_has_img: int
    target_regime: str
    expert_name: str
    rank: int
    rr: float
    hit1: int
    hit3: int
    hit10: int
    top1_score: float
    top2_score: float
    score_margin: float
    correct_score: float
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


QUERY_EVAL_HEADER = list(QueryEvalRecord.__dataclass_fields__.keys())


@dataclass(frozen=True)
class RouterFeatureRecord:
    query_id: str
    split: str
    seed: int
    direction: str
    relation_id: int
    relation_name: str
    head_id: int
    tail_id: int
    target_entity_id: int
    target_position: str
    target_has_img: int
    target_regime: str
    relation_gain_prior: float
    relation_fusion_win_rate: float
    relation_support: int
    relation_is_visual_prior: int
    text_img_cosine: float
    img_is_missing_replaced: int
    fusion_margin: float
    struct_margin: float
    fusion_correct_score: float
    struct_correct_score: float
    delta_margin: float
    rr_fusion: float
    rr_struct: float
    rank_fusion: int
    rank_struct: int
    label_gain: int | None = None
    delta_threshold: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


ROUTER_FEATURE_HEADER = list(RouterFeatureRecord.__dataclass_fields__.keys())
