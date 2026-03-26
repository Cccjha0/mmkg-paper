import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_labeled_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def deduplicate_preserve_order(triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen = set()
    out = []
    for triple in triples:
        if triple in seen:
            continue
        seen.add(triple)
        out.append(triple)
    return out


def allocate_by_capacity(counts_by_key: dict[str, int], total_target: int, min_train_left: int = 1) -> dict[str, int]:
    keys = list(counts_by_key.keys())
    total = sum(counts_by_key.values())
    if total_target < 0 or total_target > total:
        raise ValueError(f"Invalid target size: {total_target} for total={total}")

    capacities = {k: max(0, counts_by_key[k] - min_train_left) for k in keys}
    total_capacity = sum(capacities.values())
    if total_target > total_capacity:
        raise ValueError(f"Target {total_target} exceeds removable capacity {total_capacity}")

    raw = {}
    alloc = {}
    for k in keys:
        raw[k] = total_target * (counts_by_key[k] / total)
        alloc[k] = min(capacities[k], int(raw[k]))

    current = sum(alloc.values())
    remainders = sorted(
        ((raw[k] - alloc[k], counts_by_key[k], k) for k in keys),
        reverse=True,
    )
    i = 0
    while current < total_target:
        _, _, k = remainders[i % len(remainders)]
        if alloc[k] < capacities[k]:
            alloc[k] += 1
            current += 1
        i += 1
        if i > len(remainders) * (total_target + 1):
            raise RuntimeError("Failed to allocate target size under capacity constraints.")
    return alloc


def split_holdout_by_relation(
    holdout_by_rel: dict[str, list[int]],
    dev_target: int,
    rng: random.Random,
) -> tuple[set[int], set[int]]:
    holdout_counts = {rel: len(idxs) for rel, idxs in holdout_by_rel.items()}
    dev_alloc = allocate_by_capacity(holdout_counts, dev_target, min_train_left=0)
    dev_indices = set()
    test_indices = set()
    for rel, idxs in holdout_by_rel.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        cut = dev_alloc[rel]
        dev_indices.update(shuffled[:cut])
        test_indices.update(shuffled[cut:])
    return dev_indices, test_indices


def compute_train_counters(
    triples: list[tuple[str, str, str]],
    train_indices: set[int],
) -> tuple[Counter, Counter]:
    entity_counter = Counter()
    relation_counter = Counter()
    for idx in train_indices:
        h, r, t = triples[idx]
        relation_counter[r] += 1
        entity_counter[h] += 1
        entity_counter[t] += 1
    return entity_counter, relation_counter


def can_remove_from_train(
    idx: int,
    triples: list[tuple[str, str, str]],
    train_entity_counter: Counter,
    train_relation_counter: Counter,
) -> bool:
    h, r, t = triples[idx]
    if train_relation_counter[r] <= 1:
        return False
    if h == t:
        return train_entity_counter[h] > 2
    return train_entity_counter[h] > 1 and train_entity_counter[t] > 1


def move_idx_to_train(
    idx: int,
    target_split: set[int],
    train_indices: set[int],
    triples: list[tuple[str, str, str]],
    train_entity_counter: Counter,
    train_relation_counter: Counter,
):
    if idx not in target_split:
        return
    target_split.remove(idx)
    train_indices.add(idx)
    h, r, t = triples[idx]
    train_relation_counter[r] += 1
    train_entity_counter[h] += 1
    train_entity_counter[t] += 1


def move_idx_from_train(
    idx: int,
    target_split: set[int],
    train_indices: set[int],
    triples: list[tuple[str, str, str]],
    train_entity_counter: Counter,
    train_relation_counter: Counter,
):
    if idx not in train_indices:
        return
    if not can_remove_from_train(idx, triples, train_entity_counter, train_relation_counter):
        raise RuntimeError(f"Train coverage would break by moving triple index {idx}")
    train_indices.remove(idx)
    target_split.add(idx)
    h, r, t = triples[idx]
    train_relation_counter[r] -= 1
    train_entity_counter[h] -= 1
    train_entity_counter[t] -= 1


def repair_train_coverage(
    triples: list[tuple[str, str, str]],
    train_indices: set[int],
    dev_indices: set[int],
    test_indices: set[int],
) -> dict[str, int]:
    moved_from_dev = 0
    moved_from_test = 0

    train_entity_counter, train_relation_counter = compute_train_counters(triples, train_indices)

    all_entities = set()
    all_relations = set()
    entity_to_holdout = defaultdict(list)
    relation_to_holdout = defaultdict(list)

    for idx in sorted(dev_indices | test_indices):
        h, r, t = triples[idx]
        all_entities.add(h)
        all_entities.add(t)
        all_relations.add(r)
        entity_to_holdout[h].append(idx)
        entity_to_holdout[t].append(idx)
        relation_to_holdout[r].append(idx)

    missing_entities = [
        ent for ent in all_entities if train_entity_counter[ent] == 0
    ]
    for ent in sorted(missing_entities):
        candidates = entity_to_holdout[ent]
        chosen = None
        for idx in candidates:
            if idx in test_indices:
                chosen = (idx, test_indices)
                break
        if chosen is None:
            for idx in candidates:
                if idx in dev_indices:
                    chosen = (idx, dev_indices)
                    break
        if chosen is None:
            raise RuntimeError(f"Cannot repair missing entity coverage for {ent}")
        idx, split_ref = chosen
        move_idx_to_train(idx, split_ref, train_indices, triples, train_entity_counter, train_relation_counter)
        if split_ref is dev_indices:
            moved_from_dev += 1
        else:
            moved_from_test += 1

    missing_relations = [
        rel for rel in all_relations if train_relation_counter[rel] == 0
    ]
    for rel in sorted(missing_relations):
        candidates = relation_to_holdout[rel]
        chosen = None
        for idx in candidates:
            if idx in test_indices:
                chosen = (idx, test_indices)
                break
        if chosen is None:
            for idx in candidates:
                if idx in dev_indices:
                    chosen = (idx, dev_indices)
                    break
        if chosen is None:
            raise RuntimeError(f"Cannot repair missing relation coverage for {rel}")
        idx, split_ref = chosen
        move_idx_to_train(idx, split_ref, train_indices, triples, train_entity_counter, train_relation_counter)
        if split_ref is dev_indices:
            moved_from_dev += 1
        else:
            moved_from_test += 1

    return {
        "moved_from_dev_to_train": moved_from_dev,
        "moved_from_test_to_train": moved_from_test,
    }


def refill_split(
    split_name: str,
    target_size: int,
    target_split: set[int],
    train_indices: set[int],
    triples: list[tuple[str, str, str]],
    rng: random.Random,
) -> int:
    train_entity_counter, train_relation_counter = compute_train_counters(triples, train_indices)
    moved = 0
    while len(target_split) < target_size:
        candidates = list(train_indices)
        rng.shuffle(candidates)
        chosen = None
        for idx in candidates:
            if can_remove_from_train(idx, triples, train_entity_counter, train_relation_counter):
                chosen = idx
                break
        if chosen is None:
            raise RuntimeError(f"Cannot refill {split_name}; no safe removable train triple remains.")
        move_idx_from_train(chosen, target_split, train_indices, triples, train_entity_counter, train_relation_counter)
        moved += 1
    return moved


def relation_distribution(indices: set[int], triples: list[tuple[str, str, str]]) -> dict[str, int]:
    c = Counter()
    for idx in indices:
        c[triples[idx][1]] += 1
    return dict(sorted(c.items()))


def validate_split(
    triples: list[tuple[str, str, str]],
    train_indices: set[int],
    dev_indices: set[int],
    test_indices: set[int],
):
    if train_indices & dev_indices or train_indices & test_indices or dev_indices & test_indices:
        raise RuntimeError("Split overlap detected.")
    if len(train_indices | dev_indices | test_indices) != len(triples):
        raise RuntimeError("Split union does not cover all labeled triples.")

    train_entities = set()
    train_relations = set()
    for idx in train_indices:
        h, r, t = triples[idx]
        train_entities.add(h)
        train_entities.add(t)
        train_relations.add(r)

    for split_name, indices in (("dev", dev_indices), ("test", test_indices)):
        missing_entities = set()
        missing_relations = set()
        for idx in indices:
            h, r, t = triples[idx]
            if h not in train_entities:
                missing_entities.add(h)
            if t not in train_entities:
                missing_entities.add(t)
            if r not in train_relations:
                missing_relations.add(r)
        if missing_entities:
            raise RuntimeError(f"{split_name} contains entities not covered by train: {len(missing_entities)}")
        if missing_relations:
            raise RuntimeError(f"{split_name} contains relations not covered by train: {len(missing_relations)}")


def write_split(path: Path, triples: list[tuple[str, str, str]], indices: set[int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for idx in sorted(indices):
            h, r, t = triples[idx]
            f.write(f"{h}\t{r}\t{t}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/datasets/openbg_img/raw/OpenBG-IMG_train.tsv")
    ap.add_argument("--dev", default="data/datasets/openbg_img/raw/OpenBG-IMG_dev.tsv")
    ap.add_argument("--output-dir", default="data/datasets/openbg_img/paper_split")
    ap.add_argument("--train-out", default="OpenBG-IMG_paper_train.tsv")
    ap.add_argument("--dev-out", default="OpenBG-IMG_paper_dev.tsv")
    ap.add_argument("--test-out", default="OpenBG-IMG_paper_test.tsv")
    ap.add_argument("--dev-size", type=int, default=5000)
    ap.add_argument("--test-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260326)
    args = ap.parse_args()

    train_path = Path(args.train)
    dev_path = Path(args.dev)
    output_dir = Path(args.output_dir)

    raw_train = read_labeled_triples(train_path)
    raw_dev = read_labeled_triples(dev_path)
    raw_all = raw_train + raw_dev
    triples = deduplicate_preserve_order(raw_all)

    rng = random.Random(args.seed)

    total = len(triples)
    holdout_target = args.dev_size + args.test_size
    if holdout_target >= total:
        raise ValueError("dev_size + test_size must be smaller than total labeled triples.")

    by_relation = defaultdict(list)
    for idx, (_, rel, _) in enumerate(triples):
        by_relation[rel].append(idx)

    holdout_alloc = allocate_by_capacity(
        {rel: len(idxs) for rel, idxs in by_relation.items()},
        total_target=holdout_target,
        min_train_left=1,
    )

    holdout_by_rel = {}
    train_indices = set()
    for rel, idxs in by_relation.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        cut = holdout_alloc[rel]
        holdout_by_rel[rel] = shuffled[:cut]
        train_indices.update(shuffled[cut:])

    dev_indices, test_indices = split_holdout_by_relation(holdout_by_rel, args.dev_size, rng)
    train_indices = set(range(total)) - dev_indices - test_indices

    repair_stats = repair_train_coverage(triples, train_indices, dev_indices, test_indices)
    refill_dev = refill_split("dev", args.dev_size, dev_indices, train_indices, triples, rng)
    refill_test = refill_split("test", args.test_size, test_indices, train_indices, triples, rng)

    validate_split(triples, train_indices, dev_indices, test_indices)

    train_out = output_dir / args.train_out
    dev_out = output_dir / args.dev_out
    test_out = output_dir / args.test_out
    meta_out = output_dir / "split_meta.json"

    write_split(train_out, triples, train_indices)
    write_split(dev_out, triples, dev_indices)
    write_split(test_out, triples, test_indices)

    train_entities = set()
    train_relations = set()
    for idx in train_indices:
        h, r, t = triples[idx]
        train_entities.add(h)
        train_entities.add(t)
        train_relations.add(r)

    def count_missing(indices: set[int]) -> tuple[int, int]:
        missing_entities = set()
        missing_relations = set()
        for idx in indices:
            h, r, t = triples[idx]
            if h not in train_entities:
                missing_entities.add(h)
            if t not in train_entities:
                missing_entities.add(t)
            if r not in train_relations:
                missing_relations.add(r)
        return len(missing_entities), len(missing_relations)

    dev_missing_entities, dev_missing_relations = count_missing(dev_indices)
    test_missing_entities, test_missing_relations = count_missing(test_indices)

    meta = {
        "source_files": {
            "train": str(train_path.as_posix()),
            "dev": str(dev_path.as_posix()),
        },
        "output_files": {
            "train": str(train_out.as_posix()),
            "dev": str(dev_out.as_posix()),
            "test": str(test_out.as_posix()),
        },
        "seed": args.seed,
        "requested_sizes": {
            "dev": args.dev_size,
            "test": args.test_size,
        },
        "raw_counts": {
            "train": len(raw_train),
            "dev": len(raw_dev),
            "combined_labeled": len(raw_all),
            "unique_labeled": len(triples),
            "duplicates_removed": len(raw_all) - len(triples),
        },
        "final_counts": {
            "train": len(train_indices),
            "dev": len(dev_indices),
            "test": len(test_indices),
        },
        "coverage_check": {
            "dev_missing_entities_vs_train": dev_missing_entities,
            "dev_missing_relations_vs_train": dev_missing_relations,
            "test_missing_entities_vs_train": test_missing_entities,
            "test_missing_relations_vs_train": test_missing_relations,
        },
        "repair_stats": {
            **repair_stats,
            "refill_dev_from_train": refill_dev,
            "refill_test_from_train": refill_test,
        },
        "relation_stats": {
            "num_relations_total": len(by_relation),
            "train_distribution": relation_distribution(train_indices, triples),
            "dev_distribution": relation_distribution(dev_indices, triples),
            "test_distribution": relation_distribution(test_indices, triples),
        },
    }

    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[PaperSplit] unique labeled triples={len(triples)}")
    print(f"[PaperSplit] train={len(train_indices)} dev={len(dev_indices)} test={len(test_indices)}")
    print(
        "[PaperSplit] coverage "
        f"dev_missing_entities={dev_missing_entities} "
        f"dev_missing_relations={dev_missing_relations} "
        f"test_missing_entities={test_missing_entities} "
        f"test_missing_relations={test_missing_relations}"
    )
    print(f"[PaperSplit] wrote train -> {train_out.as_posix()}")
    print(f"[PaperSplit] wrote dev   -> {dev_out.as_posix()}")
    print(f"[PaperSplit] wrote test  -> {test_out.as_posix()}")
    print(f"[PaperSplit] wrote meta  -> {meta_out.as_posix()}")


if __name__ == "__main__":
    main()
