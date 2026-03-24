import os
import json
import argparse
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer


def parse_ent_id(ent_str: str) -> int:
    # "ent_001235" -> 1235
    if not ent_str.startswith("ent_"):
        raise ValueError(f"Bad entity id: {ent_str}")
    return int(ent_str.replace("ent_", ""))


def get_num_entities_from_entity2text(path: str) -> int:
    max_id = -1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            eid = parse_ent_id(parts[0])
            max_id = max(max_id, eid)
    return max_id + 1


def load_texts(path: str, num_entities: int):
    texts = [""] * num_entities
    has_text = torch.zeros(num_entities, dtype=torch.bool)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            eid = parse_ent_id(parts[0])
            text = parts[1].strip()
            if 0 <= eid < num_entities:
                texts[eid] = text
                has_text[eid] = True

    return texts, has_text


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity2text", required=True)
    ap.add_argument("--cache_dir", default="data/cache/openbg_img")
    ap.add_argument("--model_name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    num_entities = get_num_entities_from_entity2text(args.entity2text)
    texts, has_text = load_texts(args.entity2text, num_entities)

    print(f"[TextCache] num_entities={num_entities}, has_text.sum={int(has_text.sum())}")

    encoder = SentenceTransformer(args.model_name)
    raw = encoder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    raw = torch.from_numpy(raw).float()  # [N, raw_dim]
    raw_dim = raw.shape[1]
    print(f"[TextCache] raw_dim={raw_dim}")

    # Legacy projected cache is still written during the transition period.
    proj = torch.nn.Linear(raw_dim, args.d, bias=False)
    torch.nn.init.normal_(proj.weight, mean=0.0, std=0.02)
    text_emb = proj(raw)  # [N, d]

    torch.save(raw, os.path.join(args.cache_dir, "text_feat_raw.pt"))
    torch.save(text_emb, os.path.join(args.cache_dir, "text_emb.pt"))
    torch.save(has_text, os.path.join(args.cache_dir, "has_text.pt"))
    torch.save(proj.state_dict(), os.path.join(args.cache_dir, "text_proj.pt"))

    meta = {
        "created_at": datetime.utcnow().isoformat(),
        "cache_version": 2,
        "text_encoder": args.model_name,
        "num_entities": num_entities,
        "raw_dim": raw_dim,
        "d": args.d,
        "entity2text": os.path.abspath(args.entity2text),
        "feature_file": "text_feat_raw.pt",
        "legacy_files": ["text_emb.pt", "text_proj.pt"],
        "notes": "text from 2nd column of TSV; ids parsed from ent_XXXXXX",
    }
    with open(os.path.join(args.cache_dir, "text_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[TextCache] saved to {args.cache_dir}")


if __name__ == "__main__":
    main()
