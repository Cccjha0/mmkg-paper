import os
import re
import json
import argparse
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


ENT_DIR_RE = re.compile(r"^ent_(\d+)$")


def parse_ent_id(ent_str: str) -> int:
    if not ent_str.startswith("ent_"):
        raise ValueError(f"Bad entity id: {ent_str}")
    return int(ent_str.replace("ent_", ""))


def get_num_entities_from_entity2text(entity2text_path: str) -> int:
    max_id = -1
    with open(entity2text_path, "r", encoding="utf-8") as f:
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


def list_image_items(images_root: str, num_entities: int):
    """
    Returns:
      items: list of (entity_id, image_path)
      has_img: BoolTensor[num_entities]
    """
    has_img = torch.zeros(num_entities, dtype=torch.bool)
    items = []

    for name in os.listdir(images_root):
        p = os.path.join(images_root, name)
        if not os.path.isdir(p):
            continue
        m = ENT_DIR_RE.match(name)
        if not m:
            continue
        eid = int(m.group(1))
        if eid >= num_entities:
            # Typically means num_entities computed incorrectly
            continue
        img_path = os.path.join(p, "image_0.jpg")
        if os.path.exists(img_path):
            has_img[eid] = True
            items.append((eid, img_path))

    items.sort(key=lambda x: x[0])
    return items, has_img


def load_images_batch(paths):
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        images.append(img)
    return images


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity2text", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--cache_dir", default="data/cache/openbg_img")
    ap.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    num_entities = get_num_entities_from_entity2text(args.entity2text)
    items, has_img = list_image_items(args.images_root, num_entities)

    print(f"[ImageCache] num_entities={num_entities}")
    print(f"[ImageCache] images found={len(items)} (has_img.sum={has_img.sum().item()})")

    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    # CLIP ViT-B/32 image_features dim is usually 512
    raw_dim = model.config.projection_dim if hasattr(model.config, "projection_dim") else 512

    img_emb_raw = torch.zeros(num_entities, raw_dim, dtype=torch.float32)

    # Batch encoding
    for i in tqdm(range(0, len(items), args.batch_size), desc="Encoding CLIP"):
        batch = items[i:i + args.batch_size]
        eids = [x[0] for x in batch]
        paths = [x[1] for x in batch]

        images = load_images_batch(paths)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        feats = model.get_image_features(**inputs)  # [B, raw_dim]
        feats = feats.detach().cpu().float()

        for idx, eid in enumerate(eids):
            img_emb_raw[eid] = feats[idx]

    # Project raw -> d (store proj weights for reproducibility)
    proj = torch.nn.Linear(raw_dim, args.d, bias=False).to(device)
    torch.nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    img_emb = torch.zeros(num_entities, args.d, dtype=torch.float32)
    chunk = 4096
    for s in tqdm(range(0, num_entities, chunk), desc="Projecting"):
        e = min(num_entities, s + chunk)
        img_emb[s:e] = proj(img_emb_raw[s:e].to(device)).detach().cpu()

    # Save artifacts
    torch.save(img_emb_raw, os.path.join(args.cache_dir, "img_feat_raw.pt"))
    torch.save(img_emb_raw, os.path.join(args.cache_dir, "img_emb_raw.pt"))
    torch.save(img_emb, os.path.join(args.cache_dir, "img_emb.pt"))
    torch.save(has_img, os.path.join(args.cache_dir, "has_img.pt"))
    torch.save(proj.state_dict(), os.path.join(args.cache_dir, "img_proj.pt"))

    meta = {
        "created_at": datetime.utcnow().isoformat(),
        "cache_version": 2,
        "image_encoder": args.model_name,
        "num_entities": num_entities,
        "num_images": int(has_img.sum().item()),
        "raw_dim": raw_dim,
        "d": args.d,
        "images_root": os.path.abspath(args.images_root),
        "entity2text": os.path.abspath(args.entity2text),
        "feature_file": "img_feat_raw.pt",
        "legacy_files": ["img_emb_raw.pt", "img_emb.pt", "img_proj.pt"],
        "notes": "ent_XXXXXX/image_0.jpg; missing images have zero raw embedding in cache; use v_missing during training",
    }
    with open(os.path.join(args.cache_dir, "img_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ImageCache] saved to {args.cache_dir}")


if __name__ == "__main__":
    main()
