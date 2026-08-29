from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.training.scripts.preprocess_external_mmkg import read_feature_key_map, read_openke_mapping


API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit Wikidata-QID to MKG-W HDF5 feature-key crosswalk."
    )
    parser.add_argument("--entity2id", type=Path, required=True)
    parser.add_argument("--feature-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--reverse-batch-size", type=int, default=20, help="Smaller batches avoid long Wikipedia URLs.")
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument(
        "--reverse-only",
        action="store_true",
        help="Resolve remaining entities by mapping HDF5 Wikipedia titles back to Wikidata QIDs.",
    )
    parser.add_argument("--skip-reverse", action="store_true", help="Do not run the title-to-QID fallback pass.")
    return parser.parse_args()


def candidate_keys(title: str) -> list[str]:
    underscored = title.replace(" ", "_")
    # The supplied HDF5 naming convention replaces Wikipedia namespace/title
    # colons with an underscore, removes slashes, and strips a terminal dot.
    hdf5_style = underscored.replace(":", "_").replace("/", "").rstrip(".")
    encoded = quote(underscored, safe="!$&'()*+,-./:;=@_~")
    encoded_hdf5_style = quote(hdf5_style, safe="!$&'()*+,-.;=@_~")
    return list(dict.fromkeys([underscored, hdf5_style, encoded, encoded_hdf5_style, title]))


def fetch_sitelinks(qids: list[str], max_retries: int) -> dict[str, str]:
    params = urlencode(
        {
            "action": "wbgetentities",
            "format": "json",
            "props": "sitelinks",
            "sitefilter": "enwiki",
            "ids": "|".join(qids),
        }
    )
    request = Request(f"{API_URL}?{params}", headers={"User-Agent": "mmkg-project-research/1.0"})
    payload = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
            break
        except HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= max_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(60.0, 2.0 ** attempt)
            print(f"[Crosswalk] HTTP {exc.code}; retrying in {delay:.1f}s")
            time.sleep(delay)
        except (URLError, TimeoutError, ConnectionError) as exc:
            if attempt >= max_retries:
                raise
            delay = min(60.0, 2.0 ** attempt)
            print(f"[Crosswalk] network error {exc!s}; retrying in {delay:.1f}s")
            time.sleep(delay)
    if payload is None:
        raise RuntimeError("Wikidata response was not loaded.")
    out: dict[str, str] = {}
    for qid, entity in payload.get("entities", {}).items():
        title = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
        if title:
            out[qid] = str(title)
    return out


def fetch_title_qids(keys: list[str], max_retries: int) -> dict[str, str]:
    """Resolve HDF5 titles (including redirects) back to Wikidata QIDs."""
    titles = [unquote(key).replace("_", " ") for key in keys]
    params = urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "redirects": "1",
            "titles": "|".join(titles),
        }
    )
    request = Request(f"{WIKIPEDIA_API_URL}?{params}", headers={"User-Agent": "mmkg-project-research/1.0"})
    payload = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
            break
        except HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= max_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(60.0, 2.0 ** attempt)
            print(f"[Crosswalk/reverse] HTTP {exc.code}; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
        except (URLError, TimeoutError, ConnectionError) as exc:
            if attempt >= max_retries:
                raise
            delay = min(60.0, 2.0 ** attempt)
            print(f"[Crosswalk/reverse] network error {exc!s}; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    if payload is None:
        raise RuntimeError("Wikipedia response was not loaded.")
    query = payload.get("query", {})
    aliases = {
        str(row["from"]): str(row["to"])
        for group in (query.get("normalized", []), query.get("redirects", []))
        for row in group
    }
    title_to_qid = {
        str(page.get("title")): str(page.get("pageprops", {}).get("wikibase_item"))
        for page in query.get("pages", {}).values()
        if page.get("pageprops", {}).get("wikibase_item")
    }
    out: dict[str, str] = {}
    for key, title in zip(keys, titles):
        resolved = title
        seen: set[str] = set()
        while resolved in aliases and resolved not in seen:
            seen.add(resolved)
            resolved = aliases[resolved]
        qid = title_to_qid.get(resolved)
        if qid:
            out[key] = qid
    return out


def write_crosswalk(path: Path, entity2id: dict[str, int], matched: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for entity, entity_id in sorted(entity2id.items(), key=lambda item: item[1]):
            if entity in matched:
                handle.write(f"{entity}\t{matched[entity]}\n")


def main() -> None:
    args = parse_args()
    if not 1 <= args.batch_size <= 50:
        raise ValueError("Wikidata wbgetentities batch size must be in [1, 50].")
    if not 1 <= args.reverse_batch_size <= 50:
        raise ValueError("Wikipedia reverse batch size must be in [1, 50].")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("This command requires h5py.") from exc
    entity2id, _ = read_openke_mapping(args.entity2id, "entity")
    with h5py.File(args.feature_h5, "r") as handle:
        available = set(handle.keys())
    qid_to_entity = {entity.rsplit("/", 1)[-1]: entity for entity in entity2id}
    qids = sorted(qid_to_entity, key=lambda qid: entity2id[qid_to_entity[qid]])
    matched: dict[str, str] = read_feature_key_map(args.output) if args.output.exists() else {}
    unknown_entities = sorted(set(matched) - set(entity2id))
    invalid_keys = sorted({key for key in matched.values() if key not in available})
    if unknown_entities or invalid_keys:
        raise ValueError(
            "Existing crosswalk checkpoint is incompatible with the requested mapping/HDF5: "
            f"unknown_entities={unknown_entities[:5]}, invalid_keys={invalid_keys[:5]}"
        )
    no_enwiki: list[str] = []
    title_not_in_h5: list[dict[str, str]] = []
    if not args.reverse_only:
        remaining_qids = [qid for qid in qids if qid_to_entity[qid] not in matched]
        for start in range(0, len(remaining_qids), args.batch_size):
            batch = remaining_qids[start : start + args.batch_size]
            sitelinks = fetch_sitelinks(batch, args.max_retries)
            for qid in batch:
                entity = qid_to_entity[qid]
                title = sitelinks.get(qid)
                if title is None:
                    no_enwiki.append(entity)
                    continue
                selected = next((key for key in candidate_keys(title) if key in available), None)
                if selected is None:
                    title_not_in_h5.append({"entity": entity, "title": title})
                    continue
                matched[entity] = selected
            print(
                f"[Crosswalk] scanned remaining {min(start + len(batch), len(remaining_qids))}/"
                f"{len(remaining_qids)} (matched={len(matched)}/{len(qids)})",
                flush=True,
            )
            write_crosswalk(args.output, entity2id, matched)
            if args.request_delay > 0:
                time.sleep(args.request_delay)

    forward_matched = len(matched)
    if not args.skip_reverse:
        unmatched_qids = set(qid_to_entity) - {entity.rsplit("/", 1)[-1] for entity in matched}
        unused_keys = sorted(available - set(matched.values()))
        for start in range(0, len(unused_keys), args.reverse_batch_size):
            batch = unused_keys[start : start + args.reverse_batch_size]
            for key, qid in fetch_title_qids(batch, args.max_retries).items():
                if qid in unmatched_qids:
                    matched[qid_to_entity[qid]] = key
                    unmatched_qids.remove(qid)
            print(f"[Crosswalk/reverse] scanned {min(start + len(batch), len(unused_keys))}/{len(unused_keys)}", flush=True)
            write_crosswalk(args.output, entity2id, matched)
            if args.request_delay > 0:
                time.sleep(args.request_delay)
    write_crosswalk(args.output, entity2id, matched)
    report = {
        "entities": len(entity2id),
        "matched": len(matched),
        "matched_before_reverse": forward_matched,
        "matched_by_reverse": len(matched) - forward_matched,
        "no_enwiki": len(no_enwiki),
        "title_not_in_hdf5": len(title_not_in_h5),
        "no_enwiki_examples": no_enwiki[:20],
        "title_not_in_hdf5_examples": title_not_in_h5[:20],
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
