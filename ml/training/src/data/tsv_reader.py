def parse_ent(s: str) -> int:
    if not s.startswith("ent_"):
        raise ValueError(f"Bad entity token: {s}")
    return int(s.replace("ent_", ""))


def parse_rel(s: str) -> int:
    if not s.startswith("rel_"):
        raise ValueError(f"Bad relation token: {s}")
    return int(s.replace("rel_", ""))


def read_allow_2or3(path: str):
    """
    Returns:
      triples3: list[(h,r,t)] ints
      queries2: list[(h,r)] ints
    """
    triples3 = []
    queries2 = []
    bad = 0

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                h, r, t = parts
                triples3.append((parse_ent(h), parse_rel(r), parse_ent(t)))
            elif len(parts) == 2:
                h, r = parts
                queries2.append((parse_ent(h), parse_rel(r)))
            else:
                bad += 1
                # keep going; do not crash
                # print(f"[WARN] {path}:{ln} cols={len(parts)} {repr(line)}")

    return triples3, queries2, bad


def read_integer_triples(path: str):
    """Read canonical ``head<TAB>relation<TAB>tail`` integer triples."""
    triples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number}: expected exactly three tab-separated columns")
            try:
                triples.append(tuple(int(value) for value in parts))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-integer canonical triple") from exc
    return triples
