"""Download monthly Lichess broadcast dumps: lichess_db_broadcast_YYYY-MM.pgn.zst

    python -m pretrain.download --months 2026-01 --out data/raw
    python -m pretrain.download --from 2024-01 --to 2026-01 --out data/raw
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.request

BASE = "https://database.lichess.org/broadcast"

def month_range(first: str, last: str) -> list[str]:
    (y, m), (y2, m2) = map(lambda s: map(int, s.split("-")), (first, last))
    out = []
    while (y, m) <= (y2, m2):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_checksums() -> dict[str, str]:
    try:
        with urllib.request.urlopen(f"{BASE}/sha256sums.txt", timeout=30) as r:
            text = r.read().decode()
    except Exception as exc:                      # checksums are a nicety, not a requirement
        print(f"warning: could not fetch sha256sums.txt ({exc}); skipping verification", file=sys.stderr)
        return {}
    return {name: digest for digest, name in (line.split() for line in text.splitlines() if line.strip())}


def download(month: str, out_dir: str, sums: dict[str, str]) -> str:
    name = f"lichess_db_broadcast_{month}.pgn.zst"
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest):
        print(f"{name}: already present")
        return dest
    os.makedirs(out_dir, exist_ok=True)
    tmp, h = dest + ".part", hashlib.sha256()
    with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            h.update(chunk)
    if name in sums and h.hexdigest() != sums[name]:
        os.remove(tmp)
        raise RuntimeError(f"{name}: sha256 mismatch")
    os.replace(tmp, dest)
    print(f"{name}: {os.path.getsize(dest) / 1e6:.1f} MB")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", nargs="*", default=[], help="e.g. 2026-01 2026-02")
    ap.add_argument("--from", dest="first")
    ap.add_argument("--to", dest="last")
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    months = list(a.months) + (month_range(a.first, a.last) if a.first and a.last else [])
    if not months or not all(re.fullmatch(r"\d{4}-\d{2}", m) for m in months):
        ap.error("give --months YYYY-MM ... or --from/--to")
    sums = fetch_checksums()
    for m in months:
        download(m, a.out, sums)


if __name__ == "__main__":
    main()
