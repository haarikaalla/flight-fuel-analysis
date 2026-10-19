"""Download and verify the three large published trajectory archives.

Interrupted downloads resume via HTTP Range. All archives live in ignored data/raw.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from .data_audit import RECORD, checksum

ARCHIVES = {
    "flights_train.zip": (3300971911, "255cef578697dc79d5d98958b8a880f5"),
    "flights_rank.zip": (631703907, "f8c7af3510d5ffd0d5662866d12aa760"),
    "flights_final.zip": (982763659, "517df5d6210aad1286900611963345f4"),
}


def download(raw_dir: Path, name: str) -> str:
    size, expected = ARCHIVES[name]
    path = raw_dir / name
    if path.exists() and path.stat().st_size == size and checksum(path) == expected:
        return f"Verified existing {name}"
    partial = raw_dir / (name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    url = f"{RECORD}/{name}/content"
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urlopen(Request(url, headers=headers), timeout=180) as source:
        if offset and source.status != 206:
            raise RuntimeError(f"Server ignored resume range for {name}; remove {partial} before retry")
        with partial.open("ab" if offset else "wb") as output:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                output.write(block)
    if partial.stat().st_size != size or checksum(partial) != expected:
        raise ValueError(f"Incomplete or MD5-invalid {name}; check network and retry")
    partial.replace(path)
    return f"Downloaded and verified {name} ({size} bytes)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for message in pool.map(lambda name: download(args.raw_dir, name), ARCHIVES):
            print(message, flush=True)


if __name__ == "__main__":
    main()
