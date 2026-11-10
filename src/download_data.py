"""Download the Hetionet files listed in config.json into the raw data folder.

Usage:
    python src/download_data.py [--config config.json]
"""

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

from config import load_config, raw_files


class Progress:
    """Print download progress only when the whole percentage changes."""

    def __init__(self):
        self.last = -1

    def __call__(self, blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, blocks * block_size * 100 // total)
        if pct != self.last and pct % 10 == 0:
            print(f"  {pct:3d}% of {total / 1e6:.1f} MB")
            self.last = pct


def check(path: Path, url: str) -> None:
    """Fail if we got a Git LFS pointer or an HTML page instead of data."""
    with open(path, "rb") as f:
        head = f.read(200)
    if head.startswith(b"version https://git-lfs") or head.lstrip().startswith(b"<!"):
        path.unlink()
        sys.exit(f"{path.name} is not a data file. Download it manually from {url}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            f.readline()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to config file (default: config.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_url = cfg["data"]["base_url"]
    local = raw_files(cfg)

    for key, remote in cfg["data"]["files"].items():
        target = local[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            print(f"{target.name}: already downloaded, skipping")
            continue
        url = base_url + remote
        print(f"Downloading {url}")
        tmp = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(url, tmp, Progress())
        tmp.replace(target)
        check(target, url)

    print(f"Done. Files are in {next(iter(local.values())).parent}")


if __name__ == "__main__":
    main()
