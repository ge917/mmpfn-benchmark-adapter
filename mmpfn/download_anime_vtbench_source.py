"""Download the public Anime Dataset 2023 CSV used by VT-Bench.

Only the public CSV is downloaded here.  Its per-row ``Image URL`` values are
handled separately by :mod:`mmpfn.download_anime_vtbench_images`, so both
stages can be safely resumed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/raw/anime")
KAGGLE_HANDLE = "dbdmobile/myanimelist-dataset"
CSV_NAME = "anime-dataset-2023.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError as error:
        raise SystemExit("kagglehub is required; install it in the active environment first.") from error

    root = args.root.expanduser().resolve()
    destination = root / "source" / CSV_NAME
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Already present: {destination}")
        return

    root.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        kagglehub.dataset_download(KAGGLE_HANDLE, path=CSV_NAME, output_dir=str(root / "kaggle_cache"))
    )
    if downloaded.name != CSV_NAME:
        candidates = list(downloaded.rglob(CSV_NAME)) if downloaded.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"Kaggle download did not contain {CSV_NAME}: {downloaded}")
        downloaded = candidates[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, destination)
    print(f"Saved VT-Bench Anime source CSV: {destination}")


if __name__ == "__main__":
    main()
