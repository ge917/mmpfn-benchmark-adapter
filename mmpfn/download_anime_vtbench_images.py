"""Resumably download VT-Bench Anime images referenced by the public CSV."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


DEFAULT_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/raw/anime")


def _valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _download(anime_id: int, url: str, image_root: Path, timeout: float) -> tuple[int, str | None]:
    destination = image_root / f"{anime_id}.jpg"
    if _valid_image(destination):
        return anime_id, None
    temporary = destination.with_suffix(".jpg.part")
    try:
        response = requests.get(url, timeout=(10, timeout), stream=True)
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    output.write(chunk)
        if not _valid_image(temporary):
            raise ValueError("response is not a valid image")
        temporary.replace(destination)
        return anime_id, None
    except Exception as error:  # One unavailable CDN image should not stop a resumable download.
        temporary.unlink(missing_ok=True)
        return anime_id, str(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0, help="0 downloads every outstanding image.")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    root = args.root.expanduser().resolve()
    source = root / "source" / "anime-dataset-2023.csv"
    image_root = root / "source" / "images"
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source}")
    data = pd.read_csv(source, usecols=["anime_id", "Image URL"])
    data = data.dropna(subset=["anime_id", "Image URL"])
    data = data[data["Image URL"].astype(str).ne("Unknown")]
    image_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        (int(anime_id), str(url))
        for anime_id, url in zip(data["anime_id"], data["Image URL"])
        if not _valid_image(image_root / f"{int(anime_id)}.jpg")
    ]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"Referenced Anime images: {len(data)}; JPEGs to download: {len(jobs)}")
    failures: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_download, identifier, url, image_root, args.timeout) for identifier, url in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading Anime images", unit="image"):
            identifier, error = future.result()
            if error:
                failures.append((identifier, error))
    if failures:
        report = root / "anime_image_download_failures.tsv"
        report.write_text("\n".join(f"{key}\t{error}" for key, error in failures) + "\n", encoding="utf-8")
        print(f"Finished with {len(failures)} failures; details: {report}")
    else:
        print("Finished successfully.")


if __name__ == "__main__":
    main()
