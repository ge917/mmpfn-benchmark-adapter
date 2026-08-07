"""Download the MIMIC-CXR-JPG files referenced by the prepared VT-Bench splits.

The script deliberately downloads JPEGs only.  The original task preprocessors
then make the 224x224 uint8 ``.npy`` files and refresh the corresponding
``*_paths.pt`` files without changing the train/validation/test definitions.

Authentication is read at runtime, never stored in this file:

    read -rsp 'PhysioNet Cookie: ' PHYSIONET_COOKIE; echo
    export PHYSIONET_COOKIE
    python mmpfn/download_mimic_cxr_jpg.py --tasks pneumonia rr
    unset PHYSIONET_COOKIE
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import os
import threading
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
import torch
from PIL import Image, ImageFile
from tqdm import tqdm


MIMIC_ROOT = Path("/mnt/hdd/jiazy/mimic")  # read-only source supplied by the senior student
PROJECT_RAW_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/raw/mimic")
TASK_FEATURES = {
    "pneumonia": (MIMIC_ROOT / "classification" / "features", ("train", "valid", "test")),
    "rr": (MIMIC_ROOT / "regression" / "rr" / "features", ("train", "val", "test")),
}
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/files"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
MIN_BYTES = 1024
TIMEOUT = (15, 120)
MAX_RETRIES = 5

# Reject truncated/corrupt JPEGs rather than silently retaining them.
ImageFile.LOAD_TRUNCATED_IMAGES = False


def normalise_relative_path(value: object) -> Path:
    """Turn CSV image paths into ``p10/...jpg`` relative paths."""
    path = str(value).strip().replace("\\", "/")
    image_marker = "/image/"
    if image_marker in path:
        path = path.split(image_marker, 1)[1]
    marker = "/files/"
    if marker in path:
        path = path.split(marker, 1)[1]
    elif path.startswith("files/"):
        path = path[len("files/") :]
    return Path(path.lstrip("/")).with_suffix(".jpg")


def jpeg_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < MIN_BYTES:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def collect_paths(tasks: Iterable[str]) -> set[Path]:
    paths: set[Path] = set()
    for task in tasks:
        features_dir, splits = TASK_FEATURES[task]
        for split in splits:
            paths_file = features_dir / f"{split}_paths.pt"
            if not paths_file.is_file():
                raise FileNotFoundError(f"Missing {task} split paths: {paths_file}")
            paths.update(normalise_relative_path(value) for value in torch.load(paths_file, map_location="cpu"))
    return paths


def download_one(session: requests.Session, cookie: str, relative: Path, destination: Path) -> str | None:
    """Download a JPEG atomically.  Return an error message, or None on success."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    url = f"{BASE_URL}/{quote(relative.as_posix(), safe='/')}"
    headers = {"Cookie": cookie, "User-Agent": USER_AGENT}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise RuntimeError(f"unexpected content type: {content_type}")
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if not jpeg_is_valid(temporary):
                raise RuntimeError("downloaded file is not a valid JPEG")
            temporary.replace(destination)
            return None
        except Exception as exc:  # retry transient connection/server errors
            temporary.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                return str(exc)
            time.sleep(1.5 ** (attempt - 1))
    return "unreachable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_FEATURES), required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=PROJECT_RAW_ROOT / "images",
        help="Writable JPEG cache. Defaults to this user's project, never the source dataset.",
    )
    parser.add_argument("--limit", type=int, help="Download at most N missing images (smoke test).")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent image downloads (default: 8; use 1 for sequential).",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=PROJECT_RAW_ROOT / "download_failures_live.tsv",
        help="Writable per-run failure log, updated while the download is running.",
    )
    args = parser.parse_args()

    cookie = os.environ.get("PHYSIONET_COOKIE") or getpass.getpass("PhysioNet browser cookie: ")
    if len(cookie.strip()) < 20:
        raise ValueError("The PhysioNet cookie looks empty or incomplete.")

    all_paths = sorted(collect_paths(args.tasks))
    missing = [path for path in all_paths if not jpeg_is_valid(args.image_root / path)]
    if args.limit is not None:
        missing = missing[: args.limit]
    print(f"Referenced images: {len(all_paths)}; JPEGs to download: {len(missing)}")
    if not missing:
        return

    failures: list[str] = []
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    args.failure_log.parent.mkdir(parents=True, exist_ok=True)
    # Clear only this user's previous transient monitor log before a new run.
    args.failure_log.write_text("", encoding="utf-8")

    thread_state = threading.local()

    def worker(relative: Path) -> tuple[Path, str | None]:
        # Reuse one keep-alive Session per executor worker.  Creating a fresh TLS
        # connection for every image overloads the proxy at higher concurrency.
        session = getattr(thread_state, "session", None)
        if session is None:
            session = requests.Session()
            thread_state.session = session
        return relative, download_one(session, cookie, relative, args.image_root / relative)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(worker, missing)
        for relative, error in tqdm(
            results, total=len(missing), unit="image", desc="Downloading MIMIC-CXR-JPG"
        ):
            if error is not None:
                line = f"{relative}\t{error}"
                failures.append(line)
                with args.failure_log.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                if len(failures) <= 5 or len(failures) % 50 == 0:
                    tqdm.write(f"WARNING: {len(failures)} failed requests; latest: {error}")

    if failures:
        print(f"Finished with {len(failures)} failures; details: {args.failure_log}")
    else:
        print("Finished successfully.")


if __name__ == "__main__":
    main()
