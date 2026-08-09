"""Download only the MIMIC files needed to reconstruct VT-Bench LOS/RR.

The downloader writes exclusively below a user-owned output directory and
prompts for a current PhysioNet browser-cookie at runtime.  The cookie is not
written to disk.  Interrupted downloads retain only a ``.part`` file and are
resumed on the next invocation.
"""

from __future__ import annotations

import argparse
import getpass
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_OUTPUT = Path("/mnt/hdd/zhangyg/projects/tab/raw/mimic_iv_2_2")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
TIMEOUT = (20, 180)
MAX_RETRIES = 5

# These are the only raw tables referenced by VT-Bench's
# ``dataset/Constructed_datasets/built_regression.py`` for LOS and RR.
FILES: dict[str, tuple[str, Path]] = {
    "admissions": (
        "https://physionet.org/files/mimiciv/2.2/hosp/admissions.csv.gz",
        Path("hosp/admissions.csv.gz"),
    ),
    "patients": (
        "https://physionet.org/files/mimiciv/2.2/hosp/patients.csv.gz",
        Path("hosp/patients.csv.gz"),
    ),
    "transfers": (
        "https://physionet.org/files/mimiciv/2.2/hosp/transfers.csv.gz",
        Path("hosp/transfers.csv.gz"),
    ),
    "labevents": (
        "https://physionet.org/files/mimiciv/2.2/hosp/labevents.csv.gz",
        Path("hosp/labevents.csv.gz"),
    ),
    "chartevents": (
        "https://physionet.org/files/mimiciv/2.2/icu/chartevents.csv.gz",
        Path("icu/chartevents.csv.gz"),
    ),
    "cxr_metadata": (
        "https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz",
        Path("cxr/mimic-cxr-2.0.0-metadata.csv.gz"),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--files",
        nargs="+",
        choices=tuple(FILES),
        default=tuple(FILES),
        help="Subset to download; default downloads all six required files.",
    )
    return parser.parse_args()


def _download_one(session: requests.Session, cookie: str, url: str, destination: Path) -> None:
    """Resume ``destination.part`` where possible, then atomically publish it."""
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        start = partial.stat().st_size if partial.is_file() else 0
        headers = {"Cookie": cookie, "User-Agent": USER_AGENT}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise RuntimeError(f"unexpected content type: {content_type}")
                append = start > 0 and response.status_code == 206
                if not append:
                    start = 0
                total = response.headers.get("Content-Length")
                total_bytes = start + int(total) if total and total.isdigit() else None
                with partial.open("ab" if append else "wb") as handle, tqdm(
                    total=total_bytes,
                    initial=start,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            bar.update(len(chunk))
            if partial.stat().st_size < 1024:
                raise RuntimeError("download is unexpectedly small")
            partial.replace(destination)
            return
        except Exception as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {url}\n{error}") from error
            delay = 2 ** (attempt - 1)
            print(f"Attempt {attempt} failed: {error}; retrying in {delay}s.")
            time.sleep(delay)


def main() -> None:
    args = _parse_args()
    cookie = os.environ.get("PHYSIONET_COOKIE") or getpass.getpass("PhysioNet browser cookie: ")
    if len(cookie.strip()) < 20:
        raise ValueError("The PhysioNet cookie looks empty or incomplete.")
    output_root = args.output_root.expanduser().resolve()
    with requests.Session() as session:
        for key in args.files:
            url, relative_destination = FILES[key]
            _download_one(session, cookie, url, output_root / relative_destination)
    print(f"Finished required MIMIC table download under: {output_root}")


if __name__ == "__main__":
    main()
