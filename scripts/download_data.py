#!/usr/bin/env python3
"""Download EDS-lab/electricity-demand from Hugging Face into data/raw/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ID = "EDS-lab/electricity-demand"
FILES = ("demand.parquet", "metadata.parquet", "weather.parquet")

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"


def _check_login() -> None:
    from huggingface_hub import whoami

    try:
        info = whoami()
        print(f"Logged in as: {info.get('name', info)}")
    except Exception:
        print(
            "Chua dang nhap Hugging Face.\n"
            "1. Mo https://huggingface.co/datasets/EDS-lab/electricity-demand\n"
            "2. Dang nhap va bam Accept terms (dataset gated)\n"
            "3. Chay: hf auth login\n"
            "4. Chay lai: python scripts/download_data.py"
        )
        sys.exit(1)


def _resolve_existing() -> bool:
    """Return True if all parquet files already exist in data/raw/."""
    return all((DATA_RAW / name).exists() for name in FILES)


def _normalize_paths() -> None:
    """Move data/raw/data/*.parquet -> data/raw/*.parquet if needed."""
    nested = DATA_RAW / "data"
    if not nested.exists():
        return
    for name in FILES:
        src = nested / name
        dst = DATA_RAW / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def download() -> None:
    from huggingface_hub import snapshot_download

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_RAW),
        allow_patterns=["data/*.parquet"],
    )
    _normalize_paths()


def main() -> None:
    if _resolve_existing():
        print("Dataset da co san trong data/raw/")
        for name in FILES:
            print(f"  - {DATA_RAW / name}")
        return

    _check_login()

    try:
        download()
    except Exception as exc:
        err = str(exc)
        if "GatedRepoError" in err or "401" in err:
            print(
                "\nKhong co quyen truy cap dataset gated.\n"
                "Hay mo https://huggingface.co/datasets/EDS-lab/electricity-demand\n"
                "va bam Accept terms, sau do chay lai script."
            )
        raise

    missing = [name for name in FILES if not (DATA_RAW / name).exists()]
    if missing:
        print(f"Thieu file sau khi download: {missing}")
        sys.exit(1)

    print("Download thanh cong:")
    for name in FILES:
        path = DATA_RAW / name
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
