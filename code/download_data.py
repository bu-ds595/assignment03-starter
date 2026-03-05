"""Download training and validation data for Assignment 3.

Run once:
    python download_data.py

This downloads train.npz and val.npz into the data/ directory.
"""

import os
import urllib.request

DATA_DIR = "data"
FILES = {
    "train.npz": "GOOGLE_DRIVE_FILE_ID_TRAIN",  # TODO: Replace with actual file IDs
    "val.npz": "GOOGLE_DRIVE_FILE_ID_VAL",
}


def download_gdrive(file_id, dest):
    """Download a file from Google Drive."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    print(f"Downloading {dest}...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"  Saved {dest} ({size_mb:.1f} MB)")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname, file_id in FILES.items():
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest):
            print(f"  {dest} already exists, skipping")
        else:
            download_gdrive(file_id, dest)
    print("Done!")


if __name__ == "__main__":
    main()
