"""Fetch the weights into models/ so the app never downloads at runtime.

Run once locally, and again inside the Docker build. A container that
downloads its model on first request is slow to start, depends on the
internet from inside your VPC, and downloads again on every restart.

This deliberately fetches the file over plain HTTP rather than importing
Ultralytics. Importing Ultralytics pulls in OpenCV, which needs system
shared libraries (libGL, libxcb) that a lean build stage does not have.
Keeping the download dependency-free keeps the build stage lean and makes
the URL explicit and auditable, which is what you want for an artefact that
ends up baked into a production image.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FILE = os.environ.get("APP_MODEL_FILE", "yolov8n.pt")
RELEASE = os.environ.get("YOLO_ASSETS_RELEASE", "latest")

MIRRORS = [
    f"https://github.com/ultralytics/assets/releases/{RELEASE}/download/{MODEL_FILE}",
    f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{MODEL_FILE}",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / MODEL_FILE

    if target.exists():
        print(f"[ok] {target} already present ({target.stat().st_size / 1e6:.1f} MB)")
        return 0

    for url in MIRRORS:
        try:
            print(f"[..] fetching {MODEL_FILE} from {url}")
            urllib.request.urlretrieve(url, target)
            break
        except Exception as exc:  # noqa: BLE001 - mirrors are best effort
            print(f"[!!] {exc}")
            target.unlink(missing_ok=True)
    else:
        print("[!!] every mirror failed. Falling back to the Ultralytics downloader.")
        try:
            import shutil

            from ultralytics import YOLO

            model = YOLO(MODEL_FILE)
            source = Path(getattr(model, "ckpt_path", MODEL_FILE))
            shutil.move(str(source), target)
        except Exception as exc:  # noqa: BLE001
            print(f"[!!] fallback failed too: {exc}")
            return 1

    size_mb = target.stat().st_size / 1e6
    if size_mb < 1:
        print(f"[!!] {target} is only {size_mb:.2f} MB, that is not the weights file")
        return 1

    print(f"[ok] saved {target} ({size_mb:.1f} MB)")
    print(f"[ok] sha256 {sha256(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
