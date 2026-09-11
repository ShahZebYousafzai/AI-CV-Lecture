"""Grab a few test photos into samples/ so the demo has something to detect."""

import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"

ASSETS = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets"

SAMPLES = {
    "bus.jpg": [f"{ASSETS}/bus.jpg", "https://ultralytics.com/images/bus.jpg"],
    "zidane.jpg": [f"{ASSETS}/zidane.jpg", "https://ultralytics.com/images/zidane.jpg"],
}


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, urls in SAMPLES.items():
        target = SAMPLES_DIR / name
        if target.exists():
            print(f"[ok] {name} already present")
            continue

        for url in urls:
            try:
                print(f"[..] {name} from {url}")
                urllib.request.urlretrieve(url, target)
                print(f"[ok] {name} ({target.stat().st_size / 1e3:.0f} KB)")
                break
            except Exception as exc:  # noqa: BLE001 - mirrors are best effort
                print(f"[!!] {exc}")
        else:
            print(f"[!!] could not fetch {name}, drop any JPEG into samples/ instead")
    return 0


if __name__ == "__main__":
    sys.exit(main())
