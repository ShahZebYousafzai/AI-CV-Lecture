"""Side by side proof that `async def` alone does not save you.

Start the server with the demo routes on:

    APP_ENABLE_DEMO_ROUTES=true uvicorn app.main:app --port 8000

then run this. It fires the same batch at two endpoints that use the same
model on the same machine, and measures how fast /health/live is answered
while each one is working.

    /demo/predict-blocking   calls the model inside the async handler
    /predict/batch           pushes the model onto a worker thread
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def uploads(n: int):
    files = sorted(p for p in SAMPLES_DIR.glob("*.jpg"))
    if not files:
        raise SystemExit("No sample images. Run: python scripts/download_samples.py")
    blob = files[0].read_bytes()
    return [("files", (f"{i}.jpg", blob, "image/jpeg")) for i in range(n)]


async def probe_while(client: httpx.AsyncClient, path: str, payload) -> tuple[float, list[float]]:
    latencies: list[float] = []
    stop = asyncio.Event()

    async def ping():
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                await client.get("/health/live")
                latencies.append((time.perf_counter() - t0) * 1000)
            except httpx.HTTPError:
                latencies.append(float("inf"))
            await asyncio.sleep(0.02)

    pinger = asyncio.create_task(ping())
    start = time.perf_counter()
    response = await client.post(path, files=payload)
    elapsed = (time.perf_counter() - start) * 1000
    stop.set()
    await pinger

    if response.status_code == 404:
        raise SystemExit(
            "Demo route not found. Restart the server with APP_ENABLE_DEMO_ROUTES=true"
        )
    response.raise_for_status()
    return elapsed, latencies


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=180.0) as client:
        # A single connection pool with room for the health pings.
        for label, path in (
            ("BLOCKING  (model inside the async handler)", "/demo/predict-blocking"),
            ("THREADED  (model on a worker thread)", "/predict/batch"),
        ):
            elapsed, latencies = await probe_while(client, path, uploads(args.n))
            finite = [x for x in latencies if x != float("inf")]
            print(f"\n{label}")
            print(f"  inference request took   : {elapsed:7.0f} ms")
            print(f"  health checks answered   : {len(finite)}")
            if finite:
                print(f"  health latency median    : {statistics.median(finite):7.1f} ms")
                print(f"  health latency worst     : {max(finite):7.1f} ms")

        print("\nSame model, same machine. The worst case health latency is the "
              "number your load balancer sees.")


if __name__ == "__main__":
    asyncio.run(main())
