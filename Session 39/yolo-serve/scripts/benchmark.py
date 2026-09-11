"""Four measurements that turn opinions about serving into numbers.

Run this against a live server during the session:

    python scripts/benchmark.py --url http://localhost:8000 --n 8

  1. N singles vs one batch, all images the SAME size.
  2. The same batch with MIXED sizes. This one surprises people.
  3. Health checks fired while the model is busy. Proof the event loop
     is not blocked.
  4. The async job endpoint: 202 immediately, poll until done.

Do not trust the numbers in the slides. Run this on the machine you are
deploying to, because the answer depends on CPU cores, image size and
whether you have a GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"


def load_samples() -> list[tuple[str, bytes]]:
    files = sorted(
        p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise SystemExit("No sample images. Run: python scripts/download_samples.py")
    return [(p.name, p.read_bytes()) for p in files]


def as_upload(index: int, name: str, blob: bytes, field: str = "files"):
    return (field, (f"{index}_{name}", blob, "image/jpeg"))


async def run_singles(client: httpx.AsyncClient, images) -> tuple[float, float]:
    """Returns (wall_ms, summed model time)."""
    model_ms = 0.0
    start = time.perf_counter()
    for i, (name, blob) in enumerate(images):
        response = await client.post(
            "/predict", files={"file": (f"{i}_{name}", blob, "image/jpeg")}
        )
        response.raise_for_status()
        model_ms += response.json()["inference_ms"]
    return (time.perf_counter() - start) * 1000, model_ms


async def run_batch(client: httpx.AsyncClient, images) -> tuple[float, float]:
    payload = [as_upload(i, name, blob) for i, (name, blob) in enumerate(images)]
    start = time.perf_counter()
    response = await client.post("/predict/batch", files=payload)
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000, response.json()["inference_ms"]


async def run_event_loop_probe(client: httpx.AsyncClient, images) -> list[float]:
    latencies: list[float] = []
    stop = asyncio.Event()

    async def ping():
        while not stop.is_set():
            t0 = time.perf_counter()
            await client.get("/health/live")
            latencies.append((time.perf_counter() - t0) * 1000)
            await asyncio.sleep(0.02)

    pinger = asyncio.create_task(ping())
    await run_batch(client, images)
    stop.set()
    await pinger
    return latencies


async def run_job(client: httpx.AsyncClient, images) -> None:
    payload = [as_upload(i, name, blob) for i, (name, blob) in enumerate(images)]
    start = time.perf_counter()
    submit = await client.post("/jobs", files=payload)
    submit.raise_for_status()
    job_id = submit.json()["job_id"]
    print(f"  POST /jobs returned {submit.status_code} in "
          f"{(time.perf_counter() - start) * 1000:.0f} ms, job {job_id[:8]}")

    status: dict = {}
    while True:
        await asyncio.sleep(0.25)
        status = (await client.get(f"/jobs/{job_id}")).json()
        if status["status"] in {"succeeded", "failed"}:
            break

    total_ms = (time.perf_counter() - start) * 1000
    found = sum(r["detection_count"] for r in (status.get("result") or {}).get("results", []))
    print(f"  '{status['status']}' after {total_ms:.0f} ms, {found} detections total")


def report(label: str, wall_ms: float, model_ms: float, n: int) -> None:
    print(f"  {label:<34} wall {wall_ms:7.0f} ms | model {model_ms:7.0f} ms "
          f"| {wall_ms / n:6.1f} ms/image")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=8, help="images per batch")
    args = parser.parse_args()

    samples = load_samples()
    first_name, first_blob = samples[0]
    same_size = [(first_name, first_blob)] * args.n
    mixed = [samples[i % len(samples)] for i in range(args.n)]
    is_mixed = len({blob for _, blob in mixed}) > 1

    async with httpx.AsyncClient(base_url=args.url, timeout=180.0) as client:
        print(f"server: {(await client.get('/health/ready')).json()}\n")
        await run_batch(client, same_size[:1])  # warm the path

        print(f"1. Same size images, n={args.n}")
        report("N separate /predict calls", *await run_singles(client, same_size), args.n)
        report("one /predict/batch call", *await run_batch(client, same_size), args.n)
        print()

        if is_mixed:
            print(f"2. Mixed size images, n={args.n}")
            report("N separate /predict calls", *await run_singles(client, mixed), args.n)
            report("one /predict/batch call", *await run_batch(client, mixed), args.n)
            print("  A batch of differently sized images is padded to a full square,\n"
                  "  so per image cost goes UP. Resize before you batch.\n")

        print("3. Event loop responsiveness while the model is busy")
        latencies = await run_event_loop_probe(client, same_size)
        print(f"  {len(latencies)} health checks answered during inference: "
              f"median {statistics.median(latencies):.1f} ms, max {max(latencies):.1f} ms")
        print("  Blocking handlers would have stalled every one of these.\n")

        print("4. Async job endpoint")
        await run_job(client, same_size)


if __name__ == "__main__":
    asyncio.run(main())
