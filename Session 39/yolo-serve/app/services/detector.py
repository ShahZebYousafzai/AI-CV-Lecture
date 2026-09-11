"""The model wrapper. This is the only file that knows about Ultralytics.

Two rules make this production ready rather than demo ready:

1. The weights are loaded ONCE, at process startup, and held on this object.
   Loading inside a request handler would add hundreds of milliseconds to
   every call and would blow up memory under concurrency.

2. `predict` is a plain blocking function. It is never awaited directly.
   The API layer pushes it onto a worker thread so the event loop stays
   free to accept new connections while the CPU is busy.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings
from app.core.errors import ModelNotReadyError
from app.schemas.detection import BoundingBox, Detection, ImagePrediction

logger = logging.getLogger(__name__)


class Detector:
    """Holds one loaded YOLO model for the lifetime of the process."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._names: dict[int, str] = {}
        # Ultralytics keeps per-call state on the predictor object, so two
        # threads must not call predict on the same model at the same time.
        # Throughput comes from batching and from more workers, not from
        # more threads on one model.
        self._lock = threading.Lock()
        self.loaded_at: float | None = None

    # ---- lifecycle --------------------------------------------------------
    def load(self) -> None:
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(self.settings.torch_threads)

        path = Path(self.settings.model_path)
        if not path.exists():
            # Ultralytics resolves a bare name like "yolov8n.pt" by
            # downloading it. In a container we want that to have happened
            # at build time, so fail loudly instead of downloading here.
            raise FileNotFoundError(
                f"Model weights not found at {path}. "
                "Run `python scripts/download_model.py` first."
            )

        start = time.perf_counter()
        model = YOLO(str(path))
        model.to(self.settings.device)
        self._model = model
        self._names = {int(k): v for k, v in model.names.items()}
        logger.info(
            "Loaded %s on %s in %.2fs (%d classes)",
            path.name,
            self.settings.device,
            time.perf_counter() - start,
            len(self._names),
        )

        if self.settings.warmup:
            self._warmup()

        self.loaded_at = time.time()

    def _warmup(self) -> None:
        """First inference allocates buffers and is 5 to 10x slower.

        Paying that cost at startup means the first real user does not.
        """
        blank = Image.fromarray(
            np.zeros((self.settings.imgsz, self.settings.imgsz, 3), dtype=np.uint8)
        )
        start = time.perf_counter()
        self.predict([blank])
        logger.info("Warmup inference took %.0f ms", (time.perf_counter() - start) * 1000)

    def unload(self) -> None:
        self._model = None
        self._names = {}
        self.loaded_at = None
        logger.info("Model released")

    # ---- introspection ----------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    @property
    def model_name(self) -> str:
        return self.settings.model_file

    # ---- inference --------------------------------------------------------
    def predict(
        self,
        images: list[Image.Image],
        conf: float | None = None,
        iou: float | None = None,
        filenames: list[str | None] | None = None,
    ) -> tuple[list[ImagePrediction], float]:
        """Run one forward pass over a LIST of images.

        The single image endpoint calls this with a list of one, so there is
        exactly one inference code path to reason about and to test.

        What batching actually buys you, in order of size:

          * On a GPU, a lot. Batch size 1 leaves most of the device idle, so
            batching is where the 3x to 5x throughput gain lives.
          * One HTTP round trip instead of N. Over the public internet that
            saves a TLS handshake and N times the network latency, which is
            usually bigger than the inference time itself.
          * Per call Python and framework overhead paid once, not N times.
            Worth roughly 10 percent on CPU.

        It does NOT reduce the arithmetic. On a small CPU box that is
        already saturated at batch size 1, expect batching to be about
        break even. Measure with scripts/benchmark.py before you promise
        anyone a number.

        One sharp edge: Ultralytics only uses rectangular inference when
        every image in the batch has the same shape. Mix a 1080x810 and a
        1280x720 image and all of them get padded to a full square, which
        can double the cost per image. If your source images vary, resize
        them to a common shape before batching.

        Returns (predictions, inference_milliseconds).
        """
        if self._model is None:
            raise ModelNotReadyError("Model is still loading, retry shortly")

        conf = self.settings.conf_threshold if conf is None else conf
        iou = self.settings.iou_threshold if iou is None else iou
        filenames = filenames or [None] * len(images)

        start = time.perf_counter()
        with self._lock:
            raw_results = self._model.predict(
                source=images,
                imgsz=self.settings.imgsz,
                conf=conf,
                iou=iou,
                device=self.settings.device,
                verbose=False,
            )
        inference_ms = (time.perf_counter() - start) * 1000

        predictions = [
            self._to_prediction(raw, image, name)
            for raw, image, name in zip(raw_results, images, filenames)
        ]
        return predictions, inference_ms

    def _to_prediction(self, raw, image: Image.Image, filename: str | None) -> ImagePrediction:
        detections: list[Detection] = []
        boxes = getattr(raw, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), score, class_id in zip(xyxy, confs, classes):
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        class_name=self._names.get(int(class_id), str(class_id)),
                        confidence=round(float(score), 4),
                        box=BoundingBox(
                            x1=round(float(x1), 2),
                            y1=round(float(y1), 2),
                            x2=round(float(x2), 2),
                            y2=round(float(y2), 2),
                        ),
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return ImagePrediction(
            filename=filename,
            width=image.width,
            height=image.height,
            detection_count=len(detections),
            detections=detections,
        )
