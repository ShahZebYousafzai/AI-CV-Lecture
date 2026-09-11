"""Turning whatever the client sent into a PIL image, safely.

Anything that touches raw user bytes is a place where a service falls over
in production, so every decode goes through here and every failure becomes
a clean 422 instead of a 500.
"""

from __future__ import annotations

import base64
import binascii
import io

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from app.core.errors import InvalidImageError, PayloadTooLargeError
from app.schemas.detection import ImagePrediction

# Palette reused for the annotated-image endpoint.
_COLORS = [
    (2, 128, 144),
    (2, 195, 154),
    (193, 91, 78),
    (18, 52, 59),
    (140, 230, 201),
    (90, 122, 128),
]


def decode_image(
    data: bytes,
    *,
    filename: str | None = None,
    max_bytes: int,
    max_pixels: int,
) -> Image.Image:
    """bytes -> RGB PIL image, with size and format guards."""
    if not data:
        raise InvalidImageError(f"Empty file: {filename or 'image'}")

    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"{filename or 'image'} is {len(data) / 1e6:.1f} MB, "
            f"limit is {max_bytes / 1e6:.1f} MB"
        )

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force a real decode now, not lazily mid-inference
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            f"Could not decode {filename or 'image'}: not a valid image file"
        ) from exc

    if image.width * image.height > max_pixels:
        raise PayloadTooLargeError(
            f"{filename or 'image'} is {image.width}x{image.height}, too large to process"
        )

    return image.convert("RGB")


def decode_base64_image(
    payload: str,
    *,
    filename: str | None = None,
    max_bytes: int,
    max_pixels: int,
) -> Image.Image:
    """Accepts a bare base64 string or a full `data:image/png;base64,...` URI."""
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(
            f"{filename or 'image'} is not valid base64"
        ) from exc

    return decode_image(raw, filename=filename, max_bytes=max_bytes, max_pixels=max_pixels)


def draw_predictions(image: Image.Image, prediction: ImagePrediction) -> Image.Image:
    """Draw boxes and labels. Handy for a live demo, not for machine clients."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # Pillow < 10.1 has no size argument
        font = ImageFont.load_default()

    for detection in prediction.detections:
        color = _COLORS[detection.class_id % len(_COLORS)]
        box = detection.box
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=3)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_box = draw.textbbox((box.x1, box.y1), label, font=font)
        pad = 3
        draw.rectangle(
            [
                text_box[0] - pad,
                text_box[1] - pad,
                text_box[2] + pad,
                text_box[3] + pad,
            ],
            fill=color,
        )
        draw.text((box.x1, box.y1), label, fill=(255, 255, 255), font=font)

    return canvas


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
