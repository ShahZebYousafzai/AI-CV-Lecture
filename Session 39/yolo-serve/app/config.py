"""Application settings.

Every tunable value lives here and can be overridden by an environment
variable prefixed with APP_ (for example APP_CONF_THRESHOLD=0.4).

That single rule is what lets the same code run unchanged on a laptop,
inside Docker, and on an EC2 instance: only the environment changes.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# yolo-serve/app/config.py  ->  yolo-serve/
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        extra="ignore",
        protected_namespaces=(),  # we want to use model_* field names
    )

    # ---- service identity -------------------------------------------------
    app_name: str = "YOLOv8 Detection Service"
    version: str = "1.0.0"
    env: str = "local"  # local | docker | aws

    # ---- model ------------------------------------------------------------
    model_dir: Path = BASE_DIR / "models"
    model_file: str = "yolov8n.pt"
    device: str = "cpu"  # "cpu", "cuda:0", ...
    imgsz: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    warmup: bool = True

    # ---- request limits ---------------------------------------------------
    max_batch_size: int = 16
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB per image
    max_image_pixels: int = 50_000_000  # decompression-bomb guard

    # ---- runtime ----------------------------------------------------------
    torch_threads: int = 2  # intra-op threads per worker process
    inference_concurrency: int = 2  # threadpool slots reserved for inference
    job_ttl_seconds: int = 900  # how long finished async jobs are kept
    max_jobs: int = 256

    # ---- ops --------------------------------------------------------------
    log_level: str = "INFO"
    cors_origins: str = "*"
    enable_demo_routes: bool = False  # the deliberately blocking teaching route

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_file

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process."""
    return Settings()
