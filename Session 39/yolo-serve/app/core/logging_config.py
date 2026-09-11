"""One-line JSON-ish logging, configured once at startup.

Cloud log collectors (CloudWatch, Loki, Datadog) read stdout. So we log to
stdout and never to a file: the container must stay stateless.
"""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:  # uvicorn already installed handlers on reload
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Ultralytics is chatty on every single predict call.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
