"""Domain exceptions plus the handlers that turn them into JSON.

Routes raise a domain error. They never build an HTTP response by hand.
That keeps the error contract identical across every endpoint.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for errors we expect and can describe to the caller.

    Plain integers rather than the `fastapi.status` constants: Starlette has
    renamed several of them (422 and 413 among others) and the old names now
    emit deprecation warnings on import. Numbers do not churn.
    """

    status_code = 400
    code = "app_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidImageError(AppError):
    status_code = 422
    code = "invalid_image"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class BatchTooLargeError(AppError):
    status_code = 413
    code = "batch_too_large"


class ModelNotReadyError(AppError):
    status_code = 503
    code = "model_not_ready"


class JobNotFoundError(AppError):
    status_code = 404
    code = "job_not_found"


def _payload(code: str, message: str, request: Request) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message, request),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_payload("validation_error", str(exc.errors()), request),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Never leak a stack trace to the caller. It is already in the logs.
        return JSONResponse(
            status_code=500,
            content=_payload("internal_error", "Internal server error", request),
        )
