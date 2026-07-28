"""Construct optional OpenAI clients without coupling the core runtime to the SDK."""

import math
import os


_MISSING_API_KEY_MESSAGE = (
    "api_key must be provided explicitly or through OPENAI_API_KEY"
)


def _load_openai_class() -> object:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as error:
        if error.name == "openai":
            raise ImportError(
                "OpenAI SDK is required; install it with "
                "`pip install -r requirements-openai.txt`."
            ) from error
        raise
    return OpenAI


def _resolve_api_key(api_key: str | None) -> str:
    if api_key is not None:
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string or None")
        resolved_api_key = api_key
    else:
        resolved_api_key = os.environ.get("OPENAI_API_KEY")

    if resolved_api_key is None or not resolved_api_key.strip():
        raise ValueError(_MISSING_API_KEY_MESSAGE)

    return resolved_api_key


def _validate_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    if not isinstance(base_url, str):
        raise TypeError("base_url must be a string or None")
    if not base_url.strip():
        raise ValueError("base_url must not be blank")
    return base_url


def _validate_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be an int, float, or None")

    validated_timeout = float(timeout)
    if not math.isfinite(validated_timeout) or validated_timeout <= 0:
        raise ValueError("timeout must be finite and greater than zero")

    return validated_timeout


def create_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> object:
    """Create an OpenAI client using explicit, validated construction options."""

    resolved_api_key = _resolve_api_key(api_key)
    validated_base_url = _validate_base_url(base_url)
    validated_timeout = _validate_timeout(timeout)
    openai_class = _load_openai_class()

    constructor_kwargs: dict[str, object] = {
        "api_key": resolved_api_key,
        "max_retries": 0,
    }
    if validated_base_url is not None:
        constructor_kwargs["base_url"] = validated_base_url
    if validated_timeout is not None:
        constructor_kwargs["timeout"] = validated_timeout

    return openai_class(**constructor_kwargs)
