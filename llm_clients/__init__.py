"""Optional provider client construction."""

from .openai_client_factory import create_openai_client

globals().pop("openai_client_factory", None)

__all__ = ["create_openai_client"]
