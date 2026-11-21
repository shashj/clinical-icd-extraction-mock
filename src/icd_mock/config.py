"""Settings and the Langfuse client."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langfuse import Langfuse

DEFAULT_HOST = "http://localhost:3000"


@dataclass(frozen=True)
class Settings:
    host: str
    public_key: str | None
    secret_key: str | None

    @property
    def has_keys(self) -> bool:
        return bool(self.public_key and self.secret_key)


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        host=os.environ.get("LANGFUSE_HOST", DEFAULT_HOST).rstrip("/"),
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    )


def make_client(settings: Settings) -> Langfuse:
    """A Langfuse client that sends data to the configured instance."""
    if not settings.has_keys:
        raise SystemExit(
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (see .env.example), "
            "or use --dry-run to run without Langfuse."
        )
    return Langfuse(
        public_key=settings.public_key,
        secret_key=settings.secret_key,
        host=settings.host,
    )


def make_offline_client() -> Langfuse:
    """A client that records nothing and never touches the network."""
    return Langfuse(
        public_key="pk-offline",
        secret_key="sk-offline",
        host=DEFAULT_HOST,
        tracing_enabled=False,
    )
