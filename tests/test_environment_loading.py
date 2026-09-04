"""Component tests for local configuration loading at executable boundaries.

These tests use temporary, obviously synthetic values and never construct an LLM
client or contact NVIDIA. They isolate `.env` policy from provider behavior.
"""

from collections.abc import Callable
import os
from pathlib import Path

import pytest

from host.main import _load_local_environment as load_host_environment
from scripts.smoke_nvidia import (
    _load_local_environment as load_smoke_environment,
)


ENVIRONMENT_VARIABLE = "NVIDIA_API_KEY"
EnvironmentLoader = Callable[[Path], None]


@pytest.mark.parametrize(
    "load_environment",
    [load_host_environment, load_smoke_environment],
)
def test_local_env_provides_missing_nvidia_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_environment: EnvironmentLoader,
) -> None:
    """Each entry point can supply missing local-development configuration."""
    monkeypatch.delenv(ENVIRONMENT_VARIABLE, raising=False)
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=synthetic-dotenv-value\n",
        encoding="utf-8",
    )

    load_environment(tmp_path)

    assert os.environ[ENVIRONMENT_VARIABLE] == "synthetic-dotenv-value"


@pytest.mark.parametrize(
    "load_environment",
    [load_host_environment, load_smoke_environment],
)
def test_process_environment_takes_precedence_over_local_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_environment: EnvironmentLoader,
) -> None:
    """Explicit shell configuration wins because dotenv override is disabled."""
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "synthetic-shell-value")
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=synthetic-dotenv-value\n",
        encoding="utf-8",
    )

    load_environment(tmp_path)

    assert os.environ[ENVIRONMENT_VARIABLE] == "synthetic-shell-value"
