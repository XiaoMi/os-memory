# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import os
from typing import Any


def model_spec(config: dict[str, Any], key: str = "model") -> dict[str, Any]:
    value = config.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def model_name(config: dict[str, Any], default: str, key: str = "model") -> str:
    spec = model_spec(config, key)
    value = spec.get("name") or spec.get("model") or config.get("model_name")
    return str(value or default)


def api_key(
    config: dict[str, Any],
    default_env: str,
    key: str = "model",
    required: bool = False,
) -> str | None:
    spec = model_spec(config, key)
    env_name = str(spec.get("api_key_env") or config.get("api_key_env") or default_env)
    value = os.getenv(env_name)
    if required and not value:
        raise RuntimeError(
            f"Missing model API key. Set environment variable {env_name!r}; "
            "API keys must not be stored in benchmark config files."
        )
    return str(value) if value else None


def base_url(config: dict[str, Any], default_env: str = "OPENAI_BASE_URL", key: str = "model") -> str | None:
    spec = model_spec(config, key)
    value = spec.get("base_url") or config.get("base_url")
    env_name = str(spec.get("base_url_env") or config.get("base_url_env") or default_env)
    return str(value or os.getenv(env_name)) if (value or os.getenv(env_name)) else None
