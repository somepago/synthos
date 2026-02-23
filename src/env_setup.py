"""
Load .env file to set model cache path and download source.

Import this module before any diffsynth imports to ensure the environment
is configured. Already-set environment variables take precedence.

Usage:
    import env_setup  # noqa: F401 — side-effect import
    from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")

if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
