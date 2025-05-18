import logging
from typing import Any, Dict, cast

import yaml  # type: ignore

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return cast(Dict[str, Any], yaml.safe_load(file))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load config from %s: %s", config_path, exc)
        raise
