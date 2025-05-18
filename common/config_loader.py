"""중앙 설정 로더 – YAML + .env 병합."""
import os
import yaml
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]

def load(name: str):
    cfg = yaml.safe_load((_ROOT / "config" / f"{name}.yaml").read_text())
    for k, v in cfg.items():
        cfg[k] = os.getenv(k.upper(), v)
    return cfg
