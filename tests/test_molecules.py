import os

from common.helpers import load_config
from analysis.tools.molecules.quality_monitor import QualityMonitoringSystem
from analysis.tools.molecules.llm_integration import LocalKnowledgeRAG


def test_load_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("expectations_dir: exp\nvalidation_dir: val")
    cfg = load_config(str(cfg_path))
    assert cfg == {"expectations_dir": "exp", "validation_dir": "val"}


def test_quality_monitor_init(tmp_path):
    config = tmp_path / "cfg.yaml"
    exp_dir = tmp_path / "exp"
    val_dir = tmp_path / "val"
    config.write_text(f"expectations_dir: {exp_dir}\nvalidation_dir: {val_dir}")
    qm = QualityMonitoringSystem(str(config))
    assert os.path.isdir(exp_dir)
    assert os.path.isdir(val_dir)
    assert qm.expectations_dir == str(exp_dir)
    assert qm.validation_dir == str(val_dir)


def test_helper_removed():
    assert not hasattr(QualityMonitoringSystem, "_load_config")
    assert not hasattr(LocalKnowledgeRAG, "_load_config")

