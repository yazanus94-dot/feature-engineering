import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from config.feature_config import FeatureConfig
from core import AutomatedFeatureEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sample_data():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "b": ["x", "y", "x", "y", "x", "y", "x", "y"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_package_imports():
    import core
    import config
    import pipelines
    import transformers
    import utils

    assert core is not None
    assert config is not None
    assert pipelines is not None
    assert transformers is not None
    assert utils is not None
    assert AutomatedFeatureEngine is not None


def test_feature_engine_initialization():
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    assert engine.target_column == "target"
    assert engine.task_type == "classification"
    assert engine.fitted_ is False


def test_feature_config_defaults():
    config = FeatureConfig()
    assert config.max_features == 100
    assert config.correlation_threshold == 0.95


def test_feature_engine_fit_transform():
    data = _sample_data()
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    result = engine.fit_transform(data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)


def test_numerical_feature_processing():
    data = _sample_data()
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    result = engine.fit_transform(data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    assert "a" in result.columns


def test_categorical_feature_processing():
    data = _sample_data()
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    result = engine.fit_transform(data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    assert "b" in result.columns


def test_feature_engine_handles_missing_values():
    data = _sample_data().copy()
    data.loc[0, "a"] = None
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    result = engine.fit_transform(data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)


def test_feature_engine_missing_target_raises():
    data = _sample_data().drop(columns=["target"])
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    with pytest.raises(ValueError):
        engine.fit_transform(data)


def test_invalid_input_handling():
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    with pytest.raises(AttributeError):
        engine.fit_transform(None)


def test_transform_before_fit():
    data = _sample_data()
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    with pytest.raises(RuntimeError, match="fit"):
        engine.transform(data)


def test_feature_engine_invalid_task_type_does_not_crash():
    data = _sample_data()
    engine = AutomatedFeatureEngine(target_column="target", task_type="unknown")
    result = engine.fit_transform(data)
    assert isinstance(result, pd.DataFrame)


def test_cli_help_execution():
    completed = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "usage" in completed.stdout.lower()


def test_synthetic_end_to_end_pipeline_execution(tmp_path):
    sample_path = tmp_path / "sample.csv"
    output_path = tmp_path / "output.csv"
    data = _sample_data()
    data.to_csv(sample_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--mode",
            "process",
            "--input",
            str(sample_path),
            "--target",
            "target",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    result = pd.read_csv(output_path)
    assert len(result) == len(data)
    assert "target" not in result.columns
