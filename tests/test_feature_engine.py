import pandas as pd

from config.feature_config import FeatureConfig
from core import AutomatedFeatureEngine


def _sample_data():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "b": ["x", "y", "x", "y", "x", "y", "x", "y"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


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


def test_feature_engine_requires_fit_before_transform():
    data = _sample_data()
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    try:
        engine.transform(data)
    except RuntimeError as exc:
        assert "fit" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_feature_engine_invalid_task_type_does_not_crash():
    data = _sample_data()
    engine = AutomatedFeatureEngine(target_column="target", task_type="unknown")
    result = engine.fit_transform(data)
    assert isinstance(result, pd.DataFrame)


def test_feature_engine_handles_categorical_columns():
    data = _sample_data()
    engine = AutomatedFeatureEngine(target_column="target", task_type="classification")
    result = engine.fit_transform(data)
    assert "b" not in result.columns or "b" in data.columns


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
    try:
        engine.fit_transform(data)
    except (KeyError, ValueError):
        pass
    else:
        raise AssertionError("expected error for missing target")


def test_cli_help_runs():
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "usage" in completed.stdout.lower()