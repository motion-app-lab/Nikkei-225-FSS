from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .common import ServiceError
from .nikkei_dual_market import (
    FEATURE_DEFINITION_VERSION,
    MODEL_ARTIFACT_PATH,
    MODEL_VERSION,
    SETTINGS_DIR,
)

ARTIFACT_SCHEMA_VERSION = "nikkei_portable_artifact_v1"
EVALUATION_SCHEMA_VERSION = "nikkei_context_evaluation_v1"
MANIFEST_PATH = SETTINGS_DIR / "nikkei_model_manifest.json"
EVALUATION_PATHS = {
    "intraday": SETTINGS_DIR / "nikkei_intraday_evaluation.json",
    "after_close": SETTINGS_DIR / "nikkei_after_close_evaluation.json",
}
MODEL_COMPATIBILITY_ERROR = (
    "保存済みの日経平均予測モデルと、現在の実行環境に互換性がありません。"
    "付属のセットアップ手順で環境を再構築してください。"
)
PINNED_DISTRIBUTIONS = (
    "fastapi",
    "uvicorn",
    "pandas",
    "numpy",
    "yfinance",
    "scikit-learn",
    "catboost",
    "joblib",
    "matplotlib",
    "httpx",
    "pytest",
    "holidays",
)
MODEL_RUNTIME_DISTRIBUTIONS = ("pandas", "numpy", "scikit-learn", "catboost", "joblib")


def runtime_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in PINNED_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    return {"python": platform.python_version(), "packages": packages}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        portable = _portable_metadata(payload)
        temporary.write_text(json.dumps(portable, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _portable_metadata(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return {str(key): _portable_metadata(item) for key, item in value.items()}
    if isinstance(value, pd.Index):
        return [_portable_metadata(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_portable_metadata(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _portable_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_portable_metadata(item) for item in value]
    if isinstance(value, list):
        return [_portable_metadata(item) for item in value]
    return value


def make_packages_portable(packages: dict[str, Any]) -> dict[str, Any]:
    """モデル本体は維持し、周辺メタデータだけを移植可能な標準型へ変換する。"""
    return _portable_metadata(packages)


def _assert_portable_package(package: Any, path: str = "root") -> None:
    if isinstance(package, (pd.Series, pd.Index, pd.DataFrame, pd.Timestamp, pd.api.extensions.ExtensionDtype)):
        raise ValueError(f"pandas-dependent metadata remains at {path}: {type(package).__name__}")
    if isinstance(package, np.ndarray) and package.dtype == object:
        raise ValueError(f"object ndarray remains at {path}")
    if isinstance(package, dict):
        for key, value in package.items():
            if key in {"model", "scaler"}:
                continue
            _assert_portable_package(value, f"{path}.{key}")
    elif isinstance(package, (list, tuple)):
        for index, value in enumerate(package):
            _assert_portable_package(value, f"{path}[{index}]")


def _feature_schema(packages: dict[str, Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for context, payload in packages["contexts"].items():
        schema[context] = {
            "direction": {
                side: [str(item) for item in payload[side]["selected_features"]]
                for side in ("japan", "overseas")
            },
            "six_class": [str(item) for item in payload["six_class"]["features"]],
        }
    return schema


def _atomic_dump_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        joblib.dump(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_portable_artifacts(packages: dict[str, Any], evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    portable = make_packages_portable(packages)
    _assert_portable_package(portable)
    _atomic_dump_joblib(MODEL_ARTIFACT_PATH, portable)
    for context, evaluation_path in EVALUATION_PATHS.items():
        if context not in evaluations:
            raise ValueError(f"missing evaluation context: {context}")
        atomic_write_json(evaluation_path, evaluations[context])
    versions = runtime_versions()
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "python_version": versions["python"],
        "compatible_python_versions": [versions["python"]],
        "library_versions": versions["packages"],
        "model_schema_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_DEFINITION_VERSION,
        "model_file": MODEL_ARTIFACT_PATH.name,
        "model_sha256": sha256_file(MODEL_ARTIFACT_PATH),
        "prediction_contexts": sorted(portable["contexts"]),
        "feature_schema": _feature_schema(portable),
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_files": {
            context: {"file": path.name, "sha256": sha256_file(path)}
            for context, path in EVALUATION_PATHS.items()
        },
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    return manifest


def load_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデルのmanifestを確認できませんでした。", 500) from error
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデルの保存形式が一致しません。", 409)
    if manifest.get("model_schema_version") != MODEL_VERSION or manifest.get("feature_schema_version") != FEATURE_DEFINITION_VERSION:
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデルのschemaが一致しません。", 409)
    if sorted(manifest.get("prediction_contexts", [])) != ["after_close", "intraday"]:
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "必要な場中・大引け後モデルが揃っていません。", 409)
    if manifest.get("model_file") != MODEL_ARTIFACT_PATH.name:
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデルのファイル名がmanifestと一致しません。", 409)
    return manifest


def _validate_runtime(manifest: dict[str, Any]) -> None:
    current = runtime_versions()
    expected_python = str(manifest.get("python_version", ""))
    compatible_python_versions = manifest.get("compatible_python_versions")
    if isinstance(compatible_python_versions, list):
        allowed_python_versions = {str(version) for version in compatible_python_versions}
    else:
        allowed_python_versions = {expected_python}
    compatible_python_series = manifest.get("compatible_python_series")
    if isinstance(compatible_python_series, list):
        allowed_python_series = {str(version).rstrip(".") for version in compatible_python_series}
    else:
        allowed_python_series = set()
    current_python = str(current["python"])
    exact_match = current_python in allowed_python_versions
    series_match = any(
        current_python == series or current_python.startswith(f"{series}.")
        for series in allowed_python_series
    )
    if not (exact_match or series_match):
        labels = sorted(allowed_python_versions) + [f"{series}.x" for series in sorted(allowed_python_series)]
        allowed_label = " / ".join(dict.fromkeys(labels))
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, f"Python {allowed_label} の環境が必要です。", 409)
    expected = manifest.get("library_versions", {})
    mismatches = [
        name for name in MODEL_RUNTIME_DISTRIBUTIONS
        if current["packages"].get(name) != expected.get(name)
    ]
    if mismatches:
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "互換性に関係するライブラリのバージョンが一致しません。", 409)


def load_portable_artifact() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest()
    _validate_runtime(manifest)
    if not MODEL_ARTIFACT_PATH.exists() or sha256_file(MODEL_ARTIFACT_PATH) != manifest.get("model_sha256"):
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデル本体が破損しているか、ハッシュが一致しません。", 500)
    try:
        packages = joblib.load(MODEL_ARTIFACT_PATH)
        _assert_portable_package(packages)
    except Exception as error:
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデル本体を読み込めませんでした。", 500) from error
    if packages.get("model_version") != MODEL_VERSION or sorted(packages.get("contexts", {})) != ["after_close", "intraday"]:
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデル本体の内容が現在のサービスと一致しません。", 409)
    if _feature_schema(packages) != manifest.get("feature_schema"):
        raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデルの特徴量schemaがmanifestと一致しません。", 409)
    for context in ("intraday", "after_close"):
        payload = packages["contexts"][context]
        if not all(key in payload for key in ("japan", "overseas", "six_class")):
            raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデルの必須オブジェクトが不足しています。", 409)
        for side in ("japan", "overseas"):
            for component in payload[side].get("components", []):
                if not isinstance(component.get("features"), list) or not isinstance(component.get("medians"), dict):
                    raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均モデルのメタデータ形式が不正です。", 409)
                if "model" not in component and "constant" not in component:
                    raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均の方向モデルが不足しています。", 409)
        six = payload["six_class"]
        if not isinstance(six.get("features"), list) or not isinstance(six.get("medians"), dict):
            raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均6段階モデルのメタデータ形式が不正です。", 409)
        if "model" not in six and "constant" not in six:
            raise ServiceError(MODEL_COMPATIBILITY_ERROR, "日経平均6段階モデルが不足しています。", 409)
    return packages, manifest


def load_context_evaluation(context: str) -> dict[str, Any] | None:
    if context not in EVALUATION_PATHS:
        return None
    try:
        manifest = load_manifest()
        record = manifest["evaluation_files"][context]
        path = EVALUATION_PATHS[context]
        if record.get("file") != path.name or not path.exists() or sha256_file(path) != record.get("sha256"):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (KeyError, OSError, json.JSONDecodeError, ServiceError):
        return None
    if (
        payload.get("prediction_context") != context
        or payload.get("model_schema_version") != MODEL_VERSION
        or payload.get("feature_schema_version") != FEATURE_DEFINITION_VERSION
        or payload.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION
    ):
        return None
    return payload
