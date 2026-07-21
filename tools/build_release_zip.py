from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

TOP_FOLDER = "japanese-stock-strategy-app"
EXCLUDED_DIRS = {
    ".venv", "venv", "env", "compatibility_test_venv", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", ".agents",
    "backup", "backups", "audit_extract", "release_extract", "outputs",
}
EXCLUDED_NAMES = {"startup_error.log", "Thumbs.db", ".DS_Store"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip", ".log", ".tmp"}
REQUIRED_FILES = {
    "app.py", "START_HERE.cmd", "requirements.txt",
    "model_settings/nikkei_model_manifest.json",
    "model_settings/nikkei_dual_market_models.joblib",
    "model_settings/nikkei_intraday_evaluation.json",
    "model_settings/nikkei_after_close_evaluation.json",
    "services/individual_logistic_fast.py",
    "services/simulation_service.py",
    "services/nikkei_dual_model.py",
    "tools/build_release_zip.py",
}
REQUIRED_DIRS = {"services", "templates", "static", "tests", "tools", "model_settings"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def should_include(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(
        part in EXCLUDED_DIRS
        or part.startswith(".test_tmp")
        or part.startswith(".pytest")
        for part in relative.parts
    ):
        return False
    if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def build_release_zip(project_root: Path, output: Path) -> None:
    project_root = project_root.resolve()
    output = output.resolve()
    if project_root.name != TOP_FOLDER:
        raise ValueError(f"project root must be named {TOP_FOLDER}: {project_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(project_root.rglob("*")):
                if not should_include(path, project_root):
                    continue
                relative = path.relative_to(project_root).as_posix()
                archive_name = (PurePosixPath(TOP_FOLDER) / PurePosixPath(relative)).as_posix()
                archive.write(path, archive_name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_release_zip(path: Path, extract_smoke: bool = False) -> dict[str, object]:
    path = path.resolve()
    if not path.exists() or path.stat().st_size <= 0:
        raise ValueError("release ZIP does not exist or is empty")
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        names = archive.namelist()
        if bad_member is not None:
            raise ValueError(f"corrupt ZIP member: {bad_member}")
        if any("\\" in name for name in names):
            raise ValueError("backslash archive name found")
        parsed = [PurePosixPath(name) for name in names]
        if any(item.is_absolute() or ".." in item.parts or ":" in item.parts[0] for item in parsed):
            raise ValueError("unsafe archive path found")
        top_levels = sorted({item.parts[0] for item in parsed if item.parts})
        if top_levels != [TOP_FOLDER]:
            raise ValueError(f"unexpected top-level folders: {top_levels}")
        relative_names = {PurePosixPath(*item.parts[1:]).as_posix() for item in parsed if len(item.parts) > 1}
        missing_files = sorted(REQUIRED_FILES - relative_names)
        missing_dirs = sorted(directory for directory in REQUIRED_DIRS if not any(name.startswith(f"{directory}/") for name in relative_names))
        if missing_files or missing_dirs:
            raise ValueError(f"missing files={missing_files}, dirs={missing_dirs}")
        forbidden = [
            name for name in relative_names
            if any(part in EXCLUDED_DIRS for part in PurePosixPath(name).parts)
            or Path(name).suffix.lower() in EXCLUDED_SUFFIXES
        ]
        if forbidden:
            raise ValueError(f"excluded content found: {forbidden[:5]}")
        if f"{TOP_FOLDER}/{TOP_FOLDER}/app.py" in names:
            raise ValueError("double project folder found")
        if extract_smoke:
            with tempfile.TemporaryDirectory(prefix="stock_release_verify_") as temporary_dir:
                archive.extractall(temporary_dir)
                extracted = Path(temporary_dir) / TOP_FOLDER
                for required in REQUIRED_FILES:
                    if not (extracted / required).is_file():
                        raise ValueError(f"missing after extraction: {required}")
                if not (extracted / "RUN_APP_INNER.cmd").exists():
                    raise ValueError("RUN_APP_INNER.cmd is missing after extraction")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "member_count": len(names),
        "top_level": TOP_FOLDER,
        "backslash_member_count": sum("\\" in name for name in names),
        "testzip": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable Japanese Stock Strategy App release ZIP.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-smoke", action="store_true")
    args = parser.parse_args()
    build_release_zip(args.project_root, args.output)
    print(json.dumps(inspect_release_zip(args.output, args.extract_smoke), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
