from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import sys
from pathlib import Path


IS_FROZEN = bool(getattr(sys, "frozen", False))


def _candidate_onnxruntime_capi_dirs() -> list[Path]:
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "onnxruntime" / "capi")
    exe_dir = Path(sys.executable).resolve().parent
    candidates.append(exe_dir / "_internal" / "onnxruntime" / "capi")
    candidates.append(exe_dir / "onnxruntime" / "capi")
    try:
        spec = importlib.util.find_spec("onnxruntime")
    except Exception:
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidates.append(Path(location) / "capi")
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _preload_onnxruntime_dll() -> None:
    # On Windows, a stale onnxruntime.dll in System32 (e.g. shipped with Windows ML)
    # is found before the bundled one by the default DLL search order, causing
    # `import onnxruntime` to fail with WinError 1114 due to a version mismatch.
    # Pre-load the local DLL with LOAD_WITH_ALTERED_SEARCH_PATH so the right
    # version is already in memory when the .pyd extension resolves its imports.
    log_path = Path(sys.executable).resolve().parent / "onnxruntime_preload.log"
    lines: list[str] = []

    def note(message: str) -> None:
        lines.append(message)

    note(f"sys.platform={sys.platform}")
    note(f"frozen={getattr(sys, 'frozen', False)}")
    note(f"_MEIPASS={getattr(sys, '_MEIPASS', None)}")
    note(f"sys.executable={sys.executable}")

    if sys.platform != "win32":
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return

    capi_dir: Path | None = None
    dll_path: Path | None = None
    for candidate in _candidate_onnxruntime_capi_dirs():
        candidate_dll = candidate / "onnxruntime.dll"
        note(f"checking {candidate_dll} exists={candidate_dll.exists()}")
        if candidate_dll.exists():
            capi_dir = candidate
            dll_path = candidate_dll
            break

    if capi_dir is None or dll_path is None:
        note("no onnxruntime.dll found")
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return

    try:
        os.add_dll_directory(str(capi_dir))
        note(f"add_dll_directory OK: {capi_dir}")
    except (AttributeError, OSError) as error:
        note(f"add_dll_directory failed: {error}")

    LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
    try:
        ctypes.CDLL(str(dll_path), winmode=LOAD_WITH_ALTERED_SEARCH_PATH)
        note(f"preloaded {dll_path}")
    except OSError as error:
        note(f"preload failed: {error}")

    providers_dll = capi_dir / "onnxruntime_providers_shared.dll"
    if providers_dll.exists():
        try:
            ctypes.CDLL(str(providers_dll), winmode=LOAD_WITH_ALTERED_SEARCH_PATH)
            note(f"preloaded {providers_dll}")
        except OSError as error:
            note(f"providers preload failed: {error}")

    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass


_preload_onnxruntime_dll()


MODULE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else MODULE_DIR.parent
CODE_DIR = APP_DIR if IS_FROZEN else MODULE_DIR.parent
PROJECT_ROOT = APP_DIR if IS_FROZEN else CODE_DIR.parent
OUTPUT_ROOT = APP_DIR / "outputs" if IS_FROZEN else PROJECT_ROOT / "outputs"
ENV_PATH = APP_DIR / ".env" if IS_FROZEN else PROJECT_ROOT / ".env"
DEFAULT_CONFIG_PATH = APP_DIR / "config" / "default_config.json" if IS_FROZEN else MODULE_DIR / "default_config.json"


def _config_template_path() -> Path:
    if not IS_FROZEN:
        return MODULE_DIR / "default_config.json"

    candidates = [
        MODULE_DIR / "config" / "default_config.json",
        APP_DIR / "_internal" / "config" / "default_config.json",
        APP_DIR / "config" / "default_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


CONFIG_TEMPLATE_PATH = _config_template_path()


def ensure_user_default_config() -> Path:
    target = DEFAULT_CONFIG_PATH
    template = CONFIG_TEMPLATE_PATH
    if target.exists() or not template.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, target)
    return target
