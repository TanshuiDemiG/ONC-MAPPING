# Run from the Code directory:
#   pyinstaller src_exe/build_exe.spec

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


SOURCE_DIR = Path.cwd() / "src_exe"
APP_NAME = "ONC_PTWL_Tool_Standalone"
SITE_PACKAGES_DIR = Path.cwd() / ".venv-exe" / "Lib" / "site-packages"
PYSIDE6_DIR = SITE_PACKAGES_DIR / "PySide6"
SHIBOKEN6_DIR = SITE_PACKAGES_DIR / "shiboken6"

datas = []

datas += collect_data_files("rasterio", include_py_files=True)
datas += collect_data_files("onnxruntime", include_py_files=True)

config_template = SOURCE_DIR / "default_config.json"
if config_template.exists():
    datas.append((str(config_template), "config"))

app_icon = SOURCE_DIR / "stone.png"
if app_icon.exists():
    datas.append((str(app_icon), "."))

for relative_dir in (
    "plugins/platforms",
    "plugins/styles",
    "plugins/imageformats",
    "plugins/iconengines",
    "plugins/tls",
):
    plugin_dir = PYSIDE6_DIR / relative_dir
    if plugin_dir.exists():
        datas.append((str(plugin_dir), f"PySide6/{relative_dir}"))

binaries = []

for binary_name, source_dir in (
    ("opengl32sw.dll", PYSIDE6_DIR),
    ("shiboken6.abi3.dll", SHIBOKEN6_DIR),
):
    binary_path = source_dir / binary_name
    if binary_path.exists():
        binaries.append((str(binary_path), "PySide6"))

# These packages are imported dynamically inside runtime loader functions, so
# PyInstaller will miss them unless we declare them explicitly.
hiddenimports = [
    "geopandas",
    "inference_sdk",
    "numpy",
    "onnxruntime",
    "PIL.Image",
    "pyogrio",
    "pyproj",
    "rasterio",
    "rasterio.enums",
    "rasterio.features",
    "rasterio.transform",
    "rasterio.warp",
    "rasterio.windows",
    "shapely.geometry",
    "yaml",
]

excluded_modules = [
    "PySide6.scripts",
    "geopandas.io.tests",
    "geopandas.tests",
    "gi",
    "matplotlib",
    "matplotlib.backends",
    "matplotlib.pyplot",
    "pyogrio.tests",
    "shapely.tests",
    "tkinter",
    "torch",
    "torchvision",
    "_tkinter",
]


a = Analysis(
    [str(SOURCE_DIR / "gui_app.py")],
    pathex=[str(SOURCE_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
