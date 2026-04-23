# Run from the Code directory:
#   pyinstaller src/build_exe.spec

from PyInstaller.utils.hooks import collect_all


datas = []
binaries = []
hiddenimports = []

for package in (
    "PySide6",
    "rasterio",
    "geopandas",
    "shapely",
    "pyproj",
    "fiona",
    "pyogrio",
    "PIL",
    "inference_sdk",
):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
    except Exception:
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


a = Analysis(
    ["src/gui_app.py"],
    pathex=["src", "."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ONC_PTWL_Tool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name="ONC_PTWL_Tool",
)
