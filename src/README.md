# ONC PTWL Pipeline GUI

This folder contains the desktop wrapper for the existing `test.py` and
`ptwl_habitat_map.py` scripts.

Run the GUI from the `Code` directory:

```powershell
python src\gui_app.py
```

Build the Windows executable from a Windows Python environment:

```powershell
python -m pip install -r src\requirements-gui.txt
python -m PyInstaller src\build_exe.spec
```

The executable folder will be written to:

```text
dist/ONC_PTWL_Tool/
```

The main program inside that folder is:

```text
dist/ONC_PTWL_Tool/ONC_PTWL_Tool.exe
```

PyInstaller builds for the operating system it is running on, so run the
build command in Windows rather than WSL/Linux when you need an `.exe`.
