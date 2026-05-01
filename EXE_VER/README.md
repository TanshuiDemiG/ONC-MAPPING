# ONC PTWL Pipeline GUI

This folder contains the desktop wrapper for the existing `test.py` and
`ptwl_habitat_map.py` scripts.

Rock detection supports two inference backends from the Model tab:

- Roboflow API, using the existing workflow/model settings.
- Local YOLO, using `src/model/best.pt`.

The Detection tab can optionally enable rock size bins. When enabled, the
combined `rocks.shp` includes size fields, and the tool can also write one
additional Shapefile per size class. A size-bin interval can also be selected
as the rock input for habitat mapping; otherwise habitat mapping uses all
detected rocks.

The Habitat tab uses the same integrated vegetation, rock, canopy, and scoring
logic as `ptwl_habitat_zones.py`. The default block size is `1`. Only the final
zone boundary creation uses the score-raster upscaling logic from
`habitat_score_to_interpolated_zones.py`. The final zone output is
`ptwl_habitat_zones.shp`.

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
