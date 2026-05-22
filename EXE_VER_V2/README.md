# ONC PTWL EXE Build

This folder is an `.exe`-oriented copy of `src/`. It keeps the original source
unchanged and applies packaging fixes here instead.

Key changes:

- Uses the executable directory as the runtime root when frozen.
- Reads `.env` from beside the `.exe`.
- Writes the editable default config to `config/default_config.json` beside the `.exe`.
- Does not bundle model files. The app expects models in an external `models/`
  folder beside the `.exe`, or any other location the user chooses in the GUI.
- Ships a minimal default config that prefers a local `.pt` model under
  `models/yolo_pt/best.pt`.
- Uses a dedicated build script so PyInstaller does not inherit the broken
  `pathlib` package from the current Anaconda base environment.

Build from the `Code` directory:

```powershell
.\src_exe\build_exe_clean.ps1
```

Output:

```text
dist/ONC_PTWL_Tool_Standalone/ONC_PTWL_Tool_Standalone.exe
```

Recommended runtime layout:

```text
ONC_PTWL_Tool_Standalone/
  ONC_PTWL_Tool_Standalone.exe
  .env
  config/
    default_config.json
  models/
    rfdetr_large_onnx/
      weights.onnx
      class_names.txt
      environment.json
      model_type.json
    yolo_pt/
      best.pt
```

In the GUI:

- `Local ONNX folder`: select `models/rfdetr_large_onnx/`
- `Local .pt model`: select `models/yolo_pt/best.pt`

For Roboflow API mode, place `.env` beside the built `.exe` with values such as:

```env
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_MODEL_ID=project/version
ROBOFLOW_API_URL=https://serverless.roboflow.com
```
