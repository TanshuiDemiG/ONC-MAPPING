# ONC-MAPPING

[![Releases](https://img.shields.io/badge/Releases-GitHub-green)](https://github.com/TanshuiDemiG/ONC-MAPPING/releases)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Colab%20%7C%20Windows-lightgrey)](https://github.com/TanshuiDemiG/ONC-MAPPING)

ONC-MAPPING is a computer vision workflow for rock detection, habitat scoring, and mapping outputs from orthomosaic and vegetation data. The repository contains two active delivery tracks:

- `COLAB_ED`: Google Colab deployment for notebook-based execution.
- `EXE_VER_V2`: source and build assets for the Windows standalone GUI workflow.

Packaged end-user software should be downloaded from the GitHub Releases page, not from the source folders in this repository:

- Releases: https://github.com/TanshuiDemiG/ONC-MAPPING/releases

## Table of Contents

- [Overview](#overview)
- [Repository Scope](#repository-scope)
- [Download and Releases](#download-and-releases)
- [Repository Structure](#repository-structure)
- [Google Colab Workflow](#google-colab-workflow)
- [Windows EXE Workflow](#windows-exe-workflow)
- [Development Notes](#development-notes)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

This project supports a workflow for:

- rock detection from orthomosaic imagery
- habitat scoring using vegetation and rock layers
- spatial output generation for mapping and review
- Google Colab deployment for cloud-based runs
- Windows GUI packaging for local demonstrations and operational use

The repository is organized to separate active delivery paths from archived historical material.

## Repository Scope

This repository is primarily for:

- source code
- notebook workflows
- model assets used by the workflows
- build configuration
- project documentation

This repository is not the recommended distribution point for end users who only need the runnable software package. For packaged downloads, use GitHub Releases.

## Download and Releases

### For end users

Use the Releases page to download packaged software and release bundles:

- GitHub Releases: https://github.com/TanshuiDemiG/ONC-MAPPING/releases

Typical use:

1. Open the Releases page.
2. Download the relevant release asset for your workflow.
3. Extract the archive.
4. Follow the included guide or release notes.

### What should be downloaded from Releases

- Windows standalone application packages
- release snapshots prepared for users
- zipped delivery bundles referenced by project release notes

### What should be used from this repository directly

- source code under `EXE_VER_V2`
- the Colab notebook under `COLAB_ED/DEPLOY`
- project documentation under `DOC`

## Repository Structure

Current top-level structure:

```text
ONC-MAPPING/
├── COLAB_ED/
│   └── DEPLOY/
│       ├── stone_pipeline_colab.ipynb
│       ├── stone_pipeline_colab.py
│       ├── User_Guide.md
│       └── Model_Weights/
├── EXE_VER_V2/
│   ├── gui_app.py
│   ├── pipeline.py
│   ├── pipeline_config.py
│   ├── habitat_runner.py
│   ├── detection_runner.py
│   ├── default_config.json
│   ├── build_exe.spec
│   ├── build_exe_clean.ps1
│   └── model/
├── DOC/
│   ├── Poster/
│   └── USR_Story/
├── _archive/
│   ├── Code/
│   ├── DRAFTS/
│   ├── EXE_VER/
│   └── TEST/
└── README.md
```

### Active folders

- `COLAB_ED/DEPLOY`: active Colab notebook workflow, guide, and Colab-side model assets
- `EXE_VER_V2`: active Windows GUI source tree and packaging-related files
- `DOC`: supporting documents, poster files, and user-story material

### Archived folders

- `_archive`: retained historical material that is no longer part of the primary active structure

## Google Colab Workflow

Primary entry point:

- `COLAB_ED/DEPLOY/stone_pipeline_colab.ipynb`

Supporting files:

- `COLAB_ED/DEPLOY/stone_pipeline_colab.py`
- `COLAB_ED/DEPLOY/User_Guide.md`
- `COLAB_ED/DEPLOY/Model_Weights/`

### Intended usage

The Colab workflow is designed for running the pipeline in Google Colab with the deployment folder uploaded to Google Drive.

Recommended usage pattern:

1. Upload the entire `COLAB_ED/DEPLOY` folder contents to Google Drive.
2. Open `stone_pipeline_colab.ipynb` in Google Colab.
3. Configure input paths and model selection.
4. Run the workflow in the required mode.

### Main Colab run modes

- `full`: run detection and habitat scoring
- `detection_only`: generate rock detections only
- `habitat_only`: skip detection and score from an existing rock shapefile

### Typical Colab inputs

- orthomosaic imagery
- canopy shapefile data
- vegetation raster data
- local model weights or configured remote model access

For detailed parameter descriptions, see:

- [COLAB_ED/DEPLOY/User_Guide.md](COLAB_ED/DEPLOY/User_Guide.md)

## Windows EXE Workflow

Primary source folder:

- `EXE_VER_V2`

This folder contains the maintained source tree for the standalone Windows workflow, including:

- GUI entry points
- detection and habitat pipeline modules
- default runtime configuration
- packaging specification and build helper scripts
- local model folder structure used by the packaged app

### Important distribution note

If you only need the runnable Windows software, download it from GitHub Releases instead of using the source tree directly.

### Main files in `EXE_VER_V2`

- `gui_app.py`: GUI application entry point
- `pipeline.py`: pipeline orchestration
- `pipeline_config.py`: runtime configuration schema and defaults
- `detection_runner.py`: detection workflow logic
- `habitat_runner.py`: habitat workflow logic
- `default_config.json`: default runtime toggles
- `build_exe.spec`: PyInstaller specification
- `build_exe_clean.ps1`: Windows build helper script

### Runtime model layout

The EXE workflow expects model assets in a runtime-accessible `models/` layout. A representative source layout is:

```text
EXE_VER_V2/model/
├── rfdetr_large_onnx/
│   ├── class_names.txt
│   ├── environment.json
│   └── model_type.json
└── yolo_pt/
    ├── best.pt
    └── additional model variants
```

Additional model variants may also exist in the source tree for maintenance or testing, but packaged end-user delivery should follow the release bundle contents.

### Build and maintenance

`EXE_VER_V2` is primarily a maintenance and packaging folder for developers working on the Windows version. End users should prefer release assets.

For source-specific packaging details, see:

- [EXE_VER_V2/README.md](EXE_VER_V2/README.md)

## Development Notes

### Recommended usage split

- End users: download packaged assets from Releases
- Colab users: work from `COLAB_ED/DEPLOY`
- Windows build or source maintenance: work from `EXE_VER_V2`

### Archive policy

Historical or superseded material has been moved under `_archive/` to keep the active project structure focused on current delivery paths.

### Repository hygiene

Python cache files are excluded from version control through `.gitignore`, and the active tree has been reduced to the current working structure plus explicit archive content.

## Documentation

Project documentation is stored in `DOC/`, including:

- poster files
- user story documents
- supporting research/project notes

Additional workflow-specific documentation is stored beside the relevant implementation:

- Colab guide in `COLAB_ED/DEPLOY/User_Guide.md`
- EXE packaging notes in `EXE_VER_V2/README.md`

## Contributing

Contributions, issue reports, and documentation improvements are welcome.

Recommended contribution flow:

1. Open an issue describing the bug, requirement, or proposed change.
2. Keep changes scoped to the active folders unless the update is explicitly archival.
3. Submit a pull request with a clear summary of behavioral or structural changes.

## Citation

If you use this repository in research or operational documentation, cite the project repository:

```bibtex
@software{onc_mapping,
  author = {TanshuiDemiG},
  title = {ONC-MAPPING},
  year = {2026},
  url = {https://github.com/TanshuiDemiG/ONC-MAPPING}
}
```

## License

No standalone `LICENSE` file is currently present in this repository. Confirm usage and redistribution terms with the repository owner before external distribution or reuse.
