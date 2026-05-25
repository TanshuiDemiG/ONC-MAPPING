# ONC-MAPPING

[![Project Page](https://img.shields.io/badge/Project-ONC--MAPPING-blue)](https://github.com/TanshuiDemiG/ONC-MAPPING)

[![GitHub Releases](https://img.shields.io/badge/Releases-Latest-green)](https://github.com/TanshuiDemiG/ONC-MAPPING/releases)

[![Colab](https://img.shields.io/badge/Google-Colab-orange)](https://colab.research.google.com/)

[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)

[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://chatgpt.com/c/6a114fe4-2424-83ec-885c-2682cb0737b5#license)

> This software project accompanies the ONC-MAPPING workflow for stone segmentation, mapping visualization, and route analysis research.\n\n> Developed as part of an Australian National University (ANU) postgraduate internship collaboration project with the Australian Government, the project aims to reduce manual natural exploration and surveying workloads through machine learning analysis of drone imagery and vegetation overlay analysis.

---

# Project Overview

ONC-MAPPING is a lightweight computer-vision workflow designed for:

* Stone / hold segmentation
* Spatial mapping visualization
* Route analysis workflows
* Google Colab deployment
* Standalone executable usage

The project aims to reduce deployment complexity while maintaining a modular and extensible pipeline for future development and research.

---

# Repository

* GitHub Repository

  [https://github.com/TanshuiDemiG/ONC-MAPPING](https://github.com/TanshuiDemiG/ONC-MAPPING)
* Releases Page

  [https://github.com/TanshuiDemiG/ONC-MAPPING/releases](https://github.com/TanshuiDemiG/ONC-MAPPING/releases)

---

# Features

## Google Colab Deployment

* End-to-end runnable notebook workflow
* Cloud-based execution with minimal setup
* Preconfigured inference pipeline
* Simplified model weight management

Main notebook:

```text

COLAB_ED/DEPLOY/stone_pipeline_colab.ipynb

```

---

## Standalone EXE Version

* Windows executable deployment
* No Python environment required for end users
* GUI-oriented workflow
* Designed for quick demonstrations and usability testing

Release branch/tag:

```text

release-exe-ver-0.1

```

---

# Releases

## release-colab-0.1 (v0.1)

### Overview

Initial public Google Colab deployment pipeline for the ONC-MAPPING workflow.

### Added

* Google Colab deployment notebook
* End-to-end segmentation and mapping execution pipeline
* Organized model weight directory structure
* Video and Document User_Guide

### How to use

- download from [Colab Version Release](https://github.com/TanshuiDemiG/ONC-MAPPING/releases/tag/release-colab-0.1)
- follow the User_Guide for deployment and execution.
- the video guide is in the zipped file

---



## release-exe-ver-0.1

### Overview

Initial standalone executable release for:

```text

ONC_PTWL_Tool_Standalone

```

### Features

* Packaged executable deployment
* Simplified usage workflow
* Reduced environment configuration requirements
* Intended for quick local demonstrations and testing

---

# Getting Started

## Option 1 — Google Colab

### Step 1

Open:

```text

COLAB_ED/DEPLOY/stone_pipeline_colab.ipynb

```

in Google Colab.

### Step 2

Ensure model weights are placed correctly:

```text

Model_Weights/best_DY.pt

```

### Step 3

Run notebook cells sequentially.

---

## Option 2 — EXE Version

1. Download the executable release from the Releases page
2. Extract the package
3. Launch the executable application
4. Follow the interface instructions

---

# Repository Structure

```text

ONC-MAPPING/

│

├── COLAB_ED/

│   └── DEPLOY/

│       └── stone_pipeline_colab.ipynb

│

├── Model_Weights/

│   └── best_DY.pt

│

├── ONC_PTWL_Tool_Standalone/

│

└── README.md

```

---

# Compatibility

## Colab Version

Designed primarily for:

* Google Colab
* CUDA-enabled cloud runtime environments

Local execution may require:

* Dependency installation
* Path adjustments
* CUDA configuration

---

## EXE Version

Designed primarily for:

* Windows systems

Additional dependencies may still be required for GPU acceleration.

---

# Future Development

Planned improvements include:

* Improved stone/hold detection accuracy
* Route recommendation functionality
* Enhanced GUI system
* Real-time video processing
* Better local deployment support
* Additional model architecture integration

---

# Contributing

Contributions, issues, and feature suggestions are welcome.

Please use the GitHub Issues page for bug reports and enhancement requests.

---

# Citation

If you use this project in your research or development, please consider citing the repository.

```bibtex

@software{onc_mapping,

  author = {TanshuiDemiG},

  title = {ONC-MAPPING},

  year = {2026},

  url = {https://github.com/TanshuiDemiG/ONC-MAPPING}

}

```

---

# License

Please refer to the repository license information for usage details.

---

# Author

GitHub:

[https://github.com/TanshuiDemiG](https://github.com/TanshuiDemiG)
