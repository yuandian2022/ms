# Breath Metabolomics Pipeline — Fasting Study

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A consolidated pipeline for processing **SESI-HRMS breath data** from a fasting study. The pipeline aligns TIC chromatograms with exhalation pressure recordings, extracts breath-plateau intervals, processes blank (background) samples, and performs both **targeted** (known m/z) and **untargeted** screening with hierarchical clustering and linear regression to discover compounds that change significantly during fasting.

## Table of Contents

- [Overview](#overview)
- [Pipeline Steps](#pipeline-steps)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Full Pipeline](#full-pipeline)
  - [Individual Steps](#individual-steps)
  - [Custom Paths](#custom-paths)
- [Input Files](#input-files)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

## Overview

```
              ┌──────────────────┐
              │     Step 1       │ ────▶ Filename match + Blank crop + Breath crop + Target m/z
              │ Data preparation │
              └──────────────────┘
                        │
                 ┌──────▼──────┐
  mzML +         │   Step 2    │  Breath intervals + TIC plots
  Exhalion ────▶│  Alignment  │
                 └──────┬──────┘
                        │
  Blank          ┌──────▼───────┐
  mzML ─────────▶│   Step 3     │  Blank intervals + TIC plots
                 │   Blanks     │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │   Step 4     │  Detection-time CSVs
                 │  Timestamps  │
                 └──────┬───────┘
                        │
          ┌─────────────┼─────────────┐
          │                           │
  ┌───────▼───────┐           ┌───────▼───────┐
  │   Step 5      │           │   Step 6      │
  │  Known m/z    │           │  Untargeted   │
  │  Trend Plots  │           │  Screening    │
  └───────────────┘           └───────────────┘


```

## Pipeline Steps

| Step | CLI flag | Description |
|------|----------|-------------|
| 2 | `--step2` | Align TIC with exhalation pressure via cross-correlation; extract breath plateau intervals |
| 3 | `--step3` | Process blank (background) mzML files; apply manual crop rules; export retained RT intervals |
| 4 | `--step4` | Extract `startTimeStamp` from breath/blank mzML files (using intervals from steps 2+3); produce detection-time CSVs |
| 5 | `--step5` | Extract known target m/z intensities; nearest-blank subtraction; dual-panel fasting-trend plots (relative intensity + % change) |
| 6 | `--step6` | Untargeted screening: 5 ppm hierarchical m/z clustering (cluster representative = m/z of the strongest peak in each cluster) → sample×feature matrix → per-subject linear regression → BH-FDR correction → consistent-trend reporting + volcano plot |

> **Note:** Steps 1–6 correspond to the manuscript analysis workflow. Step 1 (data preparation) involves manual file organisation and CSV creation; it is not automated by this script.

## Directory Structure

```
.
├── breath_analysis.py               ★ Main pipeline script
├── README.md
├── requirements.txt
├── .gitignore
│
├── 1.1_Exhalion_SESE_Filename.csv   Breath ↔ exhalion pairing table
├── 1.2_TIC_crop_converted.csv       Manual crop rules (breath)
├── 1.2_blank_TimeChoose.csv         Manual crop rules (blanks)
├── 1.3_target_mz.csv                Known target m/z list
│
├── centroid_MS/
│   ├── Breath/              *.mzML — breath samples
│   └── Blank/               *.mzML — blank / background samples
│
├── Exhalion/                *.txt — exhalion pressure recordings
│
└── output/                  ★ All pipeline outputs (auto-created; current results are tracked in the repo)
    ├── step2_intervals/
    ├── step3_blanks/
    ├── step4_detection/
    ├── step5_known/
    └── step6_unknown/
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd <repo>

# Create and activate a conda environment (recommended)
conda create -n breath python=3.9 -y
conda activate breath

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9 · NumPy · Pandas · SciPy · Matplotlib · pymzML

## Quick Start

```bash
# Run the full pipeline with default settings
python breath_analysis.py --all
```

All output will appear under `output/`.  Re-running a step overwrites its own output files in place (Pandas `to_csv` / `fig.savefig` default behaviour).

## Usage

### Full Pipeline

```bash
python breath_analysis.py --all
```

Runs steps 2→6 sequentially. Each step reads the outputs of the previous step from the `output/` directory.

### Individual Steps

```bash
# Step 2: breath interval extraction (pos mode)
python breath_analysis.py --step2 --ion pos

# Step 3: blank intervals (pos mode)
python breath_analysis.py --step3 --ion pos

# Step 4: detection timestamps (uses step 2+3 outputs)
python breath_analysis.py --step4

# Step 5: known-target trend analysis (pos mode)
python breath_analysis.py --step5 --known-ion pos

# Step 6: untargeted screening (pos mode)
python breath_analysis.py --step6 --unknown-ion pos
```

### Custom Paths

All paths are relative to the script's location by default. Override with:

```bash
python breath_analysis.py --all \
    --base-dir   /custom/path/to/project \
    --output-dir /custom/path/to/results \
    --mzml-breath /custom/path/breath_mzML \
    --mzml-blank  /custom/path/blank_mzML \
    --exhalion-dir /custom/path/exhalion_txt \
    --match-csv /custom/path/pairing.csv
```

Run `python breath_analysis.py --help` for the full list of options.

## Input Files

All paths below are relative to the project root (or `--base-dir`).

| Path | Description |
|------|-------------|
| `1.1_Exhalion_SESE_Filename.csv` | Columns: `mzml_filename`, `txt_filename` — pairs each breath mzML to its exhalion recording |
| `1.2_TIC_crop_converted.csv` | Columns: `number`, `S1_pos`, `S1_neg`, `S2_pos`, … — manual time-window crop rules per breath file. `None` = no crop, `Delete` = skip file, `[50,]` = keep from 50 s onward |
| `1.2_blank_TimeChoose.csv` | Manual time-window crop rules per blank file. Columns include `sample_id`, `{date}_{ion}` |
| `1.3_target_mz.csv` | Columns: `Polarity` (pos/neg), `VOC` (compound name), `RMM` (exact m/z) |
| `centroid_MS/Breath/*.mzML` | Breath mzML files, naming: `DD_MM_YYYY_FASTING_S{1,2,3}_NNN_{pos,neg}.mzML` |
| `centroid_MS/Blank/*.mzML` | Blank mzML files, naming: `DD_MM_YYYY_FASTING_BN_{pos,neg}.mzML` |
| `Exhalion/*.txt` | Whitespace-delimited: `time exhaled_co2 pressure flow_rate total_exhaled_volume` |

### Filename convention

**Breath samples:** `25_03_2026_FASTING_S1_001_pos.mzML`

| Component | Meaning | Values |
|-----------|---------|--------|
| `DD_MM_YYYY` | Acquisition date | e.g. `25_03_2026` |
| `S{1,2,3}` | Subject ID | `S1`, `S2`, `S3` |
| `NNN` | Sequential sample number | `001`–`041` |
| `pos` / `neg` | Ionisation polarity | `pos` or `neg` |

**Blank samples:** `25_03_2026_FASTING_B1_pos.mzML` — same format with `B{1,2,3}` as the blank identifier.

## Output Files

All output is written under `output/` (configurable with `--output-dir`). Current results are tracked in the repository.

### Step 2 — `output/step2_intervals/`

| File | Description |
|------|-------------|
| `2_Breath_intervals_trimmed_{ion}.csv` | 5 plateau intervals per breath file (seconds from aligned TIC start) |
| `2_Breath_plots_{ion}.pdf` | TIC-over-breath-plateau QC plots for every file |
| `2_Breath_detection_summary_{ion}.csv` | Per-file total detection time |

### Step 3 — `output/step3_blanks/`

| File | Description |
|------|-------------|
| `3_Blank_{ion}_retained_intervals.csv` | Retained RT windows per blank file |
| `3_platform_plots_{ion}_Background.pdf` | Raw + retained TIC plots for every blank |

### Step 4 — `output/step4_detection/`

| File | Description |
|------|-------------|
| `4_Breath_international_detection_time.csv` | Per-breath-file detection timestamp (ISO 8601) with status |
| `4_Blank_international_detection_time.csv` | Per-blank-file detection timestamp |

### Step 5 — `output/step5_known/`

| File | Description |
|------|-------------|
| `targets_used.csv` | Target m/z list used for extraction |
| `intervals_with_detection_time.csv` | Merged interval + detection-time table |
| `processing_status.csv` | Per-file extraction status |
| `scan_peak_values_long.csv` | Scan-level intensity values for every target |
| `file_target_intensity_summary.csv` | Per-file, per-target mean/median absolute & relative intensities |
| `breath_nearest_blank_subtracted.csv` | Breath-minus-nearest-blank intensities (empty if no blank data) |
| `known_mz_relative_trends.pdf` | Dual-panel plots: relative intensity (top) + % change from first point (bottom) |
| `run_summary.csv` | Summary metrics (targets, samples, rows) |

### Step 6 — `output/step6_unknown/`

| File | Description |
|------|-------------|
| `00_run_summary.csv` | Key metrics (sample count, feature count, etc.) |
| `01_sample_qc.csv` | Per-sample extraction QC |
| `01_scan_peak_relative_long.csv` | Raw scan-level peaks (m/z, intensity, relative intensity) |
| `02_mz_5ppm_clusters.csv` | m/z → feature cluster mapping (5 ppm); `cluster_mz` is the m/z of the member with the highest summed intensity across all scans (representative m/z) |
| `03_cluster_feature_sample_matrix.csv` | Sample × feature matrix |
| `03_feature_presence_filter_summary.csv` | Feature presence statistics |
| `04_subject_feature_linear_regression.csv` | Per-subject, per-feature regression results |
| `05_consistent_trend_regression_rows.csv` | Regression rows passing FDR + direction-consistency filter |
| `05_consistent_trend_summary.csv` | Feature-level summary of consistent trends (with `rank_score`) |
| `06_plot_index.csv` | Index of individual feature trend plots |
| `07_volcano_plot.png` | Volcano plot (maximum slope across subjects vs −log10 max BH-adjusted p-value). Only generated when regression produces ≥1 row. |
| `08_volcano_data.csv` | Volcano-ready data with derived columns (`label`, `neg_log10_max_pvalue_bh`, `abs_mean_slope`). Only generated when ≥1 consistent feature is found. |
| `plots_linear_regression/*.png` | Per-feature dual-panel plots (intensity + % change) for top features |

> **Note:** Files `07_volcano_plot.png` and `08_volcano_data.csv` are only written when regression has at least one row (`MIN_SAMPLES_PER_SUBJECT` samples with ≥2 distinct detection times per subject). With single-sample or single-time-point data, the previous PNG/CSV may remain on disk if not deleted manually.

## Configuration

Key parameters are defined as module-level constants at the top of `breath_analysis.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PPM_TOL` | 5.0 | Mass tolerance (ppm) for known-target extraction |
| `MZ_MIN` / `MZ_MAX` | 50 / 350 | m/z range for untargeted screening |
| `MIN_PEAK_INTENSITY` | 500 | Minimum absolute peak intensity |
| `MZ_DECIMALS` | 4 | m/z rounding precision for clustering |
| `CLUSTER_PPM` | 5 | Hierarchical clustering tolerance (ppm) |
| `MIN_SAMPLES_PER_SUBJECT` | 25 | Minimum samples per subject for regression (lower for testing with small datasets) |
| `FDR_ALPHA` | 0.05 | Benjamini-Hochberg FDR threshold |
| `MIN_CONSISTENT_SUBJECTS` | 2 | Minimum subjects with consistent trend direction (set to 0 for single-subject analysis) |
| `TRIM_FRONT_PERCENT` / `TRIM_BACK_PERCENT` | 20 | Percentage trimmed from each exhalation plateau |
