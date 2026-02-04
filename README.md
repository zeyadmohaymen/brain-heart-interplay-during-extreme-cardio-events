# Project 1 : Brain Dynamics during «extreme» cardiovascular events

This repository implements **Project 1: Brain Dynamics during «extreme» cardiovascular events** by combining (1) point-process HRV estimation, (2) time-resolved directional Brain–Heart Interplay (BHI) estimation, and (3) non-parametric statistics to quantify how **EEG–HRV coupling dynamics change around “extreme” cardiovascular events**.

## Implementation / Methodology (as implemented here)

1. **Load multi-subject EMO recordings**
   - Subject loading and summaries are handled by [`src.data_loader.load_subject`](src/data_loader.py) / [`src.data_loader.load_all_subjects`](src/data_loader.py) in [src/data_loader.py](src/data_loader.py).
   - The main analysis is orchestrated in [experiment.ipynb](experiment.ipynb).

2. **HRV estimation via point-process modeling**
   - Baseline and experiment RR timestamps are concatenated using [`src.utils.stitch_rr_phases`](src/utils.py) in [src/utils.py](src/utils.py).
   - RR timestamps can be cleaned using [`src.utils.preprocess_rr`](src/utils.py) in [src/utils.py](src/utils.py).
   - A time-varying inverse-Gaussian point-process regression is fit (via the `pointprocess` bindings) to produce time-resolved HRV indices (e.g., LF/HF power).  
   - Summary figures are generated with [`src.utils.plot_all_hrv_summaries`](src/utils.py) in [src/utils.py](src/utils.py).

3. **“Extreme” cardiovascular event detection**
   - “Extreme” events are detected from the HF power series using [`src.utils.detect_extreme_events_gradient_peaks`](src/utils.py) in [src/utils.py](src/utils.py).  
   - Events are labeled (e.g., `SURGE` / `DROP`) and later used as anchors for epoch extraction.

4. **Directional Brain–Heart Interplay (BHI) estimation (MATLAB SDG model)**
   - Time-resolved BHI is computed using the SDG model scripts in [BHI_SDG/](BHI_SDG/README.md) (executed from Python via `matlab.engine` in [experiment.ipynb](experiment.ipynb)).
   - Output `.mat` files are saved under:
     - [results/bhi/baseline/](results/bhi/baseline/)
     - [results/bhi/experiment/](results/bhi/experiment/)

5. **Epoching BHI around extreme events**
   - Event-locked BHI epochs are assembled into a tidy dataframe using [`src.bhi_dataframe.create_bhi_dataframe`](src/bhi_dataframe.py) in [src/bhi_dataframe.py](src/bhi_dataframe.py).
   - Optional padding is used to analyze temporal dynamics before/after the event.

6. **Statistics (global and time-resolved)**
   - For each `(direction × EEG band × HRV band × channel)` combination:
     - **Friedman test** across conditions (`BL`, `SURGE`, `DROP`)
     - **Wilcoxon** post-hoc pairwise comparisons with **Bonferroni correction**
   - A **time-resolved** variant repeats the procedure after collapsing epochs per subject and computing medians within fixed windows (e.g., 5-second segments).

7. **Visualization**
   - Channel-level significance summaries as heatmaps/barplots.
   - Scalp **topomaps** of Friedman statistics with significant electrodes masked (built with `mne` inside [experiment.ipynb](experiment.ipynb)).

## How to run

### 1) Environment
- Python version is pinned via [.python-version](.python-version).
- Python dependencies are listed in [requirements.txt](requirements.txt).

Install:
```bash
pip install -r requirements.txt
```

> Note: `matlab.engine` is provided by your local MATLAB installation (not via `pip`).

### 2) Dependencies (non-PyPI / external)

#### `pointprocess` (Barbieri et al. point-process HRV)
This project uses the **Barbieri et al. point-process HRV implementation** via local bindings under `pointprocess/build/` (imported in [experiment.ipynb](experiment.ipynb) by appending that path to `sys.path`).

Upstream reference:
- https://github.com/andreabonvini/pointprocess

If `import pointprocess` fails, verify that:
- `pointprocess/build/` exists in this repository/workspace
- the bindings were built for your **current** Python version / macOS architecture

#### SDGM BHI model (Catrambone et al. Brain–Heart Interaction Indexes)
Directional BHI is computed using the **SDG/SDGM MATLAB implementation** (invoked as `SDGM_LFHF(...)`) and stored under [results/bhi/](results/bhi/).

Upstream reference:
- https://github.com/CatramboneVincenzo/Brain-Heart-Interaction-Indexes

Notes:
- MATLAB is required; the notebook starts MATLAB via `matlab.engine` and adds [BHI_SDG/](BHI_SDG/) to the MATLAB path.
- Please refer to the upstream repository (and [BHI_SDG/README.md](BHI_SDG/README.md)) for model details and licensing/attribution.

### 3) Data & results layout
- Input data is expected under [data/](data/) in per-subject folders (e.g., `data/i24/...`).
- BHI outputs are stored under [results/bhi/](results/bhi/).

### 4) Main analysis notebook
Open and run:
- [experiment.ipynb](experiment.ipynb)

This notebook performs:
- HRV computation + event detection
- BHI computation (MATLAB engine)
- Epoch dataframe construction
- Global + time-resolved statistics
- Visualizations (heatmaps/topomaps)

---

## Key code entry points

- Data loading: [`src.data_loader.load_all_subjects`](src/data_loader.py) ([src/data_loader.py](src/data_loader.py))
- RR stitching/cleaning: [`src.utils.stitch_rr_phases`](src/utils.py), [`src.utils.preprocess_rr`](src/utils.py) ([src/utils.py](src/utils.py))
- Extreme event detection: [`src.utils.detect_extreme_events_gradient_peaks`](src/utils.py) ([src/utils.py](src/utils.py))
- Epoch dataframe: [`src.bhi_dataframe.create_bhi_dataframe`](src/bhi_dataframe.py) ([src/bhi_dataframe.py](src/bhi_dataframe.py))

---

## Notes / Caveats

- MATLAB is required for the SDG BHI estimation step (see [BHI_SDG/README.md](BHI_SDG/README.md)).
- Some computations (especially BHI batch processing) are long-running; the repository is organized to persist outputs under [results/](results/).

## License
- Project-specific code in [src/](src/) and notebooks: see repository-level licensing as applicable.
- The SDG MATLAB implementation is distributed under its own license in [BHI_SDG/](BHI_SDG/LICENSE) (see also [BHI_SDG/README.md](BHI_SDG/README.md)).