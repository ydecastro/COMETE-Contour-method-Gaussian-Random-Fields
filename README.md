# Anisotropy Simulation Study

This workspace hosts the simulation and analysis notebooks used to study anisotropy estimators on Gaussian random fields. The workflow is split into two notebooks:

- **`simulations_paper.ipynb`** generates synthetic Gaussian random fields with controlled anisotropy, extracts contour-based summaries, and writes batch CSV files under `results/`.
- **`notebook_paper.ipynb`** ingests the exported CSV files, filters them to the scenarios reported in the manuscript, recreates every figure (joint scatters, marginal densities, squared-error diagnostics, and power curves), and exports an HTML version for sharing.

## Repository Layout

- `results/`: Folder where each simulation batch stores its CSV output. The analysis notebook expects files containing chi-square diagnostics (`factor_divide`, `chi2_pvalue`, `total_points_generated`, `kappa`, `T_parameter`, `level_val`, `unit_length`).
- `data/`: Auxiliary reference imagery used in exploratory notebooks.
- `simulations_paper.ipynb`: Production of Gaussian fields, contour geometry, Cabana/LKC statistics, and CSV export.
- `notebook_paper.ipynb`: Post-processing of the consolidated CSV data, including estimator diagnostics and statistical power comparisons.
- `README.md`: This document.

## Simulation Notebook (`simulations_paper.ipynb`)

1. Defines the Gaussian covariance model with `gstools`, including anisotropy ratio, angular orientation, and spatial resolution.
2. Simulates multiple random fields, extracts level-set contours, and computes descriptive statistics (perimeter, Euler characteristics, Cabana/LKC estimates, chi-square diagnostics).
3. Stores each batch of 100 simulations as a CSV file in `results/<timestamp>/batch_*.csv`.
4. Provides lightweight visualization cells to sanity-check gradient fields and contour extractions.

Key dependencies: `numpy`, `pandas`, `gstools`, `scipy`, `matplotlib`, `seaborn`, `skimage`, `tqdm`.

## Analysis Notebook (`notebook_paper.ipynb`)

1. Recursively loads every CSV file found in `results/`, retaining only the runs with the required chi-square diagnostics and `T_parameter = 200`, `unit_length = 5`, `field_size = 1000`, `total_points_generated = 1e7`, and `factor_divide ∈ {NaN, 10, 25}`.
2. Defines the theoretical Cabana and Biermé–Desolneux elliptic integrals and evaluates them on fine κ grids for plotting.
3. Produces:
   - Joint scatter plots comparing Cabana vs LKC, gradient vs Cabana, and the associated angle estimators.
   - Marginal densities for each estimator across true κ values and excursion levels `u ∈ {0, 1, 2}`.
   - Squared-error diagnostics with log-scaled axes.
   - Power curves contrasting MB-Contour and MB-LKC tests, complete with chi-square p-value ECDF overlays.
4. Exports consolidated data to `simulations_paper.csv` and (optionally) converts the notebook into `notebook_paper.html` via `nbconvert`.

## Reproducing the Pipeline

1. **Simulations**
   - Open `simulations_paper.ipynb`.
   - Adjust the anisotropy grid, noise level, or seed if required.
   - Run all cells to generate CSV files inside `results/`.

2. **Analysis**
   - Ensure the generated CSV files remain under `results/`.
   - Open `notebook_paper.ipynb` and execute all cells. Figures display inline; power curves optionally save with `save_figs = True`.
   - The filtered dataset is saved as `simulations_paper.csv` in the project root for sharing or downstream scripts.

3. **HTML Export**
   - The final cell runs `jupyter nbconvert` to create `notebook_paper.html`, providing a static artifact ready for collaboration.

## Notes

- The notebooks expect Python 3.11+ with the libraries listed above installed (e.g., via `pip install -r requirements.txt` if you maintain one, or manual installation through `pip install numpy pandas gstools seaborn scipy matplotlib scikit-image tqdm`).
- Random simulations rely on seeds defined within `simulations_paper.ipynb`; rerunning cells with different seeds yields independent batches but the same post-processing steps apply.
- When adding new simulation batches, keep each CSV aligned to multiples of 100 rows so that the filtering logic in `notebook_paper.ipynb` keeps full batches only.
