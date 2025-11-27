# Anisotropic Gaussian Field Notebooks

This workspace gathers the notebooks and helper routines used to simulate anisotropic Gaussian random fields, extract Cabana/LKC statistics from their contour sets, and evaluate downstream hypothesis tests. The material is organized by workflow stage so individual notebooks stay focused and reproducible.

## Repository Layout

| Notebook | Role |
| --- | --- |
| `simulations_paper.ipynb` | End-to-end simulation loop generating Gaussian fields, extracting contour summaries, and persisting CSV outputs for later analysis. |
| `notebook_paper.ipynb` | Post-processing notebook that ingests simulation CSVs, compares theoretical vs empirical summaries, and builds the plots used in the paper. |
| `example.ipynb` | A concise, fully commented walkthrough of the simulation and estimator pipeline, intended as an onboarding aid for readers who want to reproduce a single run before tackling the full experiments. |
| `0_visualisation_paper*.ipynb` | Optional visualization notebooks illustrating intermediate grids, contour normals, and diagnostic figures. |

Supporting assets live under `data/` (input rasters and textures), `plots*/` (cached figures), and `results/` (saved tables).

## Getting Started

1. Create and activate a fresh Python 3.11+ environment.
2. Install the scientific stack (numpy, scipy, pandas, matplotlib, seaborn, scikit-image, gstools, tqdm). A `requirements.txt` can be generated via `pip freeze` after installing the exact versions you need.
3. Launch JupyterLab (or VS Code) in this folder and open the notebook that matches your workflow stage.

## Usage Notes

- All notebooks are self-contained and read/write paths relative to the repository root.
- `example.ipynb` is the recommended entry point: run it top-to-bottom to confirm dependencies and understand the Cabana/Biermé–Desolneux estimators before scaling up.
- `simulations_paper.ipynb` may take minutes to complete depending on grid size and number of Monte Carlo samples; adjust the configuration cells at the top of the notebook to trade accuracy for speed.

## Reproducing Paper Figures

1. Run `simulations_paper.ipynb` with the desired parameter grid; this generates the CSV files referenced in the paper.
2. Open `notebook_paper.ipynb`, update the file paths if needed, and run the plotting sections to reproduce the published figures and power analyses.
3. Use the visualization notebooks if you need qualitative insight into the simulated fields (e.g., gradient directions or contour normals).

## Contributing

- Keep all explanatory updates confined to markdown cells or inline comments when possible to preserve reproducibility.
- When adding new notebooks, document their scope in this README and prefer relative paths for any inputs/outputs.
- Large binary outputs should live under `results/` or `plots*/` to keep the repository tidy.
