# Version 2: the contour method with an exact sampler, four covariance families and a calibrated Planck pipeline

This folder is the code of the revised version of

> J.-M. Azaïs, F. Dalmao, Y. De Castro, *Gaussian random field's anisotropy using excursion sets*, arXiv:2512.11085.

It replaces the notebooks at the root of the repository for everything that concerns the numerical section of the paper. Two things changed with respect to version 1. The fields are now simulated exactly by circulant embedding: the randomization method of `gstools` with 1,000 Fourier modes, used in version 1, produces fields whose empirical gradient covariance on the window departs from the model by a random amount common to the whole realization, so that the "isotropic" fields of version 1 were anisotropic at the level $\kappa\approx0.4$. And the Planck experiment starts from the HEALPix map instead of an image, with a calibration on simulated isotropic maps.

## Layout

| File | Role |
| --- | --- |
| `COMETE_round2.ipynb` | Entry point. The pipeline on one field (exact simulation, level set and normals, contour statistics, the test with its finite-sample calibration, LKC and oracle estimators), a small Monte Carlo, the tables of the paper, and the Planck pipeline on one patch. Executed, so the outputs can be read on GitHub. |
| `comete_r2.py` | Library and command-line Monte Carlo: `python comete_r2.py --help`. |
| `figures_r2.py` | Figures and tables of Section 4 of the paper from the Monte Carlo outputs. |
| `planck_r2.py` | Planck 2018 experiment: filtering of the map, stereographic patches, contour test, simulated isotropic maps. `python planck_r2.py --help`. |
| `planck_figures.py` | Figures and table of the Planck section. |
| `resume_grid.py`, `resume_all.sh` | Launcher of the complete grid of the paper; skips the configurations already saved, so it can be interrupted and restarted. |
| `make_notebook.py` | Regenerates the notebook (then `jupyter nbconvert --execute`). |
| `figure1_r2.py` | Figure 1 of the paper (field, level set, partition, normals in one cell) with the exact sampler. |
| `results_summary/` | Summary tables of the runs reported in the paper (CSV and LaTeX), copies of the tables written by the figure scripts. The raw outputs (about 1.4 GB) are not versioned. |
| `figures/` | The figures of the paper (PNG). |

## Getting started

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab COMETE_round2.ipynb
```

`healpy` is needed only for the Planck experiment. With a python.org interpreter on macOS the
first `healpy` call may need root certificates: `pip install certifi` and
`export SSL_CERT_FILE=$(python -m certifi)`.

## Reproducing the paper's runs

All commands are run from this folder with a Python that has the packages of `requirements.txt`
(`healpy` for the Planck steps). The seeds are fixed (`--seed0 0` for the fields, `--seed0 1000`
for the simulated maps), so the number of workers changes the run time only. The command lines
below were reconstructed on 2026-10-05 from the launch scripts (`run_full.sh`, `run_full2.sh`,
`resume_grid.py`), the logs (which print the configuration of every run, and the exact command
for the runs launched by `resume_grid.py`), the names and headers of the result files and the
project change-log; the points that could not be established with certainty are stated as such.

### Main grid (Section 4, Tables of Appendix C)

Four covariance families, kappa in {0, 0.1, 0.25, 0.5, 0.9}, levels u in {0, 1, 2}, 5,000 fields per
configuration (2,500 FFTs, two fields each), N in {10, 5, 20, 25} cells per side, window T = 200
(1,000 x 1,000 pixels of spacing 0.2), theta_0 = 1. One command per family (the `--n 1000`,
`--unit-length 5`, `--theta0 1.0`, `--seed0 0` and `--embed-factor 2` defaults are the paper's):

```
python comete_r2.py --model gaussian                --kappa 0.0 0.1 0.25 0.5 0.9 --level 0 1 2 --nrep 5000 --workers 7 --N-cells 10 5 20 25 --out results_full
python comete_r2.py --model matern --nu 2.5         --kappa 0.0 0.1 0.25 0.5 0.9 --level 0 1 2 --nrep 5000 --workers 7 --N-cells 10 5 20 25 --out results_full
python comete_r2.py --model cauchy --alpha 1.5      --kappa 0.0 0.1 0.25 0.5 0.9 --level 0 1 2 --nrep 5000 --workers 7 --N-cells 10 5 20 25 --out results_full
python comete_r2.py --model cauchy --alpha 0.75     --kappa 0.0 0.1 0.25 0.5 0.9 --level 0 1 2 --nrep 5000 --workers 7 --N-cells 10 5 20 25 --out results_full
```

Outputs: `results_full/<family>_k<kappa>_n1000_ul5_N10_s0[_nu2.5|_al1.5|_al0.75].csv` (one row per
field and level) and `..._cells.npz` (the N x N cell statistics). The paper's files were produced in
three launches with the same arguments and different worker counts: `run_full.sh` (14 workers:
Gaussian, all kappa, and Matérn kappa = 0; log `local_logs/results_full.log`), `run_full2.sh`
(7 workers: Matérn kappa >= 0.1, Cauchy 1.5, Cauchy 0.75 kappa in {0, 0.1}; log `results_full2.log`)
and `resume_grid.py` (4 workers: Cauchy 0.75 kappa in {0.25, 0.5, 0.9} and the window study; log
`results_resume.log`). Run times per configuration (three levels, 5,000 fields): 120 to 131 s with
14 workers, 185 to 197 s with 7 workers, 322 s with 4 workers. The circulant embedding uses a
2,048 x 2,048 torus for every configuration except Cauchy alpha = 0.75 at kappa = 0.9, where the
sampler doubles the torus until the circulant is positive semi-definite and ends at 16,384 x 16,384
(2,048, 4,096 and 8,192 fail); this configuration took 17,467 s (4.9 hours) with 4 workers, each
worker holding 16,384 x 16,384 arrays (4 GB per complex array).

### Window study (Section 4.8, `window_table.csv`, `window_study.png`)

Gaussian and Cauchy alpha = 1.5, kappa in {0, 0.25}, T = 100 (n = 500, 5,000 fields, N in {5, 10})
and T = 400 (n = 2,000, 2,000 fields, N in {20, 10, 40, 50}); T = 200 is the main grid above.
Exact commands, as printed in `results_resume.log`:

```
python comete_r2.py --model gaussian            --kappa 0.0 0.25 --level 0 1 2 --nrep 5000 --n 500  --workers 4 --N-cells 5 10        --out results_full
python comete_r2.py --model gaussian            --kappa 0.0 0.25 --level 0 1 2 --nrep 2000 --n 2000 --workers 4 --N-cells 20 10 40 50 --out results_full
python comete_r2.py --model cauchy --alpha 1.5  --kappa 0.0 0.25 --level 0 1 2 --nrep 5000 --n 500  --workers 4 --N-cells 5 10        --out results_full
python comete_r2.py --model cauchy --alpha 1.5  --kappa 0.0 0.25 --level 0 1 2 --nrep 2000 --n 2000 --workers 4 --N-cells 20 10 40 50 --out results_full
```

(`resume_grid.py` runs one kappa per call; the result is the same.) Run times with 4 workers: 76 to
80 s per configuration at T = 100, 493 to 512 s at T = 400. The figure script evaluates the
T^2 test with N = 10 for every T (column `p_T2_N10`, from `Q_full_N10`, when N = 10 is not the first value of `--N-cells`).

Everything above, including the Planck simulations and the figure scripts, is also launched by
`./resume_all.sh`, which runs `resume_grid.py` with at most 4 workers (hard cap in the script, the
author's rule of 2026-09-23; edit `W = min(a.workers, 4)` to use more) and skips every
configuration whose CSV already exists.

### Planck 2018 (Section 4.9, `planck_table.tex`, `planck_summary.csv`, figures `planck_*`)

Download from the Planck Legacy Archive into `~/Data/planck/` the map
`COM_CMB_IQU-commander_2048_R3.00_full.fits` (1.6 GB) and the best-fit spectrum
`COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt`
(paths `MAP_FILE` and `CL_FILE` at the top of `planck_r2.py`). The defaults of `planck_r2.py` are
the parameters of the paper: multipoles ell < 40 removed, lmax = 2,000, beam 15 arcmin (from the
5 arcmin of the Commander map), HEALPix N_side = 2048 (the native resolution, no degradation),
sixteen stereographic patches of 20 degrees centred at galactic latitudes +-50 (longitudes 0, 90,
180, 270) and +-65 (longitudes 45, 135, 225, 315), 1,000 x 1,000 pixels of 1.2 arcmin, patches
with more than 1% of masked pixels skipped (none is), levels mean and mean +- sd of the patch,
N = 10 (plus 5, 20, 25), 100 simulated maps with seeds 1000 to 1099.

```
python planck_r2.py data --save-patches                  # -> results_planck/planck_data_ellmin40_fwhm15_patch20.csv, patch_data_lon*_lat*.npy
python planck_r2.py sims --nsim 100 --workers 7          # -> results_planck/planck_sims_ellmin40_fwhm15_patch20_n100.csv
```

The simulated maps are `synalm` draws from the Planck 2018 best-fit TT spectrum, multiplied by the
5 arcmin beam and the N_side = 2048 pixel window (omitted, with a value above 0.97 for ell <= 2000,
when healpy cannot download it), then filtered exactly as the data. Run times:
data, 10 s for the filtered map and about 0.2 s per patch (13 s in all, after reading the 1.6 GB
map); 100 simulated maps in 107 s (`local_logs/results_planck_sims_final.log`; the number of
workers of this final run is not recorded, 4 to 7). Both runs were made on 2026-09-24 at 01:23 and
01:25 (`results_planck_data_final.log`, `results_planck_sims_final.log`); the `--save-patches` flag
is attested by the time stamps of the `.npy` files, and the data values of the log are those of
the Planck table of the paper.

### Diagnostic runs (quoted in the text of Section 4.9; not in the tables)

The over-rejection of the test on simulated isotropic maps when the map is degraded. These runs
were made when the defaults of `planck_r2.py` were 10 arcmin and N_side = 1024, so the flags below
were not typed at the time; they are needed now.

```
python planck_r2.py data --nside-out 1024 --fwhm-out 10 --ell-min 40                 # 2026-09-23, planck_data_ellmin40_fwhm10_patch20.csv
python planck_r2.py sims --nside-out 1024 --fwhm-out 10 --ell-min 40  --nsim 100 --workers 4   # rejection at 5% (N = 10, chi2 threshold) 9.2%, 84 s
python planck_r2.py data --nside-out 1024 --fwhm-out 10 --ell-min 100                # planck_data_ellmin100_fwhm10_patch20.csv
python planck_r2.py sims --nside-out 1024 --fwhm-out 10 --ell-min 100 --nsim 100     # 9.6%, 86 s
python planck_r2.py data --nside-out 1024 --fwhm-out 10 --ell-min 300                # planck_data_ellmin300_fwhm10_patch20.csv
python planck_r2.py sims --nside-out 1024 --fwhm-out 10 --ell-min 300 --nsim 100     # 17.5%, 100 s
python planck_r2.py sims --nside-out 2048 --fwhm-out 10 --nsim 60 --out results_planck_diag_nside2048   # 7.0%, 66 s
python planck_r2.py sims --nside-out 1024 --fwhm-out 20 --nsim 60 --out results_planck_diag_fwhm20      # 7.5%, 48 s
```

(The final pipeline gives 6.5% with the chi2 threshold and 5.4% to 6.4% by level with the T^2
threshold, `results_planck_sims_final.log` and `planck_summary.csv`.)

Established from the file names, the output folders and the logs (`local_logs/results_planck_data*.log`,
`results_planck_sims*.log`, `results_planck_diag_*.log`). Not recorded: the number of workers of
these runs (the first `sims` run was launched by `resume_grid.py` with 4 workers; the others by
hand), and whether `data` and `sims` were launched separately or through the `both` mode (the logs
are separate files, which suggests separate launches; the results are identical either way).

### Figures, tables and Figure 1

```
python figures_r2.py --out figures           # Section 4 figures (PNG and PDF) and the tables summary_table.csv, summary_table_<family>.tex, power_table.csv, window_table.csv
python planck_figures.py --out figures       # planck_patch, planck_pvalues, planck_kappa, planck_table.tex, planck_summary.csv (defaults --ellmin 40 --fwhm 15)
python planck_figures.py --ellmin 40 --fwhm 10 --out figures    # same figures for a diagnostic configuration, files suffixed _ellmin40_fwhm10 (also 100 and 300)
python figure1_r2.py --out figures           # Figure 1: Gaussian field, kappa = 0.9, theta_0 = 1, seed 0, level 0, 8 x 8 partition, zoom on the cell (2, 7)
```

`figures_r2.py` reads `results_full/` (`--results` to change) and the window runs are recognised by
their `n`. For the paper the three scripts were run with `--out` pointing to the Overleaf folder
`EJP_Round2/figures_R2`; `results_summary/` holds copies of the five tables (identical to the
Overleaf copies, checked on 2026-10-05), and `figures/` the PNG figures. The diagnostic Planck
figures of the Overleaf folder (`planck_*_ellmin100`, `planck_*_ellmin300`) were produced before the
`--fwhm` option existed; the current script suffixes them `_ellmin100_fwhm10` and `_ellmin300_fwhm10`.

## Notes

- The finite-sample calibration of the test uses Hotelling's $T^2$ law: with $K=N^2$ cells,
  $Q\,(K-2)/(2(K-1))$ is compared with the Fisher law $F(2,K-2)$; the $\chi^2(2)$ threshold is
  the limit for large $N$.
- In the Planck pipeline the map must be kept at its native resolution before the projection.
  Interpolating a degraded map onto the fine grid produces a faceted structure oriented along
  the HEALPix pixels, which the test reads as anisotropy; the calibration on simulated isotropic
  maps is part of the procedure.
- Seeds: the Monte Carlo runs use `seed0 + i` for the $i$-th FFT (two fields per FFT); the
  simulated CMB maps use the seeds 1000 to 1099.
