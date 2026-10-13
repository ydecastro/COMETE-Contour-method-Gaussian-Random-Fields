"""Planck 2018 CMB experiment, round 2 (EJP revision).

Pipeline, identical for the data and for the simulated isotropic maps:

1. Spherical harmonic coefficients of the temperature map (data: map2alm of the
   Commander 2018 map; simulations: synfast from the Planck 2018 best-fit TT
   spectrum with the same beam).
2. Filter: remove the multipoles ell < ELL_MIN (scales larger than a patch) and
   smooth to FWHM_OUT arcmin (the Commander map is at 5 arcmin); alm2map at
   NSIDE_OUT.
3. Flat patches by stereographic projection (conformal, so a locally isotropic
   field on the sphere stays locally isotropic on the plane), bilinear
   interpolation, PATCH_DEG degrees wide, PIX_ARCMIN arcmin per pixel, centred at
   high galactic latitude where the confidence mask is clean.
4. Contour analysis of each patch (comete_r2): level sets at standardised levels,
   Cabana estimators, cell statistics, chi-square p-values (scalar and full
   covariance) for several cell counts.

Data files live in ~/Data/planck (outside Dropbox). Results are CSV files in
results_planck/ (option --out). Run with a Python that has healpy installed:
    python planck_r2.py --help

The defaults below are the parameters of the paper (ell_min = 40, beam 15 arcmin,
nside 2048, 16 patches of 20 degrees, 1,000 x 1,000 pixels of 1.2 arcmin, N = 10,
100 simulated maps, seeds 1000 to 1099); the output files are then named
planck_data_ellmin40_fwhm15_patch20.csv and planck_sims_ellmin40_fwhm15_patch20_n100.csv.
The diagnostic runs of the paper (over-rejection of the degraded map) used
--nside-out 1024 --fwhm-out 10 with --ell-min 40, 100 and 300; see the README.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comete_r2 import FieldConfig, level_set_segments, contour_statistics, chi2_statistics, invert_g, oracle_gradient  # noqa: E402

DATA = Path.home() / "Data" / "planck"
MAP_FILE = DATA / "COM_CMB_IQU-commander_2048_R3.00_full.fits"
CL_FILE = DATA / "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"

# default pipeline parameters
ELL_MIN = 40          # remove ell < ELL_MIN (scales larger than about 4.5 degrees)
LMAX = 2000
FWHM_IN = 5.0         # arcmin, resolution of the Commander map
FWHM_OUT = 15.0       # arcmin, output resolution (extra smoothing applied to data and simulations); the paper's value
NSIDE_OUT = 2048      # HEALPix resolution of the filtered map, kept at the native resolution of the Commander map
PATCH_DEG = 20.0
PIX_ARCMIN = 1.2      # 20 deg / 1.2 arcmin = 1000 pixels
LEVELS_SD = (0.0, 1.0, -1.0)   # standardised levels u = mean + k * sd of the patch
N_LIST = (10, 5, 20, 25)
LAT_MIN = 40.0        # |b| >= LAT_MIN degrees


def patch_centres(n_per_hemisphere: int = 8, lat_min: float = LAT_MIN):
    """Galactic (lon, lat) centres, regularly spaced at |b| = lat_min + 10 and lat_min + 25."""
    cs = []
    for sign in (+1, -1):
        for k in range(n_per_hemisphere):
            lon = 360.0 * k / n_per_hemisphere
            lat = lat_min + (10.0 if k % 2 == 0 else 25.0)
            cs.append((lon, sign * lat))
    return cs


def stereographic_grid(lon0: float, lat0: float, patch_deg: float, pix_arcmin: float):
    """Directions (theta, phi) in radians of the pixels of a stereographic patch centred at
    galactic (lon0, lat0). The local frame has the patch centre on the x axis, as in the
    healpy projections, and healpy's Rotator(rot=(lon0, lat0, 0)) maps the x axis to (lon0, lat0)."""
    import healpy as hp
    npix = int(round(patch_deg * 60.0 / pix_arcmin))
    half = np.deg2rad(patch_deg) / 2.0
    u = np.linspace(-half, half, npix)
    X, Y = np.meshgrid(u, u)
    R = np.hypot(X, Y)
    rho = 2.0 * np.arctan(R / 2.0)              # angular distance from the centre (stereographic)
    az = np.arctan2(Y, X)
    # local direction: angle rho from the x axis, azimuth az around it
    vec_loc = np.stack([np.cos(rho.ravel()), np.sin(rho.ravel()) * np.cos(az.ravel()), np.sin(rho.ravel()) * np.sin(az.ravel())])
    # rotation matrix R = Rz(lon0) Ry(-lat0): maps the x axis to the direction (lon0, lat0)
    lo, la = np.deg2rad(lon0), np.deg2rad(lat0)
    Ry = np.array([[np.cos(la), 0.0, -np.sin(la)], [0.0, 1.0, 0.0], [np.sin(la), 0.0, np.cos(la)]])   # Ry(-lat0)
    Rz = np.array([[np.cos(lo), -np.sin(lo), 0.0], [np.sin(lo), np.cos(lo), 0.0], [0.0, 0.0, 1.0]])
    vec = (Rz @ Ry) @ vec_loc
    th, ph = hp.vec2ang(vec.T)          # vec2ang expects (N, 3)
    # self-check: the centre pixel must point to (lon0, lat0)
    c = npix // 2
    th_c, ph_c = th[c * npix + c], ph[c * npix + c]
    assert abs(np.rad2deg(np.pi / 2 - th_c) - lat0) < 0.05 and abs(((np.rad2deg(ph_c) - lon0) + 180) % 360 - 180) < 0.05, (lon0, lat0, np.rad2deg(np.pi/2-th_c), np.rad2deg(ph_c))
    return npix, th, ph


def project(m: np.ndarray, th: np.ndarray, ph: np.ndarray, npix: int, nearest: bool = False):
    import healpy as hp
    if nearest:
        nside = hp.get_nside(m)
        return m[hp.ang2pix(nside, th, ph)].reshape(npix, npix)
    return hp.get_interp_val(m, th, ph).reshape(npix, npix)


def filter_alm(alm, lmax: int, ell_min: int, fwhm_in: float, fwhm_out: float):
    """Zero the multipoles below ell_min and change the beam from fwhm_in to fwhm_out (arcmin)."""
    import healpy as hp
    ell = np.arange(lmax + 1)
    fl = np.ones(lmax + 1)
    fl[:ell_min] = 0.0
    if fwhm_out > fwhm_in:
        extra = np.sqrt(fwhm_out ** 2 - fwhm_in ** 2)
        fl *= hp.gauss_beam(np.deg2rad(extra / 60.0), lmax=lmax)
    return hp.almxfl(alm, fl)


def analyse_patch(Z: np.ndarray, pix_arcmin: float, levels_sd=LEVELS_SD, N_list=N_LIST):
    """Contour statistics of one flat patch at standardised levels. Physical unit: 1 degree."""
    spacing = pix_arcmin / 60.0
    n = Z.shape[0]
    T = n * spacing
    cfg = FieldConfig(model="gaussian", kappa=0.0, n=n, unit_length=1)  # only T and spacing are used
    mu, sd = float(Z.mean()), float(Z.std())
    kg, tg, _ = oracle_gradient(Z, spacing)
    rows = []
    for k in levels_sd:
        u = mu + k * sd
        seg = level_set_segments(Z, u, spacing)
        row = dict(level_sd=k, level=u, patch_mean=mu, patch_sd=sd, kappa_grad=kg, theta_grad=tg)
        if seg is None:
            rows.append(row)
            continue
        st = contour_statistics(seg, T, N_list[0])
        C, S, L = st["C"], st["S"], st["Ltot"]
        F = np.sqrt(C ** 2 + S ** 2) / L
        row.update(Ltot=L, C=C, S=S, F=F, kappa_C=invert_g(F), theta_C=float((0.5 * np.arctan2(S, C)) % np.pi))
        row.update(chi2_statistics(C, S, st["cell_C"], st["cell_S"]))
        for N in N_list[1:]:
            stN = contour_statistics(seg, T, N)
            for kk, v in chi2_statistics(C, S, stN["cell_C"], stN["cell_S"]).items():
                row[f"{kk}_N{N}"] = v
        rows.append(row)
    return rows


def load_cl(lmax: int):
    """TT power spectrum C_ell (muK^2) from the PLA best-fit file (columns: ell, D_ell TT, ...)."""
    d = np.loadtxt(CL_FILE)
    ell = d[:, 0].astype(int)
    Dl = d[:, 1]
    cl = np.zeros(lmax + 1)
    m = ell <= lmax
    cl[ell[m]] = Dl[m] * 2 * np.pi / (ell[m] * (ell[m] + 1.0))
    return cl


def run_data(args):
    import healpy as hp
    t0 = time.time()
    print("reading map", MAP_FILE, flush=True)
    I = hp.read_map(MAP_FILE, field=0, dtype=np.float64) * 1e6      # K_CMB -> muK
    mask = hp.read_map(MAP_FILE, field=3, dtype=np.float64)         # TMASK (1 = reliable)
    print(f"  nside {hp.get_nside(I)}, {time.time()-t0:.0f}s; map2alm", flush=True)
    alm = hp.map2alm(I, lmax=args.lmax, iter=1)
    alm = filter_alm(alm, args.lmax, args.ell_min, FWHM_IN, args.fwhm_out)
    mf = hp.alm2map(alm, nside=args.nside_out, lmax=args.lmax)
    mask_lo = hp.ud_grade(mask, args.nside_out)
    print(f"  filtered map ready, {time.time()-t0:.0f}s", flush=True)
    rows = []
    for (lon, lat) in patch_centres(args.n_per_hemisphere):
        npix, th, ph = stereographic_grid(lon, lat, args.patch_deg, args.pix_arcmin)
        Mk = project(mask_lo, th, ph, npix, nearest=True)
        frac_masked = float(np.mean(Mk < 0.5))
        if frac_masked > args.max_masked:
            print(f"  patch ({lon:.0f},{lat:.0f}) skipped: masked fraction {frac_masked:.3f}", flush=True)
            continue
        Z = project(mf, th, ph, npix)
        for r in analyse_patch(Z, args.pix_arcmin, N_list=args.N_cells):
            r.update(lon=lon, lat=lat, frac_masked=frac_masked, source="data")
            rows.append(r)
        print(f"  patch ({lon:.0f},{lat:.0f}) done, masked {frac_masked:.4f}, {time.time()-t0:.0f}s", flush=True)
        if args.save_patches:
            np.save(args.out / f"patch_data_lon{lon:.0f}_lat{lat:.0f}.npy", Z.astype(np.float32))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"planck_data_ellmin{args.ell_min}_fwhm{args.fwhm_out:g}_patch{args.patch_deg:g}.csv", index=False)
    print(df[["lon", "lat", "level_sd", "kappa_C", "theta_C", "p_scalar", "p_full", "kappa_grad"]].to_string(), flush=True)
    return df


def _sim_one(args_tuple):
    seed, args_dict = args_tuple
    import healpy as hp
    from types import SimpleNamespace
    args = SimpleNamespace(**args_dict)
    cl = load_cl(args.lmax)
    rng = np.random.default_rng(seed)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))       # synfast uses the legacy RNG
    alm = hp.synalm(cl, lmax=args.lmax, new=True)
    try:
        pw = hp.pixwin(2048, lmax=args.lmax)
    except Exception:
        pw = 1.0   # pixel window of nside 2048 (above 0.97 for ell <= 2000), negligible after the 15 arcmin smoothing
    alm = hp.almxfl(alm, hp.gauss_beam(np.deg2rad(FWHM_IN / 60.0), lmax=args.lmax) * pw)
    alm = filter_alm(alm, args.lmax, args.ell_min, FWHM_IN, args.fwhm_out)
    mf = hp.alm2map(alm, nside=args.nside_out, lmax=args.lmax)
    rows = []
    for (lon, lat) in patch_centres(args.n_per_hemisphere):
        npix, th, ph = stereographic_grid(lon, lat, args.patch_deg, args.pix_arcmin)
        Z = project(mf, th, ph, npix)
        for r in analyse_patch(Z, args.pix_arcmin, N_list=args.N_cells):
            r.update(lon=lon, lat=lat, seed=seed, source="sim")
            rows.append(r)
    return rows


def run_sims(args):
    import multiprocessing as mp
    t0 = time.time()
    args_dict = dict(lmax=args.lmax, ell_min=args.ell_min, fwhm_out=args.fwhm_out, nside_out=args.nside_out,
                     patch_deg=args.patch_deg, pix_arcmin=args.pix_arcmin, n_per_hemisphere=args.n_per_hemisphere,
                     N_cells=list(args.N_cells))
    jobs = [(args.seed0 + i, args_dict) for i in range(args.nsim)]
    rows = []
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_sim_one, jobs)):
            rows.extend(res)
            print(f"  sim {k+1}/{args.nsim}, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"planck_sims_ellmin{args.ell_min}_fwhm{args.fwhm_out:g}_patch{args.patch_deg:g}_n{args.nsim}.csv", index=False)
    for col in ["p_scalar", "p_full"] + [f"p_full_N{N}" for N in args.N_cells[1:]]:
        if col in df:
            print(f"{col}: rejection at 5% = {np.mean(df[col] < 0.05):.3f}, at 1% = {np.mean(df[col] < 0.01):.3f}, n = {df[col].notna().sum()}", flush=True)
    print(f"kappa_C on simulations: mean {df.kappa_C.mean():.3f}, sd {df.kappa_C.std():.3f}; kappa_grad mean {df.kappa_grad.mean():.3f}", flush=True)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["data", "sims", "both"])
    ap.add_argument("--ell-min", type=int, default=ELL_MIN)
    ap.add_argument("--lmax", type=int, default=LMAX)
    ap.add_argument("--fwhm-out", type=float, default=FWHM_OUT)
    ap.add_argument("--nside-out", type=int, default=NSIDE_OUT)
    ap.add_argument("--patch-deg", type=float, default=PATCH_DEG)
    ap.add_argument("--pix-arcmin", type=float, default=PIX_ARCMIN)
    ap.add_argument("--n-per-hemisphere", type=int, default=8)
    ap.add_argument("--max-masked", type=float, default=0.01)
    ap.add_argument("--N-cells", type=int, nargs="+", default=list(N_LIST))
    ap.add_argument("--nsim", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-patches", action="store_true")
    ap.add_argument("--out", default="results_planck")
    args = ap.parse_args(argv)
    args.out = Path(__file__).resolve().parent / args.out
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode in ("data", "both"):
        run_data(args)
    if args.mode in ("sims", "both"):
        run_sims(args)


if __name__ == "__main__":
    main()
