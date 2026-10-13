"""COMETE, round 2 (EJP revision): simulation pipeline for the contour method.

What this module does, per realization of a stationary Gaussian field on a
square pixel grid:

1. Simulate the field exactly by circulant embedding (Wood & Chan 1994,
   Dietrich & Newsam 1997).  Three isotropic base correlations are available,
   each made anisotropic by an affine change of coordinates:
       gaussian : rho(h) = exp(-h^2)
       matern   : rho(h) = 2^{1-nu}/Gamma(nu) (sqrt(2 nu) h)^nu K_nu(sqrt(2 nu) h)
       cauchy   : rho(h) = (1 + h^2/alpha)^{-alpha}        (polynomial decay h^{-2 alpha})
   The length scales are chosen so that the gradient covariance is
   Lambda = P^T diag(a^2, a^-2) P with a = (1-kappa^2)^{-1/4}, exactly as in the
   Gaussian experiment of the submitted paper (kappa^2 = 1 - a^{-4}).  The first
   row of P is (cos theta0, sin theta0), the direction of largest gradient
   variance, i.e. the paper's theta0.
2. Extract the level set {X = u} by marching squares (contourpy, the algorithm
   behind matplotlib.contour).
3. Integrate cos(2 Theta), sin(2 Theta) and the outer product of the unit normal
   exactly along the polygonal level set (segment by segment).
4. Split the window into N x N cells and store the cell statistics (L_i, C_i, S_i).
5. Compute the Cabana estimators (kappa_C, theta_C), the two chi-square
   statistics (paper's eq. (e:V) with a pooled scalar variance, and the
   studentized version with the full 2x2 covariance of the cell vectors), the
   LKC estimator of Bierme-Desolneux (Euler characteristic by Gauss-Bonnet on
   the polygon and by scikit-image), and the oracle gradient estimator.

Run `python comete_r2.py --help`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy import special as sps
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2, norm

try:
    import contourpy
except ImportError as exc:  # pragma: no cover
    raise SystemExit("contourpy is required (it ships with matplotlib)") from exc

try:
    from skimage.measure import euler_number
except ImportError:  # pragma: no cover
    euler_number = None


# ---------------------------------------------------------------------------
# Elliptic links (same definitions as anisotropy_analysis.py of the paper)
# ---------------------------------------------------------------------------

def g_cabana(kappa: float) -> float:
    """g(kappa) of the paper, eq. (def:G): E_Palm[cos 2 Theta] when theta0 = 0."""
    k2 = float(kappa) ** 2

    def num(t):
        c2 = np.cos(t) ** 2
        return (2 * c2 - 1) * (1 - k2 * c2) ** (-1.5)

    def den(t):
        c2 = np.cos(t) ** 2
        return (1 - k2 * c2) ** (-1.5)

    a, _ = quad(num, 0, np.pi / 2, epsrel=1e-12, epsabs=1e-12)
    b, _ = quad(den, 0, np.pi / 2, epsrel=1e-12, epsabs=1e-12)
    return a / b


def R_bierme_desolneux(kappa: float) -> float:
    """R(kappa) = sqrt(1-kappa^2) / E(kappa)^2 of the paper (LKC ratio)."""
    k2 = float(kappa) ** 2
    E, _ = quad(lambda t: np.sqrt(1 - k2 * np.sin(t) ** 2), 0, np.pi / 2, epsrel=1e-12, epsabs=1e-12)
    return np.sqrt(1 - k2) / E ** 2


_KMAX = 0.999
_G_MAX = None
_R_MAX = None
_R_MIN = None


def invert_g(F: float) -> float:
    """kappa_C = g^{-1}(F), clipped to [0, 0.999]."""
    global _G_MAX
    if _G_MAX is None:
        _G_MAX = g_cabana(_KMAX)
    if not np.isfinite(F) or F <= 0:
        return 0.0
    if F >= _G_MAX:
        return _KMAX
    return float(brentq(lambda k: g_cabana(k) - F, 0.0, _KMAX, xtol=1e-9))


def invert_R(stat: float) -> float:
    """kappa_LKC = R^{-1}(stat) with the truncation rule of the paper
    (stat above R(0)=4/pi^2 -> 0, stat below R(0.999) -> 0.999)."""
    global _R_MAX, _R_MIN
    if _R_MAX is None:
        _R_MAX = R_bierme_desolneux(0.0)
        _R_MIN = R_bierme_desolneux(_KMAX)
    if not np.isfinite(stat):
        return np.nan
    if stat >= _R_MAX:
        return 0.0
    if stat <= _R_MIN:
        return _KMAX
    return float(brentq(lambda k: R_bierme_desolneux(k) - stat, 0.0, _KMAX, xtol=1e-9))


# ---------------------------------------------------------------------------
# Covariance models
# ---------------------------------------------------------------------------

def base_correlation(model: str, nu: float = 2.5, alpha: float = 1.5):
    """Return (rho, c) with rho the isotropic base correlation and c = -rho''(0)."""
    if model == "gaussian":
        rho = lambda h: np.exp(-h ** 2)
    elif model == "matern":
        s = np.sqrt(2.0 * nu)

        def rho(h):
            h = np.asarray(h, dtype=float)
            out = np.ones_like(h)
            m = h > 0
            hh = s * h[m]
            out[m] = np.exp((1 - nu) * np.log(2) - sps.gammaln(nu) + nu * np.log(hh)) * sps.kv(nu, hh)
            out[~np.isfinite(out)] = 0.0
            return out
    elif model == "cauchy":
        rho = lambda h: (1 + h ** 2 / alpha) ** (-alpha)
    else:
        raise ValueError(model)
    # c = -rho''(0) by a symmetric finite difference (robust to parametrisation)
    eps = 1e-3
    c = float(-(rho(np.array([2 * eps]))[0] - 2 * rho(np.array([eps]))[0] + 1.0) / eps ** 2) if model == "matern" \
        else float(-(rho(2 * eps) - 2 * rho(eps) + 1.0) / eps ** 2)
    return rho, c


@dataclass
class FieldConfig:
    model: str = "gaussian"      # gaussian | matern | cauchy
    kappa: float = 0.0           # anisotropy parameter in [0,1)
    theta0: float = 1.0          # anisotropy direction (radians), paper's theta0
    n: int = 1000                # pixels per side
    unit_length: int = 5         # pixels per physical unit (spacing = 1/unit_length)
    nu: float = 2.5              # Matern smoothness (C^2 paths need nu > 2)
    alpha: float = 1.5           # Cauchy exponent (integrable covariance needs alpha > 1)
    embed_factor: int = 2        # circulant embedding size = embed_factor * n (power of 2 rounded up)

    @property
    def spacing(self) -> float:
        return 1.0 / self.unit_length

    @property
    def T(self) -> float:
        return self.n * self.spacing

    @property
    def a(self) -> float:
        return (1.0 / (1.0 - self.kappa ** 2)) ** 0.25


def covariance_on_torus(cfg: FieldConfig, m: int) -> np.ndarray:
    """Covariance r(t) sampled on the m x m torus (pixel lags), physical units."""
    rho, c = base_correlation(cfg.model, cfg.nu, cfg.alpha)
    a = cfg.a
    # length scales along e1 (short, largest gradient variance a^2) and e2
    l1 = np.sqrt(c) / a
    l2 = np.sqrt(c) * a
    idx = np.arange(m)
    idx = np.minimum(idx, m - idx).astype(float) * cfg.spacing
    # array convention: Z[row, col]; physical x = col * spacing, y = row * spacing
    Y, X = np.meshgrid(idx, idx, indexing="ij")
    # torus symmetry: r must be even in each lag, so use |lag| along each axis
    # and evaluate the anisotropic norm on both sign combinations (max over
    # signs would be wrong); we evaluate r on the signed lag grid instead.
    sx = np.arange(m); sx = np.where(sx <= m // 2, sx, sx - m).astype(float) * cfg.spacing
    Ysg, Xsg = np.meshgrid(sx, sx, indexing="ij")
    ct, st = np.cos(cfg.theta0), np.sin(cfg.theta0)
    u1 = (Xsg * ct + Ysg * st) / l1
    u2 = (-Xsg * st + Ysg * ct) / l2
    h = np.sqrt(u1 ** 2 + u2 ** 2)
    return rho(h)


class CirculantSampler:
    """Exact sampler of the stationary Gaussian field on the n x n grid."""

    def __init__(self, cfg: FieldConfig):
        self.cfg = cfg
        m = 1
        while m < cfg.embed_factor * cfg.n:
            m *= 2
        # increase the torus until the circulant is (numerically) positive semi-definite
        for attempt in range(4):
            C = covariance_on_torus(cfg, m)
            lam = np.fft.fft2(C).real
            self.min_eig = float(lam.min() / lam.max())
            if lam.min() >= -1e-5 * lam.max():
                break
            m *= 2
        else:
            raise RuntimeError(f"circulant embedding not positive definite (min/max = {self.min_eig:.2e}) up to m = {m // 2}")
        self.m = m
        self.sqrt_lam = np.sqrt(np.maximum(lam, 0.0))

    def sample_pair(self, rng: np.random.Generator):
        m, n = self.m, self.cfg.n
        W = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        Z = np.fft.fft2(self.sqrt_lam * W) / m
        return Z.real[:n, :n].copy(), Z.imag[:n, :n].copy()


# ---------------------------------------------------------------------------
# Level-set statistics
# ---------------------------------------------------------------------------

def level_set_segments(Z: np.ndarray, level: float, spacing: float):
    """Marching-squares level set as arrays of segments in physical units.

    Returns dict with per-segment arrays: mid (M,2), vec (M,2), length (M,),
    and per-path turning-angle sums for the Gauss-Bonnet Euler characteristic.
    """
    cg = contourpy.contour_generator(z=Z, name="mpl2014", corner_mask=False)
    res = cg.lines(level)
    paths = res[0] if isinstance(res, tuple) else res   # SeparateCode returns (vertices, codes)
    mids, vecs, lens = [], [], []
    turning_total = 0.0
    n_closed = 0
    n_open = 0
    for P in paths:
        if P.shape[0] < 2:
            continue
        P = P * spacing
        d = np.diff(P, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        keep = L > 1e-12
        d, L = d[keep], L[keep]
        if d.shape[0] == 0:
            continue
        mids.append((P[:-1][keep] + P[1:][keep]) / 2.0)
        vecs.append(d)
        lens.append(L)
        closed = np.allclose(P[0], P[-1])
        phi = np.arctan2(d[:, 1], d[:, 0])
        dphi = np.diff(phi)
        if closed and d.shape[0] > 1:
            dphi = np.concatenate([dphi, [phi[0] - phi[-1]]])
            n_closed += 1
        else:
            n_open += 1
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        turning_total += float(dphi.sum())
    if not lens:
        return None
    return dict(
        mid=np.concatenate(mids),
        vec=np.concatenate(vecs),
        length=np.concatenate(lens),
        euler_gb=turning_total / (2 * np.pi),
        n_closed=n_closed,
        n_open=n_open,
    )


def contour_statistics(seg, T: float, N_cells: int):
    """Cabana integrals, normal covariance and cell statistics from segments."""
    L = seg["length"]
    v = seg["vec"]
    # unit normal (rotation of the tangent by -pi/2); the sign is irrelevant
    nx = v[:, 1] / L
    ny = -v[:, 0] / L
    cos2 = nx ** 2 - ny ** 2          # cos(2 Theta)
    sin2 = 2 * nx * ny                # sin(2 Theta)
    Ltot = float(L.sum())
    C = float((L * cos2).sum())
    S = float((L * sin2).sum())
    Sigma = np.array([[(L * nx * nx).sum(), (L * nx * ny).sum()], [(L * nx * ny).sum(), (L * ny * ny).sum()]]) / Ltot
    # cells: N x N partition of [0,T)^2, index by segment midpoint
    cell = T / N_cells
    ci = np.minimum((seg["mid"][:, 0] // cell).astype(int), N_cells - 1)
    cj = np.minimum((seg["mid"][:, 1] // cell).astype(int), N_cells - 1)
    flat = cj * N_cells + ci
    K = N_cells * N_cells
    cell_L = np.bincount(flat, weights=L, minlength=K)
    cell_C = np.bincount(flat, weights=L * cos2, minlength=K)
    cell_S = np.bincount(flat, weights=L * sin2, minlength=K)
    return dict(Ltot=Ltot, C=C, S=S, Sigma=Sigma, cell_L=cell_L, cell_C=cell_C, cell_S=cell_S)


def chi2_statistics(C: float, S: float, cell_C: np.ndarray, cell_S: np.ndarray):
    """Both chi-square statistics.

    Q_scalar: paper's eq. (e:V)-(e:qn): V^2 = sum[(C_i-Cbar)^2+(S_i-Sbar)^2]/(2(K-1)),
              Q = (C^2+S^2)/(K V^2), K = N^2 cells.
    Q_full  : studentized with the sample covariance of the cell vectors (C_i,S_i):
              Q = (C,S) Sigma_cells^{-1} (C,S)^T / K.
    """
    K = cell_C.size
    Cc = cell_C - cell_C.mean()
    Sc = cell_S - cell_S.mean()
    V2 = (np.sum(Cc ** 2) + np.sum(Sc ** 2)) / (2 * (K - 1))
    Q_scalar = (C ** 2 + S ** 2) / (K * V2) if V2 > 0 else np.nan
    Sig = np.array([[np.sum(Cc * Cc), np.sum(Cc * Sc)], [np.sum(Cc * Sc), np.sum(Sc * Sc)]]) / (K - 1)
    try:
        Q_full = float(np.array([C, S]) @ np.linalg.solve(Sig, np.array([C, S])) / K)
    except np.linalg.LinAlgError:
        Q_full = np.nan
    return dict(
        Q_scalar=Q_scalar, p_scalar=float(chi2.sf(Q_scalar, 2)) if np.isfinite(Q_scalar) else np.nan,
        Q_full=Q_full, p_full=float(chi2.sf(Q_full, 2)) if np.isfinite(Q_full) else np.nan,
        V2=V2, cellcov_xx=Sig[0, 0], cellcov_xy=Sig[0, 1], cellcov_yy=Sig[1, 1],
    )


def lkc_estimate(euler: float, Ltot: float, area_exc: float, T: float):
    """Bierme-Desolneux estimator, exactly as in the notebook of the paper:
    w = Phi^{-1}(area_exc/T^2) (= -u when mu=0, sigma=1), P = sqrt(2 pi) L/(2 T^2 phi(w)),
    GC = 2 pi chi/(w phi(w) T^2), stat = |GC/P^2|, kappa = R^{-1}(stat)."""
    frac = area_exc / T ** 2
    if not (0 < frac < 1) or Ltot <= 0:
        return np.nan, np.nan
    w = norm.ppf(frac)
    pw = norm.pdf(w)
    P = np.sqrt(2 * np.pi) * Ltot / (2 * T ** 2 * pw)
    if abs(w) < 1e-12:
        return np.nan, np.nan
    GC = 2 * np.pi * euler / (w * pw * T ** 2)
    stat = abs(GC / P ** 2)
    return stat, invert_R(stat)


def oracle_gradient(Z: np.ndarray, spacing: float):
    gy, gx = np.gradient(Z, spacing, spacing, edge_order=2)
    G = np.column_stack((gx.ravel(), gy.ravel()))
    Cov = np.cov(G, rowvar=False, bias=True)
    ev, evec = np.linalg.eigh(Cov)
    kappa = float(np.sqrt(max(0.0, 1 - ev[0] / ev[1])))
    v = evec[:, 1]
    theta = float(np.arctan2(v[1], v[0]) % np.pi)
    return kappa, theta, Cov


def analyse_field(Z: np.ndarray, cfg: FieldConfig, level: float, N_list):
    """All statistics of one realization at one level.

    Returns (row dict, cells) where cells are the (L_i, C_i, S_i) of the first
    N of N_list. Chi-square statistics are computed for every N in N_list; the
    first N gives the unsuffixed columns, the others get the suffix _N<N>.
    """
    T, sp = cfg.T, cfg.spacing
    N_list = list(N_list)
    row = dict(level=level, N_cells=N_list[0], field_mean=float(Z.mean()), field_var=float(Z.var()))
    seg = level_set_segments(Z, level, sp)
    if seg is None:
        row.update(Ltot=0.0)
        return row, None
    st = contour_statistics(seg, T, N_list[0])
    C, S, Ltot = st["C"], st["S"], st["Ltot"]
    F = np.sqrt(C ** 2 + S ** 2) / Ltot
    row.update(
        Ltot=Ltot, C=C, S=S, F=F,
        kappa_C=invert_g(F),
        theta_C=float((0.5 * np.arctan2(S, C)) % np.pi),
        Sigma_xx=st["Sigma"][0, 0], Sigma_xy=st["Sigma"][0, 1], Sigma_yy=st["Sigma"][1, 1],
        euler_gb=seg["euler_gb"], n_closed=seg["n_closed"], n_open=seg["n_open"],
    )
    row.update(chi2_statistics(C, S, st["cell_C"], st["cell_S"]))
    for N in N_list[1:]:
        stN = contour_statistics(seg, T, N)
        for k, v in chi2_statistics(C, S, stN["cell_C"], stN["cell_S"]).items():
            row[f"{k}_N{N}"] = v
    # LKC (Bierme-Desolneux) with three Euler characteristics
    exc = Z > level
    area_exc = float(exc.sum()) * sp ** 2
    row["area_exc"] = area_exc
    stat, k = lkc_estimate(seg["euler_gb"], Ltot, area_exc, T)
    row.update(stat_LKC_gb=stat, kappa_LKC_gb=k)
    if euler_number is not None:
        e1 = float(euler_number(exc, connectivity=1))
        e2 = float(euler_number(exc, connectivity=2))
        row.update(euler_sk1=e1, euler_sk2=e2)
        row["kappa_LKC_sk1"] = lkc_estimate(e1, Ltot, area_exc, T)[1]
        row["kappa_LKC_sk2"] = lkc_estimate(e2, Ltot, area_exc, T)[1]
    cells = np.stack([st["cell_L"], st["cell_C"], st["cell_S"]], axis=1)
    return row, cells


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_SAMPLER = None
_CFG = None


def _init_worker(cfg_dict):
    global _SAMPLER, _CFG
    _CFG = FieldConfig(**cfg_dict)
    _SAMPLER = CirculantSampler(_CFG)


def _work(args):
    seed, levels, N_list = args
    rng = np.random.default_rng(seed)
    Z1, Z2 = _SAMPLER.sample_pair(rng)
    out = []
    for which, Z in ((0, Z1), (1, Z2)):
        kg, tg, _ = oracle_gradient(Z, _CFG.spacing)
        for lv in levels:
            row, cells = analyse_field(Z, _CFG, lv, N_list)
            row.update(seed=seed, half=which, kappa_grad=kg, theta_grad=tg)
            out.append((row, cells))
    return out


def run_config(cfg: FieldConfig, levels, N_list, n_rep: int, seed0: int, workers: int, out_dir: Path, tag: str):
    """Run n_rep realizations (n_rep/2 FFTs, two fields each) and save CSV + NPZ."""
    import multiprocessing as mp
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    n_fft = (n_rep + 1) // 2
    jobs = [(seed0 + i, list(levels), list(N_list)) for i in range(n_fft)]
    t0 = time.time()
    rows, cells_all = [], {}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(asdict(cfg),)) as pool:
        for k, res in enumerate(pool.imap_unordered(_work, jobs, chunksize=1)):
            for row, cells in res:
                row.update(asdict(cfg))
                rows.append(row)
                if cells is not None:
                    cells_all[f"s{row['seed']}_h{row['half']}_u{row['level']}"] = cells
            if (k + 1) % max(1, n_fft // 10) == 0:
                print(f"  [{tag}] {2*(k+1)}/{2*n_fft} fields, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{tag}.csv", index=False)
    np.savez_compressed(out_dir / f"{tag}_cells.npz", **cells_all)
    dt = time.time() - t0
    print(f"[{tag}] done: {len(df)} rows in {dt:.0f}s ({dt/max(1,len(df)):.2f}s per row)", flush=True)
    return df


def summarize(df):
    import pandas as pd
    g = df.groupby(["model", "kappa", "level"])
    s = g.agg(
        n=("seed", "size"),
        Ltot=("Ltot", "median"),
        kappa_C_mean=("kappa_C", "mean"), kappa_C_sd=("kappa_C", "std"),
        theta_C_mean=("theta_C", "mean"), theta_C_sd=("theta_C", "std"),
        kappa_grad_mean=("kappa_grad", "mean"),
        kappa_LKC_gb_mean=("kappa_LKC_gb", "mean"), kappa_LKC_gb_sd=("kappa_LKC_gb", "std"),
        Q_scalar_mean=("Q_scalar", "mean"),
        rej05_scalar=("p_scalar", lambda p: np.mean(p < 0.05)),
        Q_full_mean=("Q_full", "mean"),
        rej05_full=("p_full", lambda p: np.mean(p < 0.05)),
    )
    pd.set_option("display.width", 250)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gaussian", choices=["gaussian", "matern", "cauchy"])
    ap.add_argument("--kappa", type=float, nargs="+", default=[0.0])
    ap.add_argument("--level", type=float, nargs="+", default=[0.0])
    ap.add_argument("--theta0", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--unit-length", type=int, default=5)
    ap.add_argument("--nu", type=float, default=2.5)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--embed-factor", type=int, default=2)
    ap.add_argument("--N-cells", type=int, nargs="+", default=[10, 5, 20, 25])
    ap.add_argument("--nrep", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    out_dir = Path(__file__).resolve().parent / args.out
    dfs = []
    for kappa in args.kappa:
        cfg = FieldConfig(model=args.model, kappa=kappa, theta0=args.theta0, n=args.n,
                          unit_length=args.unit_length, nu=args.nu, alpha=args.alpha,
                          embed_factor=args.embed_factor)
        tag = f"{args.model}_k{kappa:.2f}_n{args.n}_ul{args.unit_length}_N{args.N_cells[0]}_s{args.seed0}"
        if args.model == "matern":
            tag += f"_nu{args.nu}"
        if args.model == "cauchy":
            tag += f"_al{args.alpha}"
        # cap the number of workers by memory (about 4 complex arrays of m x m per worker)
        m = 1
        while m < args.embed_factor * args.n:
            m *= 2
        workers = max(1, min(args.workers, int(40e9 / (4 * 16 * m * m * 4))))
        print(f"== {tag}: levels {args.level}, nrep {args.nrep}, workers {workers}, N {args.N_cells}", flush=True)
        dfs.append(run_config(cfg, args.level, args.N_cells, args.nrep, args.seed0, workers, out_dir, tag))
    import pandas as pd
    df = pd.concat(dfs)
    print(summarize(df).to_string())


if __name__ == "__main__":
    main()
