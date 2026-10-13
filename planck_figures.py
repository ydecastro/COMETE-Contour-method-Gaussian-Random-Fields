"""Figures of the Planck section (round 2), in the display style of version 1.

Outputs (in EJP_Round2/figures_R2/):
  planck_patch          one data patch (filtered temperature) and its level set at u = mean
  planck_pvalues        by level (three panels): CDF of the p-values on the simulated isotropic maps
                        (orange), the uniform CDF (dotted), and the data patches as blue circles
                        placed on the simulated CDF at their p-value
  planck_kappa          by level: histogram of kappa_C on the simulated maps (purple) with the data
                        patches as blue circles; red dashed line at the 95% quantile of the simulations
  planck_table.tex      table of the data patches; planck_summary.csv
Run:  python planck_figures.py [--ellmin 40 --fwhm 15 --out figures]
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f as fdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "figures"   # overridden by --out
RES = HERE / "results_planck"
sns.set_theme(context="paper", style="darkgrid", font="sans-serif", font_scale=1.9)
DPI = 200
FIGSIZE3 = (18, 6.2)
LEVELS = (0.0, 1.0, -1.0)
LEVEL_LABEL = {0.0: r"$u=\bar X$", 1.0: r"$u=\bar X+s$", -1.0: r"$u=\bar X-s$"}


def add_T2(df):
    if df is None:
        return None
    for N in [5, 10, 20, 25]:
        col = "Q_full" if N == 10 else f"Q_full_N{N}"
        if col in df:
            K = N * N
            df["p_T2" + ("" if N == 10 else f"_N{N}")] = fdist.sf(df[col] * (K - 2) / (2.0 * (K - 1)), 2, K - 2)
    return df


def load(ellmin=40, fwhm=15):
    data = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / f"planck_data_ellmin{ellmin}_fwhm{fwhm:g}_*.csv"))], ignore_index=True)
    sims_files = glob.glob(str(RES / f"planck_sims_ellmin{ellmin}_fwhm{fwhm:g}_*.csv"))
    sims = pd.concat([pd.read_csv(f) for f in sims_files], ignore_index=True) if sims_files else None
    return add_T2(data), add_T2(sims)


SUFFIX = ""


def save(fig, name):
    name = name + SUFFIX
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    print("saved", name)


def fig_patch():
    import contourpy
    files = sorted(glob.glob(str(RES / "patch_data_lon*_lat*.npy")))
    if not files:
        print("no saved patch"); return
    f = files[0]
    Z = np.load(f).astype(float); pix_arcmin = 1.2; ext = Z.shape[0] * pix_arcmin / 60.0
    lonlat = Path(f).stem.replace("patch_data_lon", "").replace("_lat", ", ")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), dpi=DPI)
    im = axes[0].imshow(Z, origin="lower", cmap="RdBu_r", extent=[0, ext, 0, ext], vmin=-3 * Z.std(), vmax=3 * Z.std())
    axes[0].set_title(rf"Filtered temperature ($\mu$K), centre $(l,b)=({lonlat})$"); axes[0].set_xlabel("degrees"); axes[0].set_ylabel("degrees"); axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cg = contourpy.contour_generator(z=Z, name="mpl2014"); res = cg.lines(float(Z.mean())); paths = res[0] if isinstance(res, tuple) else res
    for P in paths:
        axes[1].plot(P[:, 0] * pix_arcmin / 60.0, P[:, 1] * pix_arcmin / 60.0, color="black", lw=0.35)
    axes[1].set_aspect("equal"); axes[1].set_xlim(0, ext); axes[1].set_ylim(0, ext); axes[1].grid(False)
    axes[1].set_title(r"Level set $\{X=\bar X\}$"); axes[1].set_xlabel("degrees")
    fig.tight_layout(); save(fig, "planck_patch")


def fig_pvalues(data, sims):
    if sims is None:
        print("no simulations yet"); return
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI)
    for ax, lv in zip(axes, LEVELS):
        ps = np.sort(sims[np.isclose(sims.level_sd, lv)].p_T2.dropna().values)
        sns.ecdfplot(ps, ax=ax, color="orange", lw=2.6, label=r"$\chi^2\ \mathbb{H}_0$ (simulated isotropic maps)")
        ax.plot([0, 1], [0, 1], color="black", ls=":", lw=1.8, label=r"$\mathcal{U}(0,1)$")
        pd_ = data[np.isclose(data.level_sd, lv)].p_T2.values
        ax.scatter(pd_, np.searchsorted(ps, pd_) / len(ps), s=70, color="C0", edgecolor="k", linewidth=0.5, zorder=5, label="Planck patches")
        ax.axvline(0.05, color="red", ls="--", lw=2)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.01); ax.set_title(LEVEL_LABEL[lv]); ax.set_xlabel(r"$p$-value of $Q_{n,N}$ ($T^2$ threshold, $N=10$)"); ax.set_ylabel("CDF")
        ax.legend(fontsize=13, loc="lower right")
    n_sim = int(sims[np.isclose(sims.level_sd, 0.0)].shape[0])
    fig.suptitle(rf"Planck 2018 patches against {n_sim} patches of simulated isotropic maps, by level", y=1.02)
    fig.tight_layout(); save(fig, "planck_pvalues")


def fig_kappa(data, sims):
    if sims is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI)
    for ax, lv in zip(axes, LEVELS):
        x = sims[np.isclose(sims.level_sd, lv)].kappa_C.dropna().values
        sns.histplot(x, bins=30, stat="density", color="#a4148e", alpha=0.85, element="bars", ax=ax, label="simulated isotropic maps")
        q95 = np.quantile(x, 0.95); ax.axvline(q95, color="red", ls="--", lw=2.2, label="95% quantile of the simulations")
        xd = data[np.isclose(data.level_sd, lv)].kappa_C.values
        ax.scatter(xd, np.full_like(xd, 0.3), s=70, color="C0", edgecolor="k", linewidth=0.5, zorder=5, label="Planck patches")
        ax.set_title(LEVEL_LABEL[lv]); ax.set_xlabel(r"$\hat\kappa_C$"); ax.set_ylabel("Density"); ax.legend(fontsize=13)
    fig.suptitle(r"$\hat\kappa_C$ on the Planck patches and on the simulated isotropic maps, by level", y=1.02)
    fig.tight_layout(); save(fig, "planck_kappa")


def table(data, sims):
    lines = [r"\begin{tabular}{rrrrrrr}", r"\hline", r"$(l,b)$ & level & $\widehat\kappa_{\rm C}$ & $\widehat\theta_{\rm C}$ & $\widehat\kappa_{\rm grad}$ & $p$ ($T^2$, $N=10$) & $p$ ($T^2$, $N=20$) \\", r"\hline"]
    for _, r in data.sort_values(["lat", "lon", "level_sd"]).iterrows():
        lines.append(f"({r.lon:.0f},{r.lat:.0f}) & {LEVEL_LABEL[r.level_sd]} & {r.kappa_C:.3f} & {r.theta_C:.2f} & {r.kappa_grad:.3f} & {r.p_T2:.3f} & {r.p_T2_N20:.3f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (OUT / f"planck_table{SUFFIX}.tex").write_text("\n".join(lines) + "\n")
    summary = {}
    for lv in LEVELS:
        dd = data[np.isclose(data.level_sd, lv)]
        summary[lv] = dict(data_rej05=float(np.mean(dd.p_T2 < 0.05)), data_kappa_mean=float(dd.kappa_C.mean()), data_kappa_max=float(dd.kappa_C.max()), data_kappa_grad=float(dd.kappa_grad.mean()))
        if sims is not None:
            ss = sims[np.isclose(sims.level_sd, lv)]
            summary[lv].update(sim_rej05=float(np.mean(ss.p_T2 < 0.05)), sim_rej05_chi2=float(np.mean(ss.p_full < 0.05)), sim_kappa_mean=float(ss.kappa_C.mean()), sim_kappa_sd=float(ss.kappa_C.std()),
                               sim_kappa_q95=float(ss.kappa_C.quantile(0.95)), sim_kappa_grad=float(ss.kappa_grad.mean()), n_sim=int(len(ss)),
                               data_above_q95=int(np.sum(dd.kappa_C > ss.kappa_C.quantile(0.95))))
    S = pd.DataFrame(summary).T
    S.to_csv(OUT / f"planck_summary{SUFFIX}.csv")
    print(S.round(3).to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--ellmin", type=int, default=40); ap.add_argument("--fwhm", type=float, default=15); ap.add_argument("--out", default=str(OUT)); a = ap.parse_args()
    OUT = Path(a.out)
    SUFFIX = "" if (a.ellmin == 40 and a.fwhm == 15) else f"_ellmin{a.ellmin}_fwhm{a.fwhm:g}"
    OUT.mkdir(parents=True, exist_ok=True)
    data, sims = load(a.ellmin, a.fwhm)
    fig_patch(); fig_pvalues(data, sims); fig_kappa(data, sims); table(data, sims)
