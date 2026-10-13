"""Figures and tables of the numerical section (round 2), in the display style of version 1
(seaborn dark grid, one row of three panels, level encoded by marker and colour, red star for the
true value, red dashed reference lines, stacked histograms, ROC-type power curves).

Usage:  python3 figures_r2.py [--results results_full] [--out <EJP_Round2/figures_R2>]

Main text (Gaussian family, T = 200, N = 10):
  Cabana_sine_cosine_gaussian   (C^, S^) by kappa, levels as markers               [fig:Cabana_MB]
  Cabana_angle_gaussian         histograms of theta^_C by kappa                     [fig:angle_cabana]
  marginal_density_gaussian     histograms of kappa^_C and kappa^_LKC by kappa       [fig:density_kappa_estimates_MB]
  joint_lkc_gaussian            scatter kappa^_LKC vs kappa^_C by kappa             [fig:joint_lkc]
  gradient_contour_gaussian     scatter kappa^_grad vs kappa^_C by kappa            [fig:fullobs_cabana]
  fig-powers_gaussian           ROC-type power curves, H1: kappa = 0.25, by level    [fig:power]
Family comparisons (all families):
  pvalue_cdf_null               CDF of the null p-values by level, lines per family  [fig:pvalue_cdf]
  calibration_vs_N              rejection rate at 5% vs N by level, lines per family [fig:calibration_N]
  power_vs_kappa                rejection rate at 5% vs kappa by level, per family   [fig:power_kappa]
  window_study                  rejection rate vs T, Gaussian and Cauchy 1.5         [fig:window]
Appendix: fig-powers_<family> for the other families; summary tables per family.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f as fdist

# ---------------------------------------------------------------------------
# Style of version 1
sns.set_theme(context="paper", style="darkgrid", font="sans-serif", font_scale=1.9)
DPI = 200
FIGSIZE3 = (18, 6.2)
LEVELS = (0.0, 1.0, 2.0)
LEVEL_STYLE = {0.0: dict(marker="o", color="C0"), 1.0: dict(marker="s", color="C1"), 2.0: dict(marker="^", color="C2")}
PURPLES = {0.0: "#6a0d5c", 1.0: "#a4148e", 2.0: "#d63fbf"}
FAMILIES = ["gaussian", "matern", "cauchy15", "cauchy075"]
FAM_LABEL = {"gaussian": "Gaussian", "matern": r"Matérn $\nu=2.5$", "cauchy15": r"Cauchy $\alpha=1.5$", "cauchy075": r"Cauchy $\alpha=0.75$"}
FAM_STYLE = {"gaussian": dict(color="C0", marker="o"), "matern": dict(color="C1", marker="s"), "cauchy15": dict(color="C2", marker="^"), "cauchy075": dict(color="C3", marker="D")}
KAPPAS3 = (0.0, 0.5, 0.9)
THETA0 = 1.0
NMAX = 2500          # points displayed per level in the scatter plots


def family(row):
    if row["model"] == "cauchy":
        return "cauchy15" if abs(row["alpha"] - 1.5) < 1e-9 else "cauchy075"
    return row["model"]


def add_hotelling(df):
    """Finite-sample calibration: with K = N^2 cell vectors, Q_full (K-2)/(2(K-1)) ~ F(2, K-2).
    The unsuffixed columns correspond to the first N of the run (column N_cells); the others are suffixed."""
    K0 = (df["N_cells"] ** 2).astype(float)
    df["p_T2"] = fdist.sf(df["Q_full"] * (K0 - 2) / (2.0 * (K0 - 1)), 2, K0 - 2)
    for N in [5, 10, 20, 25, 40, 50]:
        col = f"Q_full_N{N}"
        if col in df:
            K = N * N
            df[f"p_T2_N{N}"] = fdist.sf(df[col] * (K - 2) / (2.0 * (K - 1)), 2, K - 2)
    return df


def p_T2_for_N(df, N):
    """p-value with the T^2 threshold for N cells per side, whatever the first N of the run."""
    out = pd.Series(np.nan, index=df.index)
    m = df["N_cells"] == N
    out[m] = df.loc[m, "p_T2"]
    col = f"p_T2_N{N}"
    if col in df:
        mm = (~m) & df[col].notna()
        out[mm] = df.loc[mm, col]
    return out


def load(results_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(results_dir / "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["family"] = df.apply(family, axis=1)
    df["T"] = df["n"] / df["unit_length"]
    return add_hotelling(df)


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def sub(df, fam, T=200):
    return df[(df.family == fam) & (df["T"] == T)]


def level_scatter(ax, x, y, lv, label=None, s=45, alpha=0.7):
    st = LEVEL_STYLE[lv]
    ax.scatter(x, y, s=s, alpha=alpha, marker=st["marker"], color=st["color"], edgecolor="k", linewidth=0.4, label=label or f"u={int(lv)}")


def star(ax, x, y, label):
    ax.scatter([x], [y], marker="*", s=320, color="red", edgecolor="k", linewidth=0.6, zorder=5, label=label)


# ---------------------------------------------------------------------------
def fig_cosine_sine(df, out, fam):
    from comete_r2 import g_cabana
    d = sub(df, fam)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI)
    for ax, k in zip(axes, KAPPAS3):
        g = d[np.isclose(d.kappa, k)]
        xs = np.linspace(-0.65, 0.65, 10)
        ax.plot(xs, xs * np.tan(2 * THETA0), "--", color="red", alpha=0.45, linewidth=2.2, label=r"$S=C\,\tan(2\theta_0)$")
        gk = g_cabana(k)
        star(ax, gk * np.cos(2 * THETA0), gk * np.sin(2 * THETA0), r"$g(\kappa)(\cos 2\theta_0,\ \sin 2\theta_0)$")
        for lv in LEVELS:
            h = g[np.isclose(g.level, lv)].head(NMAX)
            level_scatter(ax, h.C / h.Ltot, h.S / h.Ltot, lv, s=30)
        ax.axhline(0, color="grey", ls="--", lw=1); ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set_xlim(-0.65, 0.65); ax.set_ylim(-0.65, 0.65); ax.set_aspect("equal")
        ax.set_title(rf"$\kappa={k}$"); ax.set_xlabel(r"Empirical cosine $\hat C$"); ax.set_ylabel(r"Empirical sine $\hat S$")
        ax.legend(loc="lower left", fontsize=14, markerscale=1.4)
    fig.suptitle(rf"Joint density of empirical cosine and sine by $\kappa$ (level u=0,1,2), {FAM_LABEL[fam]}", y=1.02)
    fig.tight_layout()
    save(fig, out, f"Cabana_sine_cosine_{fam}")


def fig_angle(df, out, fam):
    d = sub(df, fam)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI)
    for ax, k in zip(axes, KAPPAS3):
        g = d[np.isclose(d.kappa, k)]
        for lv in LEVELS:
            th = g[np.isclose(g.level, lv)].theta_C.values
            th = np.where(th > np.pi / 2 + THETA0, th - np.pi, th)          # centre the circle on theta_0
            bins = np.linspace(THETA0 - np.pi / 2, THETA0 + np.pi / 2, 40) if k == 0 else 40
            sns.histplot(th, bins=bins, stat="density", color=PURPLES[lv], alpha=0.85, element="bars", ax=ax, label=f"u={int(lv)}")
        ax.axvline(THETA0, color="red", ls="--", lw=2.2, label=r"$\theta_0$")
        ax.set_title(rf"$\kappa={k}$"); ax.set_xlabel(r"$\hat\theta_C$"); ax.set_ylabel("Density")
        ax.legend(fontsize=14)
    fig.suptitle(rf"Densities of $\hat\theta_C$ by true $\kappa$ (levels $u=0,1,2$), {FAM_LABEL[fam]}", y=1.02)
    fig.tight_layout()
    save(fig, out, f"Cabana_angle_{fam}")


def fig_marginal(df, out, fam):
    d = sub(df, fam)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI)
    for ax, k in zip(axes, KAPPAS3):
        g = d[np.isclose(d.kappa, k)]
        bins = np.linspace(0, 1, 41)
        for lv in LEVELS:
            h = g[np.isclose(g.level, lv)]
            sns.histplot(h.kappa_C.dropna(), bins=bins, stat="density", color=PURPLES[lv], alpha=0.85, element="bars", ax=ax, label=f"Contour, u={int(lv)}")
        for lv in LEVELS:
            h = g[np.isclose(g.level, lv)]
            sns.histplot(h.kappa_LKC_gb.dropna(), bins=bins, stat="density", color=LEVEL_STYLE[lv]["color"], alpha=0.75, element="bars", ax=ax, label=f"LKC, u={int(lv)}")
        ax.axvline(k, color="red", ls="--", lw=2.2)
        # log scale, same range on the three panels: one realization in a bin of width 0.025 among 5,000
        # has density 0.008, the largest bin about 40 (LKC atom at 0, contour peak at kappa = 0.9)
        ax.set_yscale("log"); ax.set_ylim(5e-3, 60)
        ax.set_title(rf"$\kappa={k}$"); ax.set_xlabel(r"Estimated $\kappa$"); ax.set_ylabel("Density")
    fig.suptitle(rf"Marginal densities of $\hat\kappa_C$ (purples) and $\hat\kappa_{{LKC}}$ (blue/orange/green) (for level $u=0,1,2$), {FAM_LABEL[fam]}", y=1.02)
    fig.tight_layout()
    # one legend below the three panels, so that no bar is covered
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=15, frameon=False)
    save(fig, out, f"marginal_density_{fam}")


def fig_joint(df, out, fam, xcol, name, xlabel, suptitle):
    d = sub(df, fam)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI)
    for ax, k in zip(axes, KAPPAS3):
        g = d[np.isclose(d.kappa, k)]
        for lv in LEVELS:
            h = g[np.isclose(g.level, lv)].head(NMAX)
            level_scatter(ax, h[xcol], h.kappa_C, lv)
        ax.plot([-0.05, 1.05], [-0.05, 1.05], "r--", lw=1.3, alpha=0.75)
        star(ax, k, k, r"$\kappa$ true")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_title(rf"$\kappa={k}$"); ax.set_xlabel(xlabel); ax.set_ylabel(r"$\hat\kappa_C$")
        ax.legend(fontsize=14, loc="upper left")
    fig.suptitle(suptitle + f", {FAM_LABEL[fam]}", y=1.02)
    fig.tight_layout()
    save(fig, out, f"{name}_{fam}")


# ---------------------------------------------------------------------------
def fig_powers(df, out, fam, kappa1=0.25):
    """ROC-type power curves as in version 1: power against significance for H1: kappa = kappa1."""
    d = sub(df, fam)
    h0 = d[np.isclose(d.kappa, 0.0)]; h1 = d[np.isclose(d.kappa, kappa1)]
    if h0.empty or h1.empty:
        print(f"  powers {fam}: missing data"); return
    alphas = np.linspace(0.0001, 0.99, 400)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI)
    for ax, lv in zip(axes, LEVELS):
        a0 = h0[np.isclose(h0.level, lv)]; a1 = h1[np.isclose(h1.level, lv)]
        # model-based tests: quantiles of the statistic under H0, rejection frequency under H1
        for stat, direction, color, ls, label in [("kappa_grad", ">", "magenta", "--", "MB-Oracle"), ("F", ">", "blue", "-", "MB-Contour"), ("stat_LKC_gb", "<", "purple", "-", "MB-LKC")]:
            s0 = a0[stat].dropna().to_numpy(); s1 = a1[stat].dropna().to_numpy()
            if direction == ">":
                q = np.percentile(s0, 100 * (1 - alphas)); power = (s1[:, None] > q[None, :]).mean(axis=0)
            else:
                q = np.percentile(s0, 100 * alphas); power = (s1[:, None] < q[None, :]).mean(axis=0)
            ax.plot(alphas, power, color=color, ls=ls, lw=2.4, label=label)
        ax.plot([0, 1], [0, 1], color="black", ls=":", lw=1.8, label=r"$\mathcal{U}(0,1)$")
        sns.ecdfplot(a1.p_T2.dropna(), ax=ax, color="green", lw=2.4, label=r"$\chi^2\ \mathbb{H}_1$")
        sns.ecdfplot(a0.p_T2.dropna(), ax=ax, color="orange", lw=2.4, label=r"$\chi^2\ \mathbb{H}_0$")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.01)
        ax.set_title(rf"$u={int(lv)}\ |\ n_0={len(a0)}\ |\ n_1={len(a1)}$")
        ax.set_xlabel(r"Significance ($\alpha$)"); ax.set_ylabel(r"Power (1 - $\beta$)")
        ax.legend(fontsize=14, loc="lower right")
    fig.suptitle(rf"$\mathbb{{H}}_0:\ \kappa=0\ $ vs $\ \mathbb{{H}}_1:\ \kappa={kappa1}$, {FAM_LABEL[fam]}", y=1.02)
    fig.tight_layout()
    save(fig, out, f"fig-powers_{fam}")


# ---------------------------------------------------------------------------
def fams_present(df):
    return [f for f in FAMILIES if f in set(df.family)]


def fig_pvalue_cdf(df, out):
    d = df[(df["T"] == 200) & np.isclose(df.kappa, 0.0)]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI)
    for ax, lv in zip(axes, LEVELS):
        for fam in fams_present(d):
            p = d[(d.family == fam) & np.isclose(d.level, lv)].p_T2.dropna()
            if len(p):
                sns.ecdfplot(p, ax=ax, color=FAM_STYLE[fam]["color"], lw=2.4, label=FAM_LABEL[fam])
        ax.plot([0, 1], [0, 1], color="black", ls=":", lw=1.8, label=r"$\mathcal{U}(0,1)$")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.01)
        ax.set_title(rf"$u={int(lv)}$"); ax.set_xlabel(r"$p$-value of $Q_{n,N}$ under $\mathbb{H}_0$"); ax.set_ylabel("CDF")
        ax.legend(fontsize=14, loc="lower right")
    fig.suptitle(r"Calibration under quasi-isotropy ($\kappa=0$, $T=200$, $N=10$, $T^2$ threshold) by covariance family", y=1.02)
    fig.tight_layout()
    save(fig, out, "pvalue_cdf_null")


def fig_calibration_N(df, out):
    d = df[(df["T"] == 200) & np.isclose(df.kappa, 0.0)]
    Ns = [5, 10, 20, 25]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI, sharey=True)
    for ax, lv in zip(axes, LEVELS):
        for fam in fams_present(d):
            h = d[(d.family == fam) & np.isclose(d.level, lv)]
            rT = [np.mean(h["p_T2" if N == 10 else f"p_T2_N{N}"].dropna() < 0.05) for N in Ns]
            rC = [np.mean(h["p_full" if N == 10 else f"p_full_N{N}"].dropna() < 0.05) for N in Ns]
            st = FAM_STYLE[fam]
            ax.plot(Ns, rT, marker=st["marker"], ms=9, mec="k", color=st["color"], lw=2.2, label=FAM_LABEL[fam] + r" ($T^2$)")
            ax.plot(Ns, rC, marker=st["marker"], ms=7, mec="k", color=st["color"], lw=1.4, ls="--", alpha=0.6, label=FAM_LABEL[fam] + r" ($\chi^2$)")
        ax.axhline(0.05, color="red", ls="--", lw=2)
        ax.set_xticks(Ns); ax.set_title(rf"$u={int(lv)}$"); ax.set_xlabel(r"$N$ (cells per side)")
        ax.set_ylabel("Rejection rate at 5%" if lv == 0 else "")
    axes[0].legend(fontsize=11, ncol=2)
    fig.suptitle(r"Rejection rate under quasi-isotropy against the number of cells ($\kappa=0$, $T=200$)", y=1.02)
    fig.tight_layout()
    save(fig, out, "calibration_vs_N")


def power_table(df):
    rows = []
    d = df[df["T"] == 200]
    for fam in fams_present(d):
        for lv in LEVELS:
            null = d[(d.family == fam) & np.isclose(d.kappa, 0.0) & np.isclose(d.level, lv)]
            if len(null) < 100:
                continue
            qF = np.nanquantile(null.F, 0.95); qR = np.nanquantile(null.stat_LKC_gb, 0.05); qG = np.nanquantile(null.kappa_grad, 0.95)
            for k in sorted(d.kappa.unique()):
                g = d[(d.family == fam) & np.isclose(d.kappa, k) & np.isclose(d.level, lv)]
                if len(g) == 0:
                    continue
                rows.append(dict(family=fam, level=lv, kappa=k, n=len(g), chi2_full=np.mean(g.p_full < 0.05), chi2_T2=np.mean(g.p_T2 < 0.05),
                                 chi2_scalar=np.mean(g.p_scalar < 0.05), MB_contour=np.mean(g.F > qF), MB_LKC=np.mean(g.stat_LKC_gb < qR), MB_oracle=np.mean(g.kappa_grad > qG)))
    return pd.DataFrame(rows)


def fig_power_kappa(df, out):
    pt = power_table(df)
    pt.to_csv(out / "power_table.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE3, dpi=DPI, sharey=True)
    for ax, lv in zip(axes, LEVELS):
        for fam in fams_present(pt):
            g = pt[(pt.family == fam) & np.isclose(pt.level, lv)].sort_values("kappa")
            st = FAM_STYLE[fam]
            ax.plot(g.kappa, g.chi2_T2, marker=st["marker"], ms=9, mec="k", color=st["color"], lw=2.2, label=rf"$\chi^2$-Contour, {FAM_LABEL[fam]}")
        g = pt[(pt.family == "gaussian") & np.isclose(pt.level, lv)].sort_values("kappa")
        if len(g):
            ax.plot(g.kappa, g.MB_LKC, marker="v", ms=8, mec="k", color="purple", lw=2.0, ls="--", label="MB-LKC, Gaussian")
            ax.plot(g.kappa, g.MB_oracle, ls=":", color="magenta", lw=2.4, label="MB-Oracle, Gaussian")
        ax.axhline(0.05, color="red", ls="--", lw=1.6)
        ax.set_xticks([0, 0.1, 0.25, 0.5, 0.9]); ax.set_xlim(-0.03, 0.95); ax.set_ylim(0, 1.02)
        ax.set_title(rf"$u={int(lv)}$"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("Rejection rate at 5%" if lv == 0 else "")
    axes[2].legend(fontsize=12, loc="lower right")
    fig.suptitle(r"Rejection rate at the nominal level 5% against $\kappa$ ($T=200$, $N=10$, $T^2$ threshold)", y=1.02)
    fig.tight_layout()
    save(fig, out, "power_vs_kappa")
    return pt


def fig_window(df, out):
    """Rejection rate against the window side T, with N = 10 fixed (cells grow with T) and with
    cells of fixed side 20 (N = T/20 grows with T)."""
    d = df[df.family.isin(["gaussian", "cauchy15"]) & df.kappa.isin([0.0, 0.25])]
    if d["T"].nunique() < 2:
        print("  window study: not enough window sizes yet"); return None
    rows = []
    for fam in ("gaussian", "cauchy15"):
        for k in (0.0, 0.25):
            for lv in LEVELS:
                for T in sorted(d["T"].unique()):
                    g = d[(d.family == fam) & np.isclose(d.kappa, k) & np.isclose(d.level, lv) & (d["T"] == T)]
                    if len(g) == 0:
                        continue
                    r10 = np.nanmean(p_T2_for_N(g, 10) < 0.05)
                    Nfix = int(round(T / 20))
                    rfix = np.nanmean(p_T2_for_N(g, Nfix) < 0.05)
                    rows.append(dict(family=fam, kappa=k, level=lv, T=T, n=len(g), rej_N10=r10, N_fixed_cell=Nfix, rej_fixed_cell=rfix, kC_mean=g.kappa_C.mean(), kC_sd=g.kappa_C.std()))
    wt = pd.DataFrame(rows); wt.to_csv(out / "window_table.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), dpi=DPI)
    for ax, k in zip(axes, (0.0, 0.25)):
        for fam, ls in (("gaussian", "-"), ("cauchy15", "--")):
            for lv in LEVELS:
                g = wt[(wt.family == fam) & np.isclose(wt.kappa, k) & np.isclose(wt.level, lv)].sort_values("T")
                st = LEVEL_STYLE[lv]
                ax.plot(g["T"], g.rej_N10, marker=st["marker"], ms=9, mec="k", color=st["color"], ls=ls, lw=2.2, label=f"{FAM_LABEL[fam]}, u={int(lv)}")
        ax.axhline(0.05, color="red", ls="--", lw=2)
        ax.set_xscale("log", base=2); ax.set_xticks([100, 200, 400]); ax.set_xticklabels(["100", "200", "400"])
        ax.set_xlabel(r"Window side $T$"); ax.set_ylabel("Rejection rate at 5%"); ax.set_title(rf"$\kappa={k}$ ($N=10$, $T^2$ threshold)")
        ax.legend(fontsize=12)
    fig.suptitle("Calibration and power against the window size", y=1.02)
    fig.tight_layout()
    save(fig, out, "window_study")
    return wt


# ---------------------------------------------------------------------------
def summary_tables(df, out):
    d = df[df["T"] == 200]
    g = d.groupby(["family", "kappa", "level"]).agg(
        n=("seed", "size"), kC_mean=("kappa_C", "mean"), kC_sd=("kappa_C", "std"),
        th_sd=("theta_C", lambda x: np.std(np.where(x > np.pi / 2 + THETA0, x - np.pi, x))),
        kLKC_mean=("kappa_LKC_gb", "mean"), kLKC_sd=("kappa_LKC_gb", "std"), kgrad_mean=("kappa_grad", "mean"),
        rej_full=("p_full", lambda p: np.mean(p < 0.05)), rej_T2=("p_T2", lambda p: np.mean(p < 0.05)), rej_scalar=("p_scalar", lambda p: np.mean(p < 0.05)),
    ).reset_index()
    g.to_csv(out / "summary_table.csv", index=False)
    for fam in fams_present(g):
        gg = g[g.family == fam]
        L = [r"\begin{tabular}{rrrrrrrrr}", r"\hline",
             r"$\kappa$ & $u$ & $\widehat\kappa_{\rm C}$ mean (sd) & $\widehat\theta_{\rm C}$ sd & $\widehat\kappa_{\rm LKC}$ mean (sd) & $\widehat\kappa_{\rm grad}$ mean & $Q_{n,N}$, $\chi^2$ & $Q_{n,N}$, $T^2$ & $Q^{\rm iso}_{n,N}$ \\", r"\hline"]
        for _, r in gg.iterrows():
            L.append(f"{r.kappa:g} & {int(r.level)} & {r.kC_mean:.3f} ({r.kC_sd:.3f}) & {r.th_sd:.3f} & {r.kLKC_mean:.3f} ({r.kLKC_sd:.3f}) & {r.kgrad_mean:.3f} & {r.rej_full:.3f} & {r.rej_T2:.3f} & {r.rej_scalar:.3f} \\\\")
        L += [r"\hline", r"\end{tabular}"]
        (out / f"summary_table_{fam}.tex").write_text("\n".join(L) + "\n")
    print("  saved summary tables")
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results", default=str(here / "results_full"))
    ap.add_argument("--out", default=str(here / "figures"), help="output folder (the paper used the Overleaf folder EJP_Round2/figures_R2)")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import sys; sys.path.insert(0, str(here))
    df = load(Path(args.results))
    print("rows:", len(df), "families:", fams_present(df), "T:", sorted(df["T"].unique()))
    # main text, Gaussian family
    fig_cosine_sine(df, out, "gaussian")
    fig_angle(df, out, "gaussian")
    fig_marginal(df, out, "gaussian")
    fig_joint(df, out, "gaussian", "kappa_LKC_gb", "joint_lkc", r"$\hat\kappa_{LKC}$", r"Joint scatter: $\hat\kappa_{LKC}$ vs $\hat\kappa_C$ by level $u$")
    fig_joint(df, out, "gaussian", "kappa_grad", "gradient_contour", r"$\hat\kappa_{grad}$", r"Joint scatter: $\hat\kappa_{grad}$ vs $\hat\kappa_C$ by level $u$")
    fig_powers(df, out, "gaussian", kappa1=0.25)
    # family comparisons
    fig_pvalue_cdf(df, out)
    fig_calibration_N(df, out)
    pt = fig_power_kappa(df, out)
    wt = fig_window(df, out)
    # appendix: one ROC-type figure per other family
    for fam in fams_present(df):
        if fam != "gaussian":
            fig_powers(df, out, fam, kappa1=0.25)
    summary_tables(df, out)
    print(pt.to_string())
    if wt is not None:
        print(wt.round(3).to_string())


if __name__ == "__main__":
    main()
