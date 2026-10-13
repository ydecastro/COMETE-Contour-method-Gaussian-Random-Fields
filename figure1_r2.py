"""Figure 1 of the paper, regenerated with the exact sampler: field, level set, 8 x 8 partition of the
level set coloured by cell, and the unit normals in one cell. Same styles as the round-1 panels.
Usage: python3 figure1_r2.py [--out DIR]"""
import argparse, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comete_r2 import FieldConfig, CirculantSampler, level_set_segments

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cell", type=int, nargs=2, default=[2, 7], help="row, col of the zoomed cell (0-based)")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

cfg = FieldConfig(model="gaussian", kappa=0.9, theta0=1.0, n=1000, unit_length=5)
Z, _ = CirculantSampler(cfg).sample_pair(np.random.default_rng(a.seed))
T, sp, N = cfg.T, cfg.spacing, 8
seg = level_set_segments(Z, 0.0, sp)
mid, vec, L = seg["mid"], seg["vec"], seg["length"]
cell = T / N
ci = np.minimum((mid[:, 0] // cell).astype(int), N - 1); cj = np.minimum((mid[:, 1] // cell).astype(int), N - 1)
colors = plt.cm.tab10.colors

def bare(figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize); ax.set_xlim(0, T); ax.set_ylim(0, T); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax

# 1. the field
fig, ax = bare(); ax.imshow(Z, origin="lower", extent=[0, T, 0, T], cmap="viridis"); fig.savefig(out / "view_on_gf_2.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# 2. the level set (segments drawn as a LineCollection)
from matplotlib.collections import LineCollection
segs = np.stack([mid - vec / 2, mid + vec / 2], axis=1)
fig, ax = bare(); ax.add_collection(LineCollection(segs, colors="black", linewidths=0.6)); fig.savefig(out / "view_on_gf_contour_2.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# 3. the partition, level set coloured by cell
fig, ax = bare()
for i in range(N):
    for j in range(N):
        m = (ci == i) & (cj == j)
        ax.add_collection(LineCollection(segs[m], colors=[colors[(i + 3 * j) % 10]], linewidths=1.6))
for k in range(1, N):
    ax.axhline(k * cell, color="black", ls="--", lw=0.9); ax.axvline(k * cell, color="black", ls="--", lw=0.9)
fig.savefig(out / "view_on_gf_cells_2.png", dpi=170, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# 4. zoom on one cell with the unit normals (colour of the cell; neighbours in their colours)
r, c = a.cell
x0, x1, y0, y1 = c * cell, (c + 1) * cell, r * cell, (r + 1) * cell
pad = 0.06 * cell
fig, ax = plt.subplots(figsize=(6, 6)); ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad); ax.set_aspect("equal"); ax.axis("off")
win = (mid[:, 0] > x0 - pad) & (mid[:, 0] < x1 + pad) & (mid[:, 1] > y0 - pad) & (mid[:, 1] < y1 + pad)
idx = np.where(win)[0]
# thin the normals so that they stay readable: keep one segment every ~0.35 units of curve length
keep, acc = [], 0.0
for k in idx:
    acc += L[k]
    if acc >= 0.35:
        keep.append(k); acc = 0.0
keep = np.array(keep)
nx, ny = vec[keep, 1] / L[keep], -vec[keep, 0] / L[keep]
col = [colors[(ci[k] + 3 * cj[k]) % 10] for k in keep]
ax.quiver(mid[keep, 0], mid[keep, 1], nx, ny, color=col, angles="xy", scale_units="xy", scale=1.0 / 0.9, width=0.004, headwidth=4, headlength=5)
ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="black", ls="--", lw=2.5)
fig.savefig(out / "view_on_gf_cell_normals_2.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("saved 4 panels in", out, "| segments:", len(L), "| normals drawn in the zoom:", len(keep))
