"""Fig. 3: Theoretical Height Uncertainty Heatmap — Nature-style regeneration.
Core conclusion: Pixel quantization dominates height uncertainty; 10m GSD σ≈4.3m vs 0.93m GSD σ≈0.40m.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === Nature-figure rcParams ===
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "legend.fontsize": 6.5,
})

OUTDIR = Path("E:/uav_research/02data_transf/output/natureskill")
OUTDIR.mkdir(parents=True, exist_ok=True)

def save_pub_py(fig, stem, dpi=600):
    fig.savefig(OUTDIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    print(f"Saved {stem}.svg/pdf/tiff")

# === Compute sigma_h grid ===
sun_elev_deg = np.linspace(20, 70, 80)
gsd_vals_m   = np.linspace(0.5, 10, 80)
Sun, GSD = np.meshgrid(sun_elev_deg, gsd_vals_m)
sigma_h = np.tan(np.radians(Sun)) * GSD / np.sqrt(12)

# === Nature-style figure ===
fig, ax = plt.subplots(figsize=(183/25.4, 130/25.4))  # 183x130mm

# Filled contours — use a restrained sequential palette
levels_fill = np.arange(0, 12.1, 0.5)
cf = ax.contourf(Sun, GSD, sigma_h, levels=levels_fill, cmap="YlOrBr", alpha=0.85)

# Labeled contour lines at key thresholds
levels_line = [0.5, 1, 2, 3, 4.3, 6, 8, 10]
cs = ax.contour(Sun, GSD, sigma_h, levels=levels_line,
                colors="black", linewidths=0.6, linestyles="solid")
ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.1f m")

# MiTra A50 operating points
ax.scatter([56.19], [10.0], c="#2166AC", s=100, marker="o", edgecolors="white",
           linewidths=0.8, zorder=5, label="MiTra A50: Sentinel-2, 10 m GSD\n($\\sigma_{\\hat{h}}$ = 4.3 m)")
ax.scatter([56.19], [0.9346], c="#1B7837", s=100, marker="s", edgecolors="white",
           linewidths=0.8, zorder=5, label="MiTra A50: Esri, 0.93 m GSD\n($\\sigma_{\\hat{h}}$ = 0.40 m)")

# Colorbar
cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label(r"$\sigma_{\hat{h}}$  (m)", fontsize=7)
cbar.ax.tick_params(labelsize=6.5)

# Labels
ax.set_xlabel("Solar Elevation  $\\beta_{sun}$  (°)", fontsize=7.5)
ax.set_ylabel("Ground Sampling Distance  GSD  (m/px)", fontsize=7.5)
ax.set_title("Theoretical Height Uncertainty  $\\sigma_{\\hat{h}} = \\tan(\\beta_{sun}) \\cdot \\mathrm{GSD} / \\sqrt{12}$",
             fontsize=7.5, pad=8)

# Legend — subtle box
legend = ax.legend(loc="upper left", fontsize=6.5, handlelength=1.2, handletextpad=0.5,
                   frameon=True, framealpha=0.85, edgecolor="#cccccc", borderpad=0.4)
ax.grid(True, alpha=0.2, linewidth=0.3)

fig.tight_layout(pad=0.8)
save_pub_py(fig, "fig03_heatmap")
plt.close()
print("Fig. 3 done.")
