"""Fig. 5 — A50 Spatial Outputs (2-panel): (a) 2D overhead map, (b) height bar chart.
Replaces 3D ENU scene for print readability.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.6, "legend.frameon": False,
})

BASE = Path("/sessions/charming-hopeful-galileo/mnt/uav_research/02data_transf")
OUTDIR = BASE / "output" / "natureskill"
OUTDIR.mkdir(parents=True, exist_ok=True)

def save(fig, stem):
    for ext in ["svg", "pdf"]:
        fig.savefig(str(OUTDIR / f"{stem}.{ext}"), bbox_inches="tight")
    fig.savefig(str(OUTDIR / f"{stem}.tiff"), dpi=600, bbox_inches="tight")

# ── Load data ──
df = pd.read_csv(BASE / "output" / "buildings_shadow_v5.csv", encoding="utf-8-sig")
origin_E, origin_N = 511525.7, 5026509.1
df["E"] = df["UTM_E"] - origin_E
df["N"] = df["UTM_N"] - origin_N
df["H"] = df["height_m"]

COLORS = {"industrial": "#8856A7", "residential": "#238B45", "mid_rise": "#E68613"}
type_order = ["industrial", "residential", "mid_rise"]
type_labels = {"industrial": "Industrial", "residential": "Residential", "mid_rise": "Mid-rise"}

# ── 2-panel figure ──
fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(183/25.4, 110/25.4),
                                      gridspec_kw={"width_ratios": [1.5, 1]})

# ======================
# (a) 2D Overhead Map
# ======================
# Study radius circle
circle = Circle((0, 0), 800, fill=False, color="#888888", linewidth=0.8,
                linestyle="--", alpha=0.5)
ax_map.add_patch(circle)

# Buildings as filled circles, radius proportional to footprint area, color by type, alpha by height
for t in type_order:
    sub = df[df["sem_type"] == t]
    # Size: min 18, max 80, scaled by sqrt(height)
    sizes = np.clip(np.sqrt(sub["H"]) * 18, 14, 80)
    ax_map.scatter(sub["E"], sub["N"], s=sizes, c=COLORS[t], alpha=0.82,
                   edgecolors="white", linewidths=0.4, zorder=3, label=f"{type_labels[t]} (n={len(sub)})")

# A50 road corridor — approximate centerline
x_road = np.linspace(-750, 750, 100)
y_road = -0.2 * x_road
ax_map.plot(x_road, y_road, color="#333333", linewidth=1.2, alpha=0.7, zorder=2, label="A50 corridor")
# Lane markers
road_width = 45
nx, ny = 0.2 / np.sqrt(1.04), 1.0 / np.sqrt(1.04)
for lane_offset in [10, 20]:
    ax_map.plot(x_road - lane_offset*ny, y_road + lane_offset*nx,
                color="#999999", linewidth=0.4, linestyle="--", alpha=0.5)
    ax_map.plot(x_road + lane_offset*ny, y_road - lane_offset*nx,
                color="#999999", linewidth=0.4, linestyle="--", alpha=0.5)

# UAV hover center
ax_map.scatter([0], [0], marker="*", s=100, c="#E74C3C", edgecolors="white",
               linewidths=0.5, zorder=5, label="UAV hover (120 m)")

ax_map.set_xlabel("East (m)"); ax_map.set_ylabel("North (m)")
ax_map.set_xlim(-850, 850); ax_map.set_ylim(-850, 850)
ax_map.set_aspect("equal")
ax_map.set_title("(a) 2D Overhead Map", fontsize=7, fontweight="bold", loc="left")
ax_map.legend(fontsize=5.5, loc="upper right", frameon=True, framealpha=0.85,
              edgecolor="#cccccc", markerscale=0.7, handletextpad=0.4)
# Direction arrow (North)
ax_map.annotate("N", xy=(0.02, 0.96), xycoords="axes fraction", fontsize=7,
                fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#ccc"))
ax_map.grid(True, alpha=0.15, linewidth=0.3)

# ======================
# (b) Height Bar Chart
# ======================
df_sorted = df.sort_values("H", ascending=True)
bar_colors = [COLORS[t] for t in df_sorted["sem_type"]]
y_pos = range(len(df_sorted))
ax_bar.barh(y_pos, df_sorted["H"], height=0.7, color=bar_colors, alpha=0.82,
            edgecolor="white", linewidth=0.3)

# Mean and median lines
mean_h = df["H"].mean()
median_h = df["H"].median()
ax_bar.axvline(mean_h, color="#C0392B", linewidth=1.0, linestyle="--", alpha=0.7)
ax_bar.axvline(median_h, color="#2E86C1", linewidth=1.0, linestyle=":", alpha=0.7)
ax_bar.text(mean_h+0.5, len(df)-1, f"Mean={mean_h:.1f}", fontsize=5.5, color="#C0392B", va="top")
ax_bar.text(median_h+0.5, len(df)-3, f"Med={median_h:.1f}", fontsize=5.5, color="#2E86C1", va="top")

# Color legend for semantic types
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=COLORS[t], alpha=0.82, label=type_labels[t]) for t in type_order]
ax_bar.legend(handles=legend_patches, fontsize=5.5, loc="lower right",
              frameon=True, framealpha=0.85, edgecolor="#cccccc", handlelength=1.0)

ax_bar.set_xlabel("Height (m)")
ax_bar.set_yticks([])
ax_bar.set_xlim(0, 24)
ax_bar.set_title("(b) Building Heights (n=27)", fontsize=7, fontweight="bold", loc="left")

fig.suptitle("MiTra A50 Milan — Structured Location Prior  $P(\\mathcal{R}) = (\\mathcal{B}, \\mathcal{G})$",
             fontsize=8, fontweight="bold", y=1.01)
fig.tight_layout(pad=1.0)
save(fig, "fig05_a50_2d")
plt.close()
print("Fig.5 2-panel saved.")
