"""Fig. 4: Building Height Statistics — Nature-style 4-panel (a-d)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6,
    "legend.frameon": False,
})

BASE = "/sessions/charming-hopeful-galileo/mnt/uav_research/02data_transf"
OUTDIR = Path(BASE) / "output" / "natureskill"
OUTDIR.mkdir(parents=True, exist_ok=True)

def save_pub_py(fig, stem, dpi=600):
    for ext in ["svg", "pdf"]:
        fig.savefig(str(OUTDIR / f"{stem}.{ext}"), bbox_inches="tight")
    fig.savefig(str(OUTDIR / f"{stem}.tiff"), dpi=dpi, bbox_inches="tight")
    print(f"Saved {stem}.svg/pdf/tiff")

csv_path = Path(BASE) / "output" / "buildings_shadow_v5.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")
print(f"Loaded {len(df)} buildings: mean height {df.height_m.mean():.1f}m")

type_colors = {"industrial": "#8856A7", "residential": "#238B45", "mid-rise": "#E68613"}
type_order = ["industrial", "residential", "mid-rise"]

fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 170/25.4))
(ax_a, ax_b), (ax_c, ax_d) = axes

# (a) Height histogram
bins = np.arange(2.5, 27.6, 2.5)
ax_a.hist(df.height_m, bins=bins, color="#4A90A4", edgecolor="white", linewidth=0.5, alpha=0.85)
ax_a.axvline(df.height_m.mean(), color="#C0392B", linewidth=1.0, linestyle="--")
ax_a.axvline(df.height_m.median(), color="#2E86C1", linewidth=1.0, linestyle=":")
ax_a.text(0.98, 0.92, f"Mean={df.height_m.mean():.1f}m\nMedian={df.height_m.median():.1f}m",
          transform=ax_a.transAxes, ha="right", va="top", fontsize=5.5,
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc"))
ax_a.set_xlabel("Height (m)"); ax_a.set_ylabel("Count")
ax_a.set_title("(a) Height Histogram", fontsize=7, fontweight="bold", loc="left")

# (b) Box plots by type
bp_data = [df[df.sem_type == t].height_m.values for t in type_order]
bp = ax_b.boxplot(bp_data, patch_artist=True, widths=0.5,
                  medianprops={"color": "black", "linewidth": 1.0},
                  whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
for patch, t in zip(bp["boxes"], type_order):
    patch.set_facecolor(type_colors[t]); patch.set_alpha(0.7)
ax_b.set_xticklabels(["Industrial\n(n=10)", "Residential\n(n=14)", "Mid-rise\n(n=3)"], fontsize=6.5)
ax_b.set_ylabel("Height (m)")
ax_b.set_title("(b) Height by Semantic Type", fontsize=7, fontweight="bold", loc="left")

# (c) Distribution bars
counts = [len(df[df.sem_type == t]) for t in type_order]
pcts = [c/len(df)*100 for c in counts]
bars = ax_c.barh(type_order, counts, color=[type_colors[t] for t in type_order], height=0.55, alpha=0.8)
for bar, c, p in zip(bars, counts, pcts):
    ax_c.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
              f"{c} ({p:.0f}%)", va="center", fontsize=6.5)
ax_c.set_xlabel("Count"); ax_c.set_xlim(0, 17)
ax_c.set_title("(c) Semantic Distribution", fontsize=7, fontweight="bold", loc="left")

# (d) Height vs confidence
for t in type_order:
    sub = df[df.sem_type == t]
    ax_d.scatter(sub.height_m, sub.height_conf, c=type_colors[t], s=28, alpha=0.75,
                 edgecolors="white", linewidths=0.3, label=t.capitalize(), zorder=3)
ax_d.set_xlabel("Height (m)"); ax_d.set_ylabel("Confidence")
ax_d.legend(fontsize=5.5, loc="lower right", frameon=True, framealpha=0.8, edgecolor="#ccc", markerscale=0.8)
ax_d.set_title("(d) Height vs. Confidence", fontsize=7, fontweight="bold", loc="left")

fig.suptitle("MiTra A50 T1-D1 Building Height Statistics ($n$=27, Sentinel-2A Shadow Geometry)",
             fontsize=7.5, fontweight="bold")
fig.tight_layout(pad=1.2)
save_pub_py(fig, "fig04_stats")
plt.close()
print("Fig. 4 done.")
