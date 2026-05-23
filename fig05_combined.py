"""Fig. 5: MiTra A50 spatial outputs.

(a) 3D location prior with structured height priors and semantic classes.
(b) ENU topology map with road-over-building masking for OSM overlap cases.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon


SCRIPT_DIR = Path(__file__).resolve().parent
BASE = Path("E:/uav_research/02data_transf")
OUTDIR = Path("E:/uav_research/02data_transf/output/natureskill")
BUILDING_CSV = Path("E:/uav_research/02data_transf/output/buildings_shadow_v5.csv")
ORIGIN_E, ORIGIN_N = 511525.7, 5026509.1
TYPE_ORDER = ["industrial", "residential", "mid_rise"]
TYPE_LABELS = {
    "industrial": "Industrial",
    "residential": "Residential",
    "mid_rise": "Mid-rise",
}
TYPE_COLORS = {
    "industrial": "#7B3294",
    "residential": "#008837",
    "mid_rise": "#E08214",
}

UAV_CYAN = "#00A6D6"
ROAD_FILL = "#4D5663"
ROAD_EDGE = "#242A31"
LANE_YELLOW = "#F6C85F"
RADIUS_M = 800
HOVER_H = 120
ROAD_WIDTH = 45

X_ROAD = np.linspace(-750, 750, 160)
Y_ROAD = -0.2 * X_ROAD
N_DIR = np.array([0.2, 1.0]) / np.sqrt(1.04)


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.1,
    "axes.linewidth": 0.55,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})


def load_buildings() -> pd.DataFrame:
    df = pd.read_csv(BUILDING_CSV, encoding="utf-8-sig")
    df["E"] = df["UTM_E"] - ORIGIN_E
    df["N"] = df["UTM_N"] - ORIGIN_N
    df["H"] = df["height_m"]
    return df


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("svg", "pdf"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def road_polygon() -> np.ndarray:
    nx, ny = N_DIR
    x_upper = X_ROAD + ROAD_WIDTH / 2 * ny
    y_upper = Y_ROAD + ROAD_WIDTH / 2 * nx
    x_lower = X_ROAD - ROAD_WIDTH / 2 * ny
    y_lower = Y_ROAD - ROAD_WIDTH / 2 * nx
    return np.column_stack([
        np.hstack([x_upper, x_lower[::-1]]),
        np.hstack([y_upper, y_lower[::-1]]),
    ])


def draw_3d_panel(ax, df: pd.DataFrame) -> None:
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#D9D9D9")
    ax.grid(True, color="#DADADA", lw=0.35, alpha=0.5)

    nx, ny = N_DIR
    y_surf = np.vstack([Y_ROAD - ROAD_WIDTH / 2 * nx, Y_ROAD + ROAD_WIDTH / 2 * nx])
    x_surf = np.vstack([X_ROAD - ROAD_WIDTH / 2 * ny, X_ROAD + ROAD_WIDTH / 2 * ny])
    ax.plot_surface(x_surf, y_surf, np.zeros_like(x_surf), color=ROAD_FILL,
                    alpha=0.30, shade=False, zorder=1)
    ax.plot(X_ROAD, Y_ROAD, zs=0.05, color=LANE_YELLOW, lw=0.9, ls="--", alpha=0.8)
    for offset in (10, 20):
        ax.plot(X_ROAD - offset * ny, Y_ROAD + offset * nx, zs=0.04,
                color="white", lw=0.45, ls="--", alpha=0.38)
        ax.plot(X_ROAD + offset * ny, Y_ROAD - offset * nx, zs=0.04,
                color="white", lw=0.45, ls="--", alpha=0.38)

    for sem_type in TYPE_ORDER:
        sub = df[df["sem_type"] == sem_type]
        color = TYPE_COLORS[sem_type]
        for _, row in sub.iterrows():
            ax.plot([row["E"], row["E"]], [row["N"], row["N"]], [0, row["H"]],
                    color=color, linewidth=0.9, alpha=0.62)
        ax.scatter(sub["E"], sub["N"], 0, color=color, s=12, alpha=0.35,
                   edgecolors="none", depthshade=False)
        ax.scatter(sub["E"], sub["N"], sub["H"], color=color, s=26, marker="^",
                   alpha=0.92, edgecolors="white", linewidths=0.25, depthshade=False)

    ax.plot([0, 0], [0, 0], [0, HOVER_H], color=UAV_CYAN, lw=1.35, alpha=0.86)
    ax.scatter([0], [0], [HOVER_H], marker="*", s=105, color=UAV_CYAN,
               edgecolors="white", linewidths=0.45, depthshade=False, zorder=10)

    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(RADIUS_M * np.cos(theta), RADIUS_M * np.sin(theta), zs=0,
            color="#8A8A8A", lw=0.65, ls="--", alpha=0.42)

    ax.set_xlim(-RADIUS_M, RADIUS_M)
    ax.set_ylim(-RADIUS_M, RADIUS_M)
    ax.set_zlim(0, 140)
    ax.set_xlabel("East (m)", labelpad=-1)
    ax.set_ylabel("N (m)", labelpad=-1)
    ax.set_zlabel("")
    ax.tick_params(axis="both", which="major", pad=0)
    ax.tick_params(axis="z", which="major", pad=1)
    ax.set_box_aspect((1.15, 1.0, 0.38))
    ax.set_proj_type("ortho")
    ax.view_init(elev=28, azim=-55)
    try:
        ax.dist = 8.8
    except AttributeError:
        pass
    ax.set_title("(a) 3D location prior", fontweight="bold", loc="left", pad=-2)


def draw_2d_panel(ax, df: pd.DataFrame) -> None:
    nx, ny = N_DIR
    ax.add_patch(Circle((0, 0), RADIUS_M, fill=False, color="#8A8A8A",
                        lw=0.75, ls="--", alpha=0.55, zorder=0))

    for sem_type in TYPE_ORDER:
        sub = df[df["sem_type"] == sem_type]
        sizes = np.clip(np.sqrt(sub["H"]) * 23, 22, 92)
        ax.scatter(sub["E"], sub["N"], s=sizes, c=TYPE_COLORS[sem_type],
                   alpha=0.82, edgecolors="white", linewidths=0.55, zorder=3)

    # Keep the road above buildings so OSM objects that overlap the carriageway
    # are visually masked by the road surface instead of looking like map errors.
    ax.add_patch(Polygon(road_polygon(), closed=True, facecolor=ROAD_FILL,
                         alpha=0.35, edgecolor=ROAD_EDGE, lw=0.55, zorder=5))
    ax.plot(X_ROAD, Y_ROAD, color=LANE_YELLOW, lw=0.95, ls="--",
            alpha=0.82, zorder=6)
    for offset in (10, 20):
        ax.plot(X_ROAD - offset * ny, Y_ROAD + offset * nx, color="white",
                lw=0.45, ls="--", alpha=0.58, zorder=6)
        ax.plot(X_ROAD + offset * ny, Y_ROAD - offset * nx, color="white",
                lw=0.45, ls="--", alpha=0.58, zorder=6)

    ax.scatter([0], [0], marker="*", s=105, c=UAV_CYAN, edgecolors="white",
               linewidths=0.55, zorder=10)
    ax.annotate("UAV", xy=(0, 0), xytext=(42, -48), textcoords="data",
                fontsize=6.2, color="#202020",
                arrowprops=dict(arrowstyle="-", lw=0.45, color="#555555"))
    ax.set_xlim(-850, 850)
    ax.set_ylim(-850, 850)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, color="#DADADA", lw=0.35, alpha=0.62, zorder=-1)
    ax.set_title("(b) ENU topology map", fontweight="bold", loc="left")


def main() -> None:
    df = load_buildings()
    fig = plt.figure(figsize=(183 / 25.4, 120 / 25.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0], wspace=0.34)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d", facecolor="white")
    ax2d = fig.add_subplot(gs[0, 1])

    draw_3d_panel(ax3d, df)
    draw_2d_panel(ax2d, df)

    semantic_handles = [
        Patch(facecolor=TYPE_COLORS[t], edgecolor="none",
              label=f"{TYPE_LABELS[t]} (n={(df['sem_type'] == t).sum()})")
        for t in TYPE_ORDER
    ]
    layer_handles = [
        Patch(facecolor=ROAD_FILL, edgecolor=ROAD_EDGE, alpha=0.35,
              label="A50 road mask"),
        Line2D([0], [0], color=LANE_YELLOW, lw=0.95, ls="--",
               label="Lane center/markers"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=UAV_CYAN,
               markeredgecolor="white", markersize=8, label="UAV hover"),
    ]
    fig.legend(handles=[*semantic_handles, *layer_handles], loc="lower center",
               ncol=3, bbox_to_anchor=(0.52, -0.006), frameon=False,
               handlelength=1.35, columnspacing=1.35)

    fig.tight_layout(pad=0.8)

    save_figure(fig, "fig05_a50_combined")
    plt.close(fig)
    print(f"Saved Fig. 5 outputs to {OUTDIR} ({len(df)} buildings).")


if __name__ == "__main__":
    main()
