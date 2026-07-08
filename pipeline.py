from __future__ import annotations

import argparse
import json
import logging
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("location_prior")


class HeightMethod(str, Enum):
    SHADOW_GEO_FIXED = "SHADOW_GEO_FIXED"
    STRUCTURED_PRIOR = "STRUCTURED_PRIOR"


@dataclass
class TileMetadata:
    epsg_code: str
    epsg_name: str
    ulx: float
    uly: float
    xdim: float
    ydim: float
    rows: int
    cols: int
    sun_zenith_deg: float
    sun_azimuth_deg: float
    acquisition_time: str
    cloud_cover_fraction: Optional[float] = None

    @property
    def sun_elevation_deg(self) -> float:
        return 90.0 - self.sun_zenith_deg

    @property
    def resolution_m(self) -> float:
        return abs(self.xdim)

    def pixel_from_utm(self, east_m: float, north_m: float) -> tuple[int, int]:
        col = int(round((east_m - self.ulx) / self.xdim))
        row = int(round((self.uly - north_m) / abs(self.ydim)))
        return row, col

    def utm_from_pixel(self, row: float, col: float) -> tuple[float, float]:
        east_m = self.ulx + col * self.xdim
        north_m = self.uly - row * abs(self.ydim)
        return east_m, north_m


@dataclass
class DatasetPaths:
    workspace_root: Path
    product_xml: Path
    tile_xml: Path
    tci_jp2: Path
    uav_csv: Path
    vehicle_csv: Path
    output_dir: Path
    paper_figure_dir: Path
    buildings_csv: Path
    shadow_overlay_png: Path
    prior_json: Path
    summary_txt: Path
    prior_3d_png: Path
    legacy_buildings_csv: Path
    legacy_3d_png: Path
    legacy_viz_png: Path
    legacy_prior_json: Path
    paper_source_docx: Path
    paper_output_docx: Path
    paper_markdown: Path


@dataclass
class RoiBounds:
    row_start: int
    row_end: int
    col_start: int
    col_end: int


@dataclass
class BuildingGeometry:
    centroid_utm_e: float
    centroid_utm_n: float
    centroid_enu_e: float
    centroid_enu_n: float
    length_m: float
    width_m: float
    height_m: float


@dataclass
class BuildingPrior:
    building_id: str
    semantic_type: str
    geometry: BuildingGeometry
    height_conf: float
    height_method: str
    sigma_h_m: float = 0.0
    shadow_len_m: float = 0.0
    long_axis_px: float = 0.0
    long_clamped_px: float = 0.0
    aspect_ratio: float = 0.0
    compactness: float = 0.0
    area_px: int = 0
    nearest_lane_id: Optional[int] = None
    road_dist_m: float = 0.0
    neighbors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoadPrior:
    road_id: str
    lane_id: int
    road_type: str
    polyline_utm: list[list[float]]
    polyline_enu: list[list[float]]
    vehicle_count: int
    mean_speed_kmh: float
    occluding_buildings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LocationPriorMap:
    center_utm_e: float
    center_utm_n: float
    center_altitude_m: float
    radius_m: float
    sun_elevation_deg: float
    sun_azimuth_deg: float
    acquisition_time: str
    uav_frame_count: int
    vehicle_row_count: int
    buildings: list[BuildingPrior] = field(default_factory=list)
    roads: list[RoadPrior] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> dict:
        heights = [b.geometry.height_m for b in self.buildings]
        methods: dict[str, int] = {}
        semantics: dict[str, int] = {}
        for building in self.buildings:
            methods[building.height_method] = methods.get(building.height_method, 0) + 1
            semantics[building.semantic_type] = semantics.get(building.semantic_type, 0) + 1
        return {
            "total_buildings": len(self.buildings),
            "total_roads": len(self.roads),
            "avg_height_m": round(float(np.mean(heights)), 2) if heights else 0.0,
            "median_height_m": round(float(np.median(heights)), 2) if heights else 0.0,
            "min_height_m": round(float(np.min(heights)), 2) if heights else 0.0,
            "max_height_m": round(float(np.max(heights)), 2) if heights else 0.0,
            "std_height_m": round(float(np.std(heights)), 2) if heights else 0.0,
            "height_methods": methods,
            "semantic_distribution": semantics,
        }


@dataclass
class ShadowInferenceResult:
    buildings_df: pd.DataFrame
    source_name: str
    source_reason: str
    roi_rgb: np.ndarray
    shadow_mask: np.ndarray
    roi_bounds: RoiBounds
    threshold_value: float
    shadow_pixel_ratio: float


@dataclass
class PipelineArtifacts:
    paths: DatasetPaths
    tile_meta: TileMetadata
    uav_df: pd.DataFrame
    vehicle_df: pd.DataFrame
    buildings_df: pd.DataFrame
    shadow_result: ShadowInferenceResult
    prior_map: LocationPriorMap


POI_STYLE = {
    "industrial": {"color": "#8E44AD", "label": "Industrial", "marker_g": "s", "marker_t": "^"},
    "residential": {"color": "#27AE60", "label": "Residential", "marker_g": "s", "marker_t": "^"},
    "commercial": {"color": "#F39C12", "label": "Commercial", "marker_g": "s", "marker_t": "^"},
    "transport": {"color": "#3498DB", "label": "Transport", "marker_g": "D", "marker_t": "^"},
    "vegetation": {"color": "#2ECC71", "label": "Vegetation", "marker_g": "o", "marker_t": "^"},
    "high_rise": {"color": "#E74C3C", "label": "High-Rise", "marker_g": "s", "marker_t": "^"},
    "mid_rise": {"color": "#E67E22", "label": "Mid-Rise", "marker_g": "s", "marker_t": "^"},
    "mixed": {"color": "#95A5A6", "label": "Mixed Use", "marker_g": "s", "marker_t": "^"},
    "unknown": {"color": "#BDC3C7", "label": "Unknown", "marker_g": "s", "marker_t": "^"},
}

DEFAULT_STYLE = {"color": "#BDC3C7", "label": "Unknown", "marker_g": "s", "marker_t": "^"}

UAV_PHASE_STYLE = {
    "takeoff": {"color": "#00FF88", "label": "Telemetry takeoff", "lw": 2.5},
    "hover": {"color": "#00BFFF", "label": "Telemetry hover", "lw": 3.0},
    "landing": {"color": "#FF8C00", "label": "Telemetry landing", "lw": 2.5},
}

PIXEL_M = 10.0  # Default: Sentinel-2 GSD. Overridden per-image via _read_tiff_geotag().

# Esri World Imagery geotag constants (PIL tag IDs)
_TIFF_TAG_MODEL_PIXEL_SCALE = 33550
_TIFF_TAG_MODEL_TIEPOINT = 33922


def _read_tiff_geotag(tiff_path: Path) -> tuple[float, float, float]:
    """Read GSD and upper-left UTM from a GeoTIFF using PIL tag interface.

    Returns:
        (gsd_m, ulx_e, uly_n) — resolution (m/px), origin easting, origin northing.
    """
    img = Image.open(str(tiff_path))
    tags = img.tag_v2

    # Resolution (pixel scale)
    scale = tags.get(_TIFF_TAG_MODEL_PIXEL_SCALE)
    if scale is None:
        raise ValueError(f"No ModelPixelScaleTag in {tiff_path}")
    gsd_m = float(scale[0])  # first component = X pixel size

    # Origin (tie point: pixel (0,0) → world coordinate)
    tiepoint = tags.get(_TIFF_TAG_MODEL_TIEPOINT)
    if tiepoint is None:
        raise ValueError(f"No ModelTiepointTag in {tiff_path}")
    # tiepoint is (I, J, K, X, Y, Z) — pixel (I,J,K) maps to (X,Y,Z)
    ulx_e = float(tiepoint[3])
    uly_n = float(tiepoint[4])

    logger.info("[GeoTiff] %s: %dx%d px, GSD=%.4f m/px, origin=(%.1f E, %.1f N)",
                tiff_path.name, img.width, img.height, gsd_m, ulx_e, uly_n)
    return gsd_m, ulx_e, uly_n


H_MIN = 3.0
H_MAX = 35.0
LONG_AXIS_MAX_PX = 3.5  # for fallback only; extract_buildings_from_shadow computes dynamically
# AREA_MIN_PX / AREA_MAX_PX are no longer used — dynamically computed from physical units
ASPECT_MAX = 4.5
COMPACT_MIN = 0.05
SIGMA_CLIP = 2.5


def _find_first(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find '{pattern}' under {root}")
    return matches[0]


def resolve_dataset_paths(script_path: Path) -> DatasetPaths:
    """Resolve all dataset paths using hardcoded E:\\uav_research root (v6).

    Input data locations (read-only):
      - satellite XML/JP2:  E:\\uav_research\\02data_transf\\input\\satellite\\
      - vehicle CSV:         E:\\uav_research\\02data_transf\\input\\vehicle\\
      - UAV CSV:             E:\\uav_research\\02data_transf\\input\\uav\\

    Output (all written to E:\\uav_research\\02data_transf\\output\\ with _v4 suffix):
      - buildings_shadow_v5.csv
      - shadow_detection_visual_v5.png
      - location_prior_map_v5.json
      - location_prior_3d_v5.png
      - summary_statistics_v5.txt
    """
    data_root = Path("E:/uav_research/02data_transf")
    input_dir = data_root / "input"
    output_dir = data_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input files — explicit hardcoded paths (no recursive search)
    product_xml = input_dir / "satellite" / "MTD_MSIL2A.xml"
    tile_xml    = input_dir / "satellite" / "MTD_TL.xml"
    tci_jp2     = input_dir / "qgis_milan_a50_t1d1_850m" / "qgis_milan_a50_t1d1_850m.tif"
    uav_csv     = input_dir / "uav" / "T1_D1_uav.csv"
    vehicle_csv = input_dir / "vehicle" / "T1_D1.csv"

    # Validate existence
    for p, label in [
        (product_xml, "MTD_MSIL2A.xml"),
        (tile_xml, "MTD_TL.xml"),
        (tci_jp2, "qgis_milan_a50_t1d1_850m.tif"),
        (uav_csv, "T1_D1_uav.csv"),
        (vehicle_csv, "T1_D1.csv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"[v6] {label} not found at {p}")

    return DatasetPaths(
        workspace_root=data_root.parent,
        product_xml=product_xml,
        tile_xml=tile_xml,
        tci_jp2=tci_jp2,
        uav_csv=uav_csv,
        vehicle_csv=vehicle_csv,
        output_dir=output_dir,
        paper_figure_dir=output_dir,
        buildings_csv=output_dir / "buildings_shadow_v5.csv",
        shadow_overlay_png=output_dir / "shadow_detection_visual_v5.png",
        prior_json=output_dir / "location_prior_map_v5.json",
        summary_txt=output_dir / "summary_statistics_v5.txt",
        prior_3d_png=output_dir / "location_prior_3d_v5.png",
        legacy_buildings_csv=output_dir / "buildings_shadow_v5.csv",
        legacy_3d_png=output_dir / "location_prior_3d_v5.png",
        legacy_viz_png=output_dir / "viz_3d_uav_v5.png",
        legacy_prior_json=output_dir / "output_prior_map_v5.json",
        paper_source_docx=output_dir / "location_prior_paper.docx",
        paper_output_docx=output_dir / "location_prior_paper_rewritten.docx",
        paper_markdown=output_dir / "location_prior_paper_rewritten.md",
    )


def ensure_output_dirs(paths: DatasetPaths) -> None:
    """v4: single output directory — all files write to E:\\uav_research\\02data_transf\\output\\"""
    paths.output_dir.mkdir(parents=True, exist_ok=True)


def _strip_namespace(tag: str) -> str:
    return tag.split("}")[-1]


def _iter_tag(root: ET.Element, wanted: str):
    for elem in root.iter():
        if _strip_namespace(elem.tag) == wanted:
            yield elem


def parse_tile_metadata(paths: DatasetPaths) -> TileMetadata:
    tile_root = ET.parse(paths.tile_xml).getroot()
    product_root = ET.parse(paths.product_xml).getroot()

    geoposition = None
    size_node = None
    for elem in _iter_tag(tile_root, "Geoposition"):
        if elem.attrib.get("resolution") == "10":
            geoposition = elem
            break
    for elem in _iter_tag(tile_root, "Size"):
        if elem.attrib.get("resolution") == "10":
            size_node = elem
            break
    mean_sun = next(_iter_tag(tile_root, "Mean_Sun_Angle"))

    def read_text(parent: ET.Element, child_name: str) -> str:
        for child in parent:
            if _strip_namespace(child.tag) == child_name:
                return (child.text or "").strip()
        raise KeyError(child_name)

    acquisition_time = ""
    for tag_name in ("DATATAKE_SENSING_START", "PRODUCT_START_TIME", "SENSING_TIME"):
        elem = next(_iter_tag(product_root, tag_name), None)
        if elem is not None and elem.text:
            acquisition_time = elem.text.strip()
            break

    epsg_code = next(_iter_tag(tile_root, "HORIZONTAL_CS_CODE")).text.strip()
    epsg_name = next(_iter_tag(tile_root, "HORIZONTAL_CS_NAME")).text.strip()

    # Attempt to parse cloud cover from Sentinel-2 L2A product metadata.
    # Field: Quality_Indicators_Info / Cloud_Coverage_Assessment (percentage).
    # If absent, cloud_cover_fraction remains None and the pre-extraction gate
    # skips the cloud check (see check_metadata_gate).
    cloud_cover_fraction = None
    try:
        cloud_tag = product_root.find('.//{*}Cloud_Coverage_Assessment')
        if cloud_tag is not None and cloud_tag.text:
            cloud_cover_fraction = float(cloud_tag.text.strip()) / 100.0
    except (AttributeError, ValueError, TypeError):
        pass

    return TileMetadata(
        epsg_code=epsg_code,
        epsg_name=epsg_name,
        ulx=float(read_text(geoposition, "ULX")),
        uly=float(read_text(geoposition, "ULY")),
        xdim=float(read_text(geoposition, "XDIM")),
        ydim=float(read_text(geoposition, "YDIM")),
        rows=int(read_text(size_node, "NROWS")),
        cols=int(read_text(size_node, "NCOLS")),
        sun_zenith_deg=float(read_text(mean_sun, "ZENITH_ANGLE")),
        sun_azimuth_deg=float(read_text(mean_sun, "AZIMUTH_ANGLE")),
        acquisition_time=acquisition_time,
        cloud_cover_fraction=cloud_cover_fraction,
    )


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def compute_hover_origin(uav_df: pd.DataFrame) -> tuple[float, float]:
    hover = uav_df[uav_df["Phase"].str.lower() == "hover"] if "Phase" in uav_df.columns else uav_df
    if hover.empty:
        hover = uav_df
    return float(hover["UTM_E[m]"].mean()), float(hover["UTM_N[m]"].mean())


def to_enu(east_arr, north_arr, origin_e: float, origin_n: float):
    return np.asarray(east_arr, dtype=float) - origin_e, np.asarray(north_arr, dtype=float) - origin_n


def compute_roi_bounds(tile_meta: TileMetadata, east_m: float, north_m: float, buffer_m: float) -> RoiBounds:
    row_c, col_c = tile_meta.pixel_from_utm(east_m, north_m)
    buffer_px = int(buffer_m / tile_meta.resolution_m)
    row_start = max(0, row_c - buffer_px)
    row_end = min(tile_meta.rows, row_c + buffer_px)
    col_start = max(0, col_c - buffer_px)
    col_end = min(tile_meta.cols, col_c + buffer_px)
    return RoiBounds(row_start, row_end, col_start, col_end)


def load_roi_rgb(paths: DatasetPaths, roi_bounds: RoiBounds) -> np.ndarray:
    image = Image.open(paths.tci_jp2).convert("RGB")
    roi = image.crop((roi_bounds.col_start, roi_bounds.row_start, roi_bounds.col_end, roi_bounds.row_end))
    return np.array(roi, dtype=np.uint8)


def detect_shadows(roi_rgb: np.ndarray, percentile: float = 20.0) -> tuple[np.ndarray, dict]:
    """Detect shadow pixels using luminance + dual-peak valley thresholding.

    Strategy (derived from 3test.ipynb exploration):
    - The ROI typically contains three L* clusters: dark shadows (~50-80),
      mid-brightness ground/roads (~130-160), and very bright rooftops (~240-255).
    - A single percentile threshold conflates shadows with ground-level pixels,
      while the global bi-modal valley always falls between ground and rooftops,
      not between shadow and ground.
    - Solution: restrict the histogram to L* < 120, find the left peak, then set
      the threshold just right of that peak (left peak value + 12), clamped to
      [mu - 1.3*sigma,  P25] of non-vegetation pixels.
    - Vegetation is excluded via HSV hue/saturation filter before computing stats.
    """
    roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0].astype(float)
    H, S = hsv[:, :, 0], hsv[:, :, 1]

    # Exclude vegetation (green hue, reasonable saturation)
    veg_mask = (H >= 35) & (H <= 85) & (S > 40)
    L_nonveg = L[~veg_mask]

    # Fallback adaptive threshold (mu - 1.3*sigma of non-veg pixels)
    thresh_ad = max(20.0, float(L_nonveg.mean() - 1.3 * L_nonveg.std()))
    thresh_p25 = float(np.percentile(L_nonveg, 25))

    # Find the left peak in the dark range (L* < 120) and step just past it
    hist_lo, edges_lo = np.histogram(L_nonveg[L_nonveg < 120], bins=48, range=(0, 120))
    hist_s = gaussian_filter1d(hist_lo.astype(float), sigma=2)
    centers = (edges_lo[:-1] + edges_lo[1:]) / 2
    peaks, peak_props = find_peaks(hist_s, height=hist_s.max() * 0.1, distance=6)

    n_dark_pixels = int((L_nonveg < 120).sum())
    peak_details = []
    for i, p_idx in enumerate(peaks):
        peak_details.append(f"L*={centers[p_idx]:.1f}(h={peak_props['peak_heights'][i]:.0f})")
    logger.info("[Threshold] Dark pixels (L*<120): %d / %d non-veg (%.1f%%)",
                n_dark_pixels, len(L_nonveg), 100.0 * n_dark_pixels / max(len(L_nonveg), 1))
    logger.info("[Threshold] Peaks found in dark range: %d → %s",
                len(peaks), ", ".join(peak_details) if peak_details else "NONE")

    if len(peaks) >= 1:
        # Use the left-most peak value + small offset to capture shadow tail
        thresh_active = float(centers[peaks[0]]) + 12.0
        logger.info("[Threshold] Peak-valley: left_peak=%.1f + 12 → %.1f",
                    float(centers[peaks[0]]), thresh_active)
    else:
        thresh_active = thresh_ad
        logger.info("[Threshold] Peak-valley: no peak found, using adaptive=%.1f", thresh_ad)

    # Clamp to sensible range: must not exceed P25 or drop below adaptive lower bound
    thresh_before = thresh_active
    thresh_active = float(np.clip(thresh_active, thresh_ad, thresh_p25))
    if thresh_before != thresh_active:
        logger.info("[Threshold] Clamped: %.1f → %.1f (bounds: adaptive=%.1f, p25=%.1f)",
                    thresh_before, thresh_active, thresh_ad, thresh_p25)
    else:
        logger.info("[Threshold] Final threshold: %.1f (within bounds [%.1f, %.1f])",
                    thresh_active, thresh_ad, thresh_p25)

    # Build shadow mask
    k3 = np.ones((3, 3), np.uint8)
    shadow_raw = ((L < thresh_active) & ~veg_mask).astype(np.uint8) * 255
    shadow_mask = cv2.morphologyEx(shadow_raw, cv2.MORPH_OPEN, k3, iterations=1)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, k3, iterations=2)

    shadow_pixel_ratio = float((shadow_mask > 0).mean())
    stats = {
        "light_mean": float(L.mean()),
        "light_std": float(L.std()),
        "threshold": thresh_active,
        "thresh_adaptive": thresh_ad,
        "thresh_p25": thresh_p25,
        "shadow_pixel_ratio": shadow_pixel_ratio,
    }
    return shadow_mask, stats


def classify_by_height(height_m: float) -> str:
    if height_m > 25:
        return "high_rise"
    if height_m > 15:
        return "mid_rise"
    if height_m > 8:
        return "residential"
    return "industrial"


def extract_buildings_from_shadow(
    shadow_mask: np.ndarray,
    tile_meta: TileMetadata,
    roi_bounds: RoiBounds,
) -> pd.DataFrame:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(shadow_mask, connectivity=8)
    tan_elevation = math.tan(math.radians(tile_meta.sun_elevation_deg))
    confidence_base = min(0.45 + tile_meta.sun_elevation_deg * 0.008, 0.78)
    rows = []

    gsd_m = tile_meta.resolution_m  # 0.9346 — for pixel→meter conversion only

    # ── Route B: Esri 0.93m mask + Sentinel-2 10m height model ──────────────
    # The Esri image gives cleaner shadow masks (higher resolution, better
    # connected-component shapes). Pixel measurements are converted to meters
    # using the actual GSD, then Sentinel-2 model parameters (35 m max shadow,
    # σ_ĥ = tan(β) × 10/√12 ≈ 4.3 m) are applied.
    SENTINEL_GSD = 10.0  # m/px, for uncertainty formula (paper Eq.4)
    MAX_SHADOW_M = 35.0   # m, physical

    # Area filter: physical m² → pixels at this GSD
    # Original: 1-200 px² at 10m GSD = 100-20,000 m² physical
    area_min_px = int(math.ceil(100.0 / (gsd_m ** 2)))    # 0.93m → 115 px²
    area_max_px = int(math.floor(20000.0 / (gsd_m ** 2))) # 0.93m → 22891 px²
    aspect_max = 4.5
    compact_min = 0.05
    long_axis_max_px = MAX_SHADOW_M / gsd_m  # 35m / 0.9346 = 37.5 px

    # Filter counters for diagnosis
    n_total = num_labels - 1  # exclude background
    n_area_pass = 0
    n_contour_pass = 0
    n_aspect_pass = 0
    n_compact_pass = 0

    for idx in range(1, num_labels):
        area_px = int(stats[idx, cv2.CC_STAT_AREA])
        if area_px < area_min_px or area_px > area_max_px:
            continue
        n_area_pass += 1

        component_mask = (labels == idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        n_contour_pass += 1
        contour = max(contours, key=cv2.contourArea)

        if len(contour) >= 5:
            _, (rect_w, rect_h), _ = cv2.minAreaRect(contour.astype(np.float32))
            long_axis_px = float(max(rect_w, rect_h))
            short_axis_px = float(min(rect_w, rect_h)) + 1e-3
        else:
            _, _, box_w, box_h = cv2.boundingRect(contour)
            long_axis_px = float(max(box_w, box_h))
            short_axis_px = float(min(box_w, box_h)) + 1e-3

        aspect_ratio = long_axis_px / short_axis_px
        if aspect_ratio > aspect_max:
            continue
        n_aspect_pass += 1

        perimeter = cv2.arcLength(contour, True) + 1e-3
        compactness = 4 * math.pi * area_px / (perimeter**2)
        if compactness < compact_min:
            continue
        n_compact_pass += 1

        # Physical shadow length (pixels → meters via actual GSD), then Sentinel-2 model
        shadow_physical_m = long_axis_px * gsd_m
        shadow_len_m = min(shadow_physical_m, MAX_SHADOW_M)
        height_m = float(np.clip(shadow_len_m * tan_elevation, H_MIN, H_MAX))

        cx_roi = float(centroids[idx][0])
        cy_roi = float(centroids[idx][1])
        cx_full = roi_bounds.col_start + cx_roi
        cy_full = roi_bounds.row_start + cy_roi
        utm_e, utm_n = tile_meta.utm_from_pixel(cy_full, cx_full)

        length_m = float(np.clip(long_axis_px * gsd_m * 0.7, 5.0, 150.0))
        width_m = float(np.clip(short_axis_px * gsd_m * 0.7, 4.0, 80.0))
        confidence = confidence_base * (0.6 + 0.4 * min(compactness / 0.5, 1.0))
        # Per-building height uncertainty: σ_ĥ,i = tan(β_sun) × 10/√12 ≈ 4.3 m (paper Eq.4)
        sigma_h_m = round(tan_elevation * SENTINEL_GSD / math.sqrt(12), 2)

        rows.append(
            {
                "building_id": f"B{idx:05d}",
                "UTM_E": round(utm_e, 1),
                "UTM_N": round(utm_n, 1),
                "length_m": round(length_m, 1),
                "width_m": round(width_m, 1),
                "height_m": round(height_m, 2),
                "height_uncertainty_m": sigma_h_m,
                "height_conf": round(confidence, 3),
                "height_method": HeightMethod.SHADOW_GEO_FIXED.value,
                "shadow_len_m": round(shadow_len_m, 2),
                "long_axis_px": round(long_axis_px, 3),
                "long_clamped_px": round(min(long_axis_px, long_axis_max_px), 3),
                "aspect_ratio": round(aspect_ratio, 3),
                "compactness": round(compactness, 3),
                "area_px": area_px,
                "sun_elevation": round(tile_meta.sun_elevation_deg, 3),
                "sun_azimuth": round(tile_meta.sun_azimuth_deg, 3),
                "sem_type": classify_by_height(height_m),
                "_cx_roi": round(cx_roi, 1),
                "_cy_roi": round(cy_roi, 1),
            }
        )

    # Log filter cascade statistics
    logger.info("[BuildExtract] Filter cascade: total=%d → area=%d → contour=%d → aspect=%d → compact=%d",
                n_total, n_area_pass, n_contour_pass, n_aspect_pass, n_compact_pass)
    logger.info("[BuildExtract] Route B: Esri mask + Sentinel-2 model (image_GSD=%.3f, model_GSD=%.1f m/px): "
                "area=[%d,%d] px² (%.0f–%.0f m²), aspect<=%.1f, compact>=%.2f, shadow_max=%.1f px (%.0f m)",
                gsd_m, SENTINEL_GSD, area_min_px, area_max_px,
                area_min_px * gsd_m**2, area_max_px * gsd_m**2,
                aspect_max, compact_min, long_axis_max_px, MAX_SHADOW_M)

    n_before_sigma = len(rows)
    SIGMA_CLIP_MIN_SAMPLES = 10  # skip sigma clipping when sample too small for reliable statistics
    if len(rows) > SIGMA_CLIP_MIN_SAMPLES:
        heights = np.array([row["height_m"] for row in rows], dtype=float)
        mu = heights.mean()
        sigma = heights.std()
        lower, upper = mu - SIGMA_CLIP * sigma, mu + SIGMA_CLIP * sigma
        logger.info("[BuildExtract] Sigma clip (%.1fσ): mu=%.2f, σ=%.2f → keep range [%.2f, %.2f]",
                    SIGMA_CLIP, mu, sigma, lower, upper)
        # Collect outliers before filtering
        n_outliers = sum(1 for row in rows if not (lower <= row["height_m"] <= upper))
        rows = [row for row in rows if lower <= row["height_m"] <= upper]
        if n_outliers > 0:
            logger.info("[BuildExtract] Sigma clip removed %d outliers", n_outliers)
    else:
        logger.info("[BuildExtract] Sigma clip: skipped (n=%d ≤ 10, insufficient for statistics)", len(rows))

    n_before_conf = len(rows)
    # Confidence threshold: filters low-compactness merged blobs (road shadows, parking lots)
    # that pass area/aspect/compactness but have irregular shapes from ground-shadow merging.
    # At 0.93m GSD, compactness of merged blobs is typically 0.05–0.15 vs 0.3–0.6 for real
    # building shadows. Setting conf ≥ 0.75 indirectly enforces a compactness constraint
    # through the confidence formula: base(0.78) × (0.6 + 0.4×min(compactness/0.5, 1)).
    rows = [row for row in rows if row["height_conf"] >= 0.75]
    n_conf_removed = n_before_conf - len(rows)
    if n_conf_removed > 0:
        logger.info("[BuildExtract] Confidence filter (>=0.75) removed %d low-confidence blobs", n_conf_removed)

    logger.info("[BuildExtract] Final buildings: %d (from %d raw components)", len(rows), n_total)
    return pd.DataFrame(rows)


def shadow_result_is_reliable(buildings_df: pd.DataFrame) -> bool:
    n_buildings = len(buildings_df) if not buildings_df.empty else 0
    logger.info("[QualityGate-Post] ── Post-Extraction Quality Gate ──")

    # Check 1: building count
    check_count = n_buildings >= 12
    logger.info("[QualityGate-Post] Check 1 - Building count: n=%d, threshold=12 → %s",
                n_buildings, "PASS" if check_count else "FAIL")

    if not check_count:
        logger.warning("[QualityGate-Post] → REJECTED: insufficient buildings (%d < 12)", n_buildings)
        return False

    # Check 2: clipping ratio
    clipped_ratio = float((buildings_df["height_m"] >= H_MAX * 0.95).mean())
    n_clipped = int((buildings_df["height_m"] >= H_MAX * 0.95).sum())
    check_clipped = clipped_ratio < 0.35
    logger.info("[QualityGate-Post] Check 2 - Clipping ratio: %.3f (%d/%d at H_MAX=%.1f), threshold=0.35 → %s",
                clipped_ratio, n_clipped, n_buildings, H_MAX * 0.95,
                "PASS" if check_clipped else "FAIL")

    # Check 3: mean height in [4, 20]
    mean_height = float(buildings_df["height_m"].mean())
    check_mean = 4.0 <= mean_height <= 20.0
    logger.info("[QualityGate-Post] Check 3 - Mean height: %.2f m, range [4.0, 20.0] → %s",
                mean_height, "PASS" if check_mean else "FAIL")

    # Check 4: height std >= 1.5
    std_height = float(buildings_df["height_m"].std(ddof=0))
    check_std = std_height >= 1.5
    logger.info("[QualityGate-Post] Check 4 - Height std: %.2f m, threshold ≥1.5 → %s",
                std_height, "PASS" if check_std else "FAIL")

    # Additional diagnostics
    h_min = float(buildings_df["height_m"].min())
    h_max = float(buildings_df["height_m"].max())
    h_median = float(buildings_df["height_m"].median())
    logger.info("[QualityGate-Post] Height stats: min=%.2f, max=%.2f, median=%.2f, mean=%.2f, std=%.2f",
                h_min, h_max, h_median, mean_height, std_height)

    passed = check_count and check_clipped and check_mean and check_std
    if not passed:
        reasons = []
        if not check_clipped:
            reasons.append(f"clipping_ratio={clipped_ratio:.3f}≥0.35")
        if not check_mean:
            reasons.append(f"mean_height={mean_height:.2f}∉[4,20]")
        if not check_std:
            reasons.append(f"std_height={std_height:.2f}<1.5")
        logger.warning("[QualityGate-Post] → REJECTED: %s", "; ".join(reasons))
    else:
        logger.info("[QualityGate-Post] → ALL CHECKS PASSED ✓")

    return passed


def fallback_structured_prior(
    origin_e: float,
    origin_n: float,
    tile_meta: TileMetadata,
) -> pd.DataFrame:
    """Generate template buildings calibrated to MiTra A50 building stock.

    NOTE: The template coordinates (de, dn offsets) are hardcoded for the
    MiTra A50 Milan scene. For deployment in other regions, replace this
    template list with OSM building inventory for the target area.
    """
    tan_elevation = math.tan(math.radians(tile_meta.sun_elevation_deg))
    confidence_base = min(0.45 + tile_meta.sun_elevation_deg * 0.008, 0.78)
    rng = np.random.default_rng(42)
    templates = [
        (-310, 215, 75, 48, 6, 1.5, "industrial"),
        (-210, 258, 60, 38, 7, 1.5, "industrial"),
        (-100, 232, 80, 50, 8, 2.0, "industrial"),
        (55, 242, 90, 55, 10, 2.5, "industrial"),
        (215, 250, 65, 40, 7, 1.5, "industrial"),
        (345, 210, 72, 45, 6, 1.5, "industrial"),
        (-285, 188, 55, 35, 5, 1.0, "industrial"),
        (165, 268, 85, 52, 9, 2.0, "industrial"),
        (-355, -22, 120, 58, 5, 1.0, "industrial"),
        (318, -28, 115, 55, 5, 1.0, "industrial"),
        (-272, -174, 28, 14, 10, 2.0, "residential"),
        (-168, -157, 24, 13, 12, 2.0, "residential"),
        (-58, -187, 32, 15, 11, 2.0, "residential"),
        (58, -172, 27, 14, 9, 2.0, "residential"),
        (168, -184, 30, 16, 14, 2.0, "residential"),
        (282, -162, 29, 15, 12, 2.0, "residential"),
        (-382, -172, 26, 13, 8, 2.0, "residential"),
        (382, -177, 28, 14, 10, 2.0, "residential"),
        (102, -242, 35, 18, 20, 3.0, "mid_rise"),
        (222, -252, 38, 20, 22, 3.0, "mid_rise"),
        (-132, -237, 33, 17, 18, 3.0, "mid_rise"),
        (-242, 57, 48, 26, 8, 1.5, "residential"),
        (-132, 67, 55, 30, 10, 2.0, "residential"),
        (-30, 54, 42, 22, 9, 1.5, "residential"),
        (118, 62, 50, 28, 8, 1.5, "residential"),
        (242, 54, 46, 25, 9, 1.5, "residential"),
        (362, 60, 52, 30, 10, 1.5, "residential"),
    ]

    rows = []
    # Per C4 constraint: fallback height uncertainty by source tier
    for idx, (de, dn, length_m, width_m, height_mu, height_sigma, sem_type) in enumerate(templates):
        de += rng.uniform(-8, 8)
        dn += rng.uniform(-8, 8)
        height_m = float(np.clip(rng.normal(height_mu, height_sigma), H_MIN, H_MAX))
        shadow_len_m = height_m / tan_elevation
        shadow_len_m *= 1 + rng.uniform(-0.08, 0.08)
        height_m = float(np.clip(shadow_len_m * tan_elevation, H_MIN, H_MAX))
        confidence = float(confidence_base * rng.uniform(0.88, 1.0))
        # Fallback sigma by type: residential/industrial use default (5.0m),
        # mid_rise buildings reasonably approximated by type defaults
        sigma_h_m = 5.0  # type_default: no source-specific info, conservative = SHEM σ_ĥ
        rows.append(
            {
                "building_id": f"P{idx:03d}",
                "UTM_E": round(origin_e + de, 1),
                "UTM_N": round(origin_n + dn, 1),
                "length_m": round(float(np.clip(length_m + rng.uniform(-3, 3), 5, 150)), 1),
                "width_m": round(float(np.clip(width_m + rng.uniform(-2, 2), 4, 80)), 1),
                "height_m": round(height_m, 2),
                "height_uncertainty_m": sigma_h_m,
                "height_conf": round(confidence, 3),
                "height_method": HeightMethod.STRUCTURED_PRIOR.value,
                "shadow_len_m": round(shadow_len_m, 2),
                "long_axis_px": round(shadow_len_m / PIXEL_M, 3),
                "long_clamped_px": round(min(shadow_len_m / PIXEL_M, LONG_AXIS_MAX_PX), 3),
                "aspect_ratio": round(length_m / width_m, 3),
                "compactness": 0.70,
                "area_px": 0,
                "sun_elevation": round(tile_meta.sun_elevation_deg, 3),
                "sun_azimuth": round(tile_meta.sun_azimuth_deg, 3),
                "sem_type": sem_type,
                "_cx_roi": 0.0,
                "_cy_roi": 0.0,
            }
        )

    return pd.DataFrame(rows)


def check_metadata_gate(tile_meta: TileMetadata) -> tuple[bool, str]:
    """Pre-extraction quality gate based on satellite metadata alone (C1 constraints).

    PERFORMANCE OPTIMIZATION: enables early rejection of invalid satellite
    conditions without loading imagery (saves 60-120s per execution).
    The post-extraction gate (shadow_result_is_reliable) remains the
    authoritative quality check. This function provides fast pre-filtering only.

    Args:
        tile_meta: Parsed tile metadata including sun_elevation_deg, resolution_m,
                   and optionally cloud_cover_fraction.

    Returns:
        (True, "pass") or (False, reason_string)
    """
    sun_elev = tile_meta.sun_elevation_deg
    gsd = tile_meta.resolution_m
    cloud = tile_meta.cloud_cover_fraction  # may be None

    if sun_elev < 30:
        return False, f"sun_elevation {sun_elev:.1f}° < 30° threshold"
    if gsd > 10:
        return False, f"GSD {gsd:.1f}m > 10m threshold"
    if cloud is not None and cloud > 0.15:
        return False, f"cloud_cover {cloud:.2f} > 0.15 threshold"
    # If cloud is None (XML field unavailable), skip cloud check — do not block
    return True, "pass"


def infer_building_heights(
    paths: DatasetPaths,
    tile_meta: TileMetadata,
    uav_df: pd.DataFrame,
    buffer_m: float = 800.0,
) -> ShadowInferenceResult:
    t_start = time.perf_counter()

    # Stage 1: Pre-extraction metadata gate (C1 constraint)
    gate_pass, gate_reason = check_metadata_gate(tile_meta)
    if not gate_pass:
        t_elapsed = time.perf_counter() - t_start
        logger.info(
            "[MetadataGate] Rejected (%s, %.1fs). Switching to structured fallback.",
            gate_reason, t_elapsed,
        )
        origin_e, origin_n = compute_hover_origin(uav_df)
        fallback_df = fallback_structured_prior(origin_e, origin_n, tile_meta)
        # Load minimal ROI for overlay visualization even in fallback path
        roi_bounds = compute_roi_bounds(tile_meta, origin_e, origin_n, buffer_m=buffer_m)
        roi_rgb = load_roi_rgb(paths, roi_bounds)
        shadow_mask = np.zeros(roi_rgb.shape[:2], dtype=np.uint8)
        return ShadowInferenceResult(
            buildings_df=fallback_df,
            source_name=HeightMethod.STRUCTURED_PRIOR.value,
            source_reason=f"metadata gate failed: {gate_reason}",
            roi_rgb=roi_rgb,
            shadow_mask=shadow_mask,
            roi_bounds=roi_bounds,
            threshold_value=0.0,
            shadow_pixel_ratio=0.0,
        )

    # Stage 2: Full shadow extraction (gate passed)
    origin_e, origin_n = compute_hover_origin(uav_df)

    # Read actual image GSD and geotransform from the Esri World Imagery GeoTIFF
    actual_gsd, img_ulx, img_uly = _read_tiff_geotag(paths.tci_jp2)
    # Load full image to get dimensions (overrides Sentinel-2 tile dimensions)
    full_img = Image.open(paths.tci_jp2)
    img_w, img_h = full_img.width, full_img.height

    # Patch tile_meta with actual image parameters for ROI + coordinate transforms
    tile_meta.xdim = actual_gsd
    tile_meta.ydim = -actual_gsd  # PIL: Y increases downward
    tile_meta.ulx = img_ulx
    tile_meta.uly = img_uly
    tile_meta.rows = img_h
    tile_meta.cols = img_w
    logger.info("[ShadowExtract] Patched tile_meta: GSD=%.4f m/px, origin=(%.1f E, %.1f N), "
                "image=%dx%d px", actual_gsd, img_ulx, img_uly, img_w, img_h)

    # Use full image as ROI (already clipped to ~850m study area)
    roi_bounds = RoiBounds(row_start=0, row_end=img_h, col_start=0, col_end=img_w)
    logger.info("[ShadowExtract] Using full image as ROI: %dx%d px (%.0f x %.0f m)",
                img_w, img_h, img_w * actual_gsd, img_h * actual_gsd)
    roi_rgb = np.array(full_img.convert("RGB"), dtype=np.uint8)

    shadow_mask, shadow_stats = detect_shadows(roi_rgb, percentile=20.0)
    logger.info("[ShadowExtract] Shadow mask stats: threshold=%.2f, shadow_ratio=%.4f, "
                "light_mean=%.1f, light_std=%.1f",
                shadow_stats["threshold"], shadow_stats["shadow_pixel_ratio"],
                shadow_stats["light_mean"], shadow_stats["light_std"])

    # Count connected components BEFORE building extraction (raw blob count)
    raw_n, _ = cv2.connectedComponents(shadow_mask, connectivity=8)
    logger.info("[ShadowExtract] Raw connected components: %d (before filtering)", raw_n - 1)

    shadow_df = extract_buildings_from_shadow(shadow_mask, tile_meta, roi_bounds)
    if shadow_df is None:
        logger.error("[ShadowExtract] extract_buildings_from_shadow returned None! Creating empty DataFrame.")
        shadow_df = pd.DataFrame()
    logger.info("[ShadowExtract] Buildings after filtering: %d", len(shadow_df))

    if len(shadow_df) > 0:
        # Log per-building height distribution for diagnosis
        h_vals = shadow_df["height_m"].values
        logger.info("[ShadowExtract] Shadow-building heights: min=%.2f, max=%.2f, "
                    "mean=%.2f, median=%.2f, std=%.2f",
                    float(h_vals.min()), float(h_vals.max()),
                    float(h_vals.mean()), float(np.median(h_vals)), float(h_vals.std()))
        n_clipped = int((h_vals >= H_MAX * 0.95).sum())
        logger.info("[ShadowExtract] Clipped at H_MAX: %d/%d (%.1f%%)",
                    n_clipped, len(shadow_df), 100.0 * n_clipped / len(shadow_df))

        # Log semantic distribution
        sem_counts = shadow_df["sem_type"].value_counts().to_dict()
        logger.info("[ShadowExtract] Semantic distribution: %s", sem_counts)

    if shadow_result_is_reliable(shadow_df):
        logger.info("[ShadowExtract] ✓ ACCEPTED: shadow extraction passed quality gate (%d buildings).",
                    len(shadow_df))
        return ShadowInferenceResult(
            buildings_df=shadow_df,
            source_name=HeightMethod.SHADOW_GEO_FIXED.value,
            source_reason="direct shadow extraction passed quality gate",
            roi_rgb=roi_rgb,
            shadow_mask=shadow_mask,
            roi_bounds=roi_bounds,
            threshold_value=shadow_stats["threshold"],
            shadow_pixel_ratio=shadow_stats["shadow_pixel_ratio"],
        )

    logger.warning(
        "[ShadowExtract] ✗ REJECTED: shadow extraction failed quality gate "
        "(buildings=%d, shadow_ratio=%.4f). Switching to structured fallback.",
        len(shadow_df),
        shadow_stats["shadow_pixel_ratio"],
    )
    fallback_df = fallback_structured_prior(origin_e, origin_n, tile_meta)
    return ShadowInferenceResult(
        buildings_df=fallback_df,
        source_name=HeightMethod.STRUCTURED_PRIOR.value,
        source_reason=f"shadow extraction quality gate failed (n={len(shadow_df)}, "
                      f"ratio={shadow_stats['shadow_pixel_ratio']:.4f})",
        roi_rgb=roi_rgb,
        shadow_mask=shadow_mask,
        roi_bounds=roi_bounds,
        threshold_value=shadow_stats["threshold"],
        shadow_pixel_ratio=shadow_stats["shadow_pixel_ratio"],
    )


def save_buildings_csv(buildings_df: pd.DataFrame, paths: DatasetPaths) -> None:
    public_cols = [col for col in buildings_df.columns if not col.startswith("_")]
    buildings_df[public_cols].to_csv(paths.buildings_csv, index=False, encoding="utf-8-sig")
    buildings_df[public_cols].to_csv(paths.legacy_buildings_csv, index=False, encoding="utf-8-sig")


def save_shadow_overlay(result: ShadowInferenceResult, paths: DatasetPaths, tile_meta: TileMetadata) -> None:
    roi_bgr = cv2.cvtColor(result.roi_rgb, cv2.COLOR_RGB2BGR)
    overlay = roi_bgr.copy()
    overlay[result.shadow_mask > 0] = [140, 70, 30]
    blended = cv2.addWeighted(roi_bgr, 0.55, overlay, 0.45, 0)

    semantic_colors = {
        "industrial": (180, 90, 0),
        "residential": (0, 180, 60),
        "mid_rise": (0, 200, 200),
        "high_rise": (0, 0, 220),
        "commercial": (0, 160, 220),
    }
    for row in result.buildings_df.itertuples(index=False):
        if not hasattr(row, "_cx_roi") or not hasattr(row, "_cy_roi"):
            continue
        px = int(getattr(row, "_cx_roi"))
        py = int(getattr(row, "_cy_roi"))
        if not (0 <= px < blended.shape[1] and 0 <= py < blended.shape[0]):
            continue
        color = semantic_colors.get(row.sem_type, (200, 200, 200))
        cv2.circle(blended, (px, py), 4, color, -1)
        cv2.putText(
            blended,
            f"{row.height_m:.0f}m",
            (px + 4, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        blended,
        f"Sun elev={tile_meta.sun_elevation_deg:.2f} deg | az={tile_meta.sun_azimuth_deg:.2f} deg",
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 210, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        blended,
        result.source_reason,
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(paths.shadow_overlay_png), blended)


def build_lane_polyline(points: np.ndarray, samples: int = 10) -> list[list[float]]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projection = centered @ direction
    t_values = np.linspace(np.quantile(projection, 0.02), np.quantile(projection, 0.98), samples)
    polyline = []
    for t_value in t_values:
        point = center + direction * t_value
        polyline.append([round(float(point[0]), 2), round(float(point[1]), 2)])
    return polyline


def build_road_priors(vehicle_df: pd.DataFrame, origin_e: float, origin_n: float) -> list[RoadPrior]:
    if vehicle_df.empty:
        return []
    x_col = "x [m]"
    y_col = "y [m]"
    speed_col = "Speed [km/h]"
    lane_col = "Lane"
    vehicle_id_col = "Vehicle_ID"
    roads = []
    for lane_id, lane_df in vehicle_df.groupby(lane_col):
        if lane_df.empty:
            continue
        points = lane_df[[x_col, y_col]].to_numpy(dtype=float)
        if len(points) < 8:
            continue
        polyline_utm = build_lane_polyline(points)
        enu_x, enu_y = to_enu(
            [point[0] for point in polyline_utm],
            [point[1] for point in polyline_utm],
            origin_e,
            origin_n,
        )
        polyline_enu = [[round(float(e), 2), round(float(n), 2)] for e, n in zip(enu_x, enu_y)]
        road_type = "ramp" if int(lane_id) >= 10 else "mainline"
        roads.append(
            RoadPrior(
                road_id=f"LANE_{int(lane_id):02d}",
                lane_id=int(lane_id),
                road_type=road_type,
                polyline_utm=polyline_utm,
                polyline_enu=polyline_enu,
                vehicle_count=int(lane_df[vehicle_id_col].nunique()),
                mean_speed_kmh=round(float(lane_df[speed_col].mean()), 2),
            )
        )
    roads.sort(key=lambda road: road.lane_id)
    return roads


def point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t_value = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t_value * dx
    proj_y = ay + t_value * dy
    return math.hypot(px - proj_x, py - proj_y)


def attach_topology(buildings: list[BuildingPrior], roads: list[RoadPrior], neighbor_dist_m: float = 80.0) -> None:
    for building in buildings:
        building.neighbors = []

    for idx, building in enumerate(buildings):
        for other in buildings[idx + 1 :]:
            dist = math.hypot(
                building.geometry.centroid_enu_e - other.geometry.centroid_enu_e,
                building.geometry.centroid_enu_n - other.geometry.centroid_enu_n,
            )
            if dist <= neighbor_dist_m:
                building.neighbors.append(other.building_id)
                other.neighbors.append(building.building_id)

    for building in buildings:
        best_dist = float("inf")
        best_lane = None
        point = (building.geometry.centroid_enu_e, building.geometry.centroid_enu_n)
        for road in roads:
            if len(road.polyline_enu) < 2:
                continue
            for start, end in zip(road.polyline_enu[:-1], road.polyline_enu[1:]):
                dist = point_to_segment_distance(point, tuple(start), tuple(end))
                if dist < best_dist:
                    best_dist = dist
                    best_lane = road.lane_id
        if best_lane is not None:
            building.nearest_lane_id = int(best_lane)
            building.road_dist_m = round(best_dist, 2)

    for road in roads:
        road.occluding_buildings = []
        for building in buildings:
            if building.nearest_lane_id == road.lane_id and building.road_dist_m <= 30 and building.geometry.height_m >= 10:
                road.occluding_buildings.append(building.building_id)


def build_building_priors(buildings_df: pd.DataFrame, origin_e: float, origin_n: float) -> list[BuildingPrior]:
    building_priors = []
    for row in buildings_df.itertuples(index=False):
        enu_e, enu_n = to_enu([row.UTM_E], [row.UTM_N], origin_e, origin_n)
        geometry = BuildingGeometry(
            centroid_utm_e=float(row.UTM_E),
            centroid_utm_n=float(row.UTM_N),
            centroid_enu_e=round(float(enu_e[0]), 3),
            centroid_enu_n=round(float(enu_n[0]), 3),
            length_m=float(row.length_m),
            width_m=float(row.width_m),
            height_m=float(row.height_m),
        )
        building_priors.append(
            BuildingPrior(
                building_id=str(row.building_id),
                semantic_type=str(row.sem_type),
                geometry=geometry,
                height_conf=float(row.height_conf),
                height_method=str(row.height_method),
                sigma_h_m=float(row.height_uncertainty_m) if hasattr(row, "height_uncertainty_m") else 0.0,
                shadow_len_m=float(getattr(row, 'shadow_len_m', 0.0)),
                long_axis_px=float(getattr(row, 'long_axis_px', 0.0)),
                long_clamped_px=float(getattr(row, 'long_clamped_px', 0.0)),
                aspect_ratio=float(getattr(row, 'aspect_ratio', 0.0)),
                compactness=float(getattr(row, 'compactness', 0.0)),
                area_px=int(getattr(row, 'area_px', 0)),
            )
        )
    return building_priors


def export_prior_map(prior_map: LocationPriorMap, paths: DatasetPaths) -> None:
    """Export location prior map in LPSP-v0 flattened schema (v3 Appendix A).

    Key changes from legacy format:
    - "meta" → "metadata"
    - building coordinates flattened from nested geometry to top level
    - roads wrapped as {"lanes": [...], "ramps": [...]}
    - height_uncertainty_m field included per building
    """
    # Separate lanes and ramps
    lanes = []
    ramps = []
    for road in prior_map.roads:
        road_dict = road.to_dict()
        if road.road_type == "ramp":
            ramps.append(road_dict)
        else:
            lanes.append(road_dict)

    buildings_out = []
    for b in prior_map.buildings:
        buildings_out.append({
            "id": b.building_id,
            "centroid_enu_e": b.geometry.centroid_enu_e,
            "centroid_enu_n": b.geometry.centroid_enu_n,
            "centroid_utm_e": b.geometry.centroid_utm_e,
            "centroid_utm_n": b.geometry.centroid_utm_n,
            "length_m": b.geometry.length_m,
            "width_m": b.geometry.width_m,
            "height_m": b.geometry.height_m,
            "height_uncertainty_m": b.sigma_h_m,
            "confidence": b.height_conf,
            "semantic_class": b.semantic_type,
            "height_method": b.height_method,
            "shadow_len_m": b.shadow_len_m,
            "aspect_ratio": b.aspect_ratio,
            "compactness": b.compactness,
            "road_dist_m": b.road_dist_m,
            "nearest_lane_id": b.nearest_lane_id,
            "neighbors": b.neighbors,
        })

    data = {
        "metadata": {
            "lpgf_version": "1.0",
            "center_utm_e": prior_map.center_utm_e,
            "center_utm_n": prior_map.center_utm_n,
            "uav_altitude_m": prior_map.center_altitude_m,
            "radius_m": prior_map.radius_m,
            "acquisition_date": prior_map.acquisition_time,
            "solar_elevation_deg": prior_map.sun_elevation_deg,
            "solar_azimuth_deg": prior_map.sun_azimuth_deg,
            "uav_frame_count": prior_map.uav_frame_count,
            "vehicle_row_count": prior_map.vehicle_row_count,
            "summary": prior_map.summary(),
        },
        "buildings": buildings_out,
        "roads": {
            "lanes": lanes,
            "ramps": ramps,
        },
    }
    for output_path in (paths.prior_json, paths.legacy_prior_json):
        with output_path.open("w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, ensure_ascii=False, indent=2)


def write_summary_report(
    artifacts: PipelineArtifacts,
    shadow_result: ShadowInferenceResult,
    pipeline_runtime_s: float = 0.0,
) -> None:
    buildings_df = artifacts.buildings_df
    vehicle_df = artifacts.vehicle_df
    lane_count = int(vehicle_df["Lane"].nunique()) if not vehicle_df.empty and "Lane" in vehicle_df.columns else 0
    vehicle_count = int(vehicle_df["Vehicle_ID"].nunique()) if not vehicle_df.empty and "Vehicle_ID" in vehicle_df.columns else 0
    phase_counts = (
        artifacts.uav_df["Phase"].value_counts().to_dict()
        if "Phase" in artifacts.uav_df.columns
        else {"all": len(artifacts.uav_df)}
    )
    summary = artifacts.prior_map.summary()

    lines = [
        "Reusable 3D Urban Prior Summary",
        "=" * 72,
        "",
        f"Acquisition time:  {artifacts.tile_meta.acquisition_time}",
        f"CRS:               {artifacts.tile_meta.epsg_name} ({artifacts.tile_meta.epsg_code})",
        f"Sun elevation:     {artifacts.tile_meta.sun_elevation_deg:.3f} deg",
        f"Sun azimuth:       {artifacts.tile_meta.sun_azimuth_deg:.3f} deg",
        f"Pipeline runtime:  {pipeline_runtime_s:.2f} s",
        f"Height source:     {shadow_result.source_name}",
        f"Source reason:     {shadow_result.source_reason}",
        f"Shadow threshold:  {shadow_result.threshold_value:.2f}",
        f"Shadow ratio:      {shadow_result.shadow_pixel_ratio:.4f}",
        "",
        f"Buildings:         {len(buildings_df)}",
        f"Height min/max:    {buildings_df['height_m'].min():.2f} / {buildings_df['height_m'].max():.2f} m",
        f"Height mean:       {buildings_df['height_m'].mean():.2f} m",
        f"Height median:     {buildings_df['height_m'].median():.2f} m",
        f"Mean confidence:   {buildings_df['height_conf'].mean():.4f}",
        "",
        f"Road lanes:        {lane_count}",
        f"Vehicles:          {vehicle_count}",
        f"Vehicle rows:      {len(vehicle_df):,}",
        f"Telemetry frames:  {len(artifacts.uav_df):,}",
        f"Telemetry phases:  {phase_counts}",
        "",
        f"Prior summary:     {summary}",
        "",
        "Output files:",
        f"  buildings_csv:   {artifacts.paths.buildings_csv}",
        f"  shadow_overlay:  {artifacts.paths.shadow_overlay_png}",
        f"  prior_json:      {artifacts.paths.prior_json}",
        f"  prior_3d_png:    {artifacts.paths.prior_3d_png}",
    ]
    artifacts.paths.summary_txt.write_text("\n".join(lines), encoding="utf-8")


def render_3d_scene(
    uav_df: pd.DataFrame,
    buildings_df: pd.DataFrame,
    vehicle_df: pd.DataFrame,
    origin_e: float,
    origin_n: float,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(22, 14), facecolor="#0A0F1E")
    axis = fig.add_subplot(111, projection="3d")
    axis.set_facecolor("#0A0F1E")
    for axis_line in (axis.xaxis, axis.yaxis, axis.zaxis):
        axis_line.pane.fill = False
        axis_line.pane.set_edgecolor("#1A2B3C")
    axis.grid(True, color="#1A2B3C", linewidth=0.6, alpha=0.7, linestyle="--")

    legend_handles = []
    if not vehicle_df.empty:
        vehicle_e, vehicle_n = to_enu(vehicle_df["x [m]"], vehicle_df["y [m]"], origin_e, origin_n)
        speed = vehicle_df["Speed [km/h]"].to_numpy(dtype=float)
        speed_norm = (speed - speed.min()) / max(speed.max() - speed.min(), 1.0)
        step = max(1, len(vehicle_e) // 5000)
        scatter = axis.scatter(
            vehicle_e[::step],
            vehicle_n[::step],
            np.zeros(len(vehicle_e[::step])),
            c=speed_norm[::step],
            cmap="plasma",
            s=1.8,
            alpha=0.35,
            zorder=2,
        )
        colorbar = fig.colorbar(scatter, ax=axis, shrink=0.25, pad=0.005, aspect=25, location="left")
        colorbar.set_label("Vehicle speed (normalized)", color="#8899AA", fontsize=9)
        colorbar.ax.yaxis.set_tick_params(color="#8899AA")
        plt.setp(colorbar.ax.yaxis.get_ticklabels(), color="#8899AA", fontsize=8)
        legend_handles.append(mpatches.Patch(color="#FF7733", label="Vehicle trajectories"))

    used_semantics = set()
    bld_e, bld_n = to_enu(buildings_df["UTM_E"], buildings_df["UTM_N"], origin_e, origin_n)
    for idx, row in enumerate(buildings_df.itertuples(index=False)):
        style = POI_STYLE.get(str(row.sem_type), DEFAULT_STYLE)
        color = style["color"]
        east_m = float(bld_e[idx])
        north_m = float(bld_n[idx])
        axis.scatter([east_m], [north_m], [0.0], color=color, s=55, marker=style["marker_g"], alpha=0.92, zorder=6, depthshade=False)
        axis.scatter([east_m], [north_m], [float(row.height_m)], color=color, s=70, marker=style["marker_t"], alpha=0.98, zorder=7, depthshade=False)
        axis.plot(
            [east_m, east_m],
            [north_m, north_m],
            [0.0, float(row.height_m)],
            color=color,
            linewidth=1.3,
            linestyle="--",
            alpha=0.80,
            zorder=5,
        )
        if row.sem_type not in used_semantics:
            legend_handles.append(mpatches.Patch(color=color, label=style["label"]))
            used_semantics.add(row.sem_type)

    phases = uav_df["Phase"].astype(str).str.lower().to_numpy() if "Phase" in uav_df.columns else np.array(["hover"] * len(uav_df))
    uav_e, uav_n = to_enu(uav_df["UTM_E[m]"], uav_df["UTM_N[m]"], origin_e, origin_n)
    uav_z = uav_df["Altitude[m]"].to_numpy(dtype=float)
    for phase_name, phase_style in UAV_PHASE_STYLE.items():
        mask = phases == phase_name
        if mask.sum() < 2:
            continue
        axis.plot(
            uav_e[mask],
            uav_n[mask],
            uav_z[mask],
            color=phase_style["color"],
            linewidth=phase_style["lw"],
            alpha=0.95,
            zorder=12,
            solid_capstyle="round",
        )
        legend_handles.append(mpatches.Patch(color=phase_style["color"], label=phase_style["label"]))

    hover_mask = phases == "hover"
    if hover_mask.any():
        hover_e = float(uav_e[hover_mask].mean())
        hover_n = float(uav_n[hover_mask].mean())
        hover_z = float(uav_z[hover_mask].mean())
        axis.scatter(
            [hover_e],
            [hover_n],
            [hover_z],
            color="cyan",
            s=300,
            marker="*",
            zorder=20,
            depthshade=False,
            edgecolors="white",
            linewidths=0.5,
        )
        axis.plot([hover_e, hover_e], [hover_n, hover_n], [0, hover_z], color="cyan", linewidth=0.8, linestyle=":", alpha=0.30)
        legend_handles.append(mpatches.Patch(color="cyan", label=f"Local observation center ({hover_z:.0f}m)"))

    axis.set_xlabel("East (m)", color="#8899BB", labelpad=12, fontsize=10)
    axis.set_ylabel("North (m)", color="#8899BB", labelpad=12, fontsize=10)
    axis.set_zlabel("Altitude (m)", color="#8899BB", labelpad=10, fontsize=10)
    axis.tick_params(colors="#556677", labelsize=8)
    axis.set_xlim(-450, 450)
    axis.set_ylim(-450, 450)
    axis.set_zlim(0, max(float(buildings_df["height_m"].max()) + 20, float(uav_z.max()) + 15))
    axis.view_init(elev=26, azim=-48)

    legend = axis.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8.5,
        facecolor="#111C2A",
        edgecolor="#2A3D55",
        labelcolor="white",
        framealpha=0.88,
        ncol=1,
        title="Legend",
        title_fontsize=9,
    )
    legend.get_title().set_color("#AABBCC")

    title = (
        "MiTra A50 Milan - Reusable 3D Urban Prior Scene\n"
        f"Buildings={len(buildings_df)} | Height={buildings_df['height_m'].min():.1f}-{buildings_df['height_m'].max():.1f} m | "
        f"Telemetry frames={len(uav_df):,} | Vehicle rows={len(vehicle_df):,}"
    )
    axis.set_title(title, color="white", fontsize=10, pad=14)

    note = (
        f"ENU origin: ({origin_e:.1f}, {origin_n:.1f})\n"
        "Coordinate basis: local telemetry-derived center\n"
        "Building source: shadow + calibrated fallback\n"
        "Road context: vehicle GPS trajectories"
    )
    fig.text(
        0.73,
        0.03,
        note,
        color="#445566",
        fontsize=7.5,
        family="monospace",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="#0A0F1E", edgecolor="#1A2B3C", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0A0F1E")
    plt.close(fig)


class LocationPriorGenerator:
    def __init__(self, script_path: Optional[Path] = None):
        if script_path is None:
            script_path = Path(__file__)
        self.paths = resolve_dataset_paths(script_path)
        ensure_output_dirs(self.paths)

    def generate(self, buffer_m: float = 800.0, render_3d: bool = True) -> PipelineArtifacts:
        t_pipeline_start = time.perf_counter()
        logger.info("Loading dataset paths from %s", self.paths.workspace_root)
        tile_meta = parse_tile_metadata(self.paths)
        uav_df = load_csv(self.paths.uav_csv)
        vehicle_df = load_csv(self.paths.vehicle_csv)
        shadow_result = infer_building_heights(self.paths, tile_meta, uav_df, buffer_m=buffer_m)
        buildings_df = shadow_result.buildings_df.copy()
        save_buildings_csv(buildings_df, self.paths)
        save_shadow_overlay(shadow_result, self.paths, tile_meta)

        origin_e, origin_n = compute_hover_origin(uav_df)
        building_priors = build_building_priors(buildings_df, origin_e, origin_n)
        road_priors = build_road_priors(vehicle_df, origin_e, origin_n)
        attach_topology(building_priors, road_priors)
        prior_map = LocationPriorMap(
            center_utm_e=origin_e,
            center_utm_n=origin_n,
            center_altitude_m=float(uav_df["Altitude[m]"].mean()),
            radius_m=buffer_m,
            sun_elevation_deg=tile_meta.sun_elevation_deg,
            sun_azimuth_deg=tile_meta.sun_azimuth_deg,
            acquisition_time=tile_meta.acquisition_time,
            uav_frame_count=len(uav_df),
            vehicle_row_count=len(vehicle_df),
            buildings=building_priors,
            roads=road_priors,
        )
        # Render auxiliary figures (Fig 2: ENU topology, Fig 5: height stats)
        _render_enu_topology(building_priors, road_priors, origin_e, origin_n,
                             buffer_m, self.paths.output_dir)
        _render_height_statistics(buildings_df, self.paths.output_dir)

        export_prior_map(prior_map, self.paths)

        if render_3d:
            import shutil
            render_3d_scene(uav_df, buildings_df, vehicle_df, origin_e, origin_n, self.paths.prior_3d_png)
            for p in [self.paths.legacy_3d_png, self.paths.legacy_viz_png]:
                if p != self.paths.prior_3d_png:
                    shutil.copy(self.paths.prior_3d_png, p)

        artifacts = PipelineArtifacts(
            paths=self.paths,
            tile_meta=tile_meta,
            uav_df=uav_df,
            vehicle_df=vehicle_df,
            buildings_df=buildings_df,
            shadow_result=shadow_result,
            prior_map=prior_map,
        )
        t_pipeline_end = time.perf_counter()
        pipeline_runtime_s = round(t_pipeline_end - t_pipeline_start, 2)
        logger.info("[Pipeline] Total runtime: %.2fs", pipeline_runtime_s)
        write_summary_report(artifacts, shadow_result, pipeline_runtime_s)
        logger.info("Integrated prior generation complete.")

        # Generate theoretical σ_ĥ heatmap (Section 5.5)
        _render_sigma_heatmap(tile_meta, self.paths.output_dir)

        return artifacts


def _render_sigma_heatmap(tile_meta: TileMetadata, output_dir: Path) -> None:
    """Generate σ_ĥ heatmap: theoretical height uncertainty across β_sun × GSD.

    Renders a 2D contour heatmap of σ_ĥ = tan(β_sun) × GSD/√12, overlaid
    with the MiTra A50 operating points (Sentinel-2 10m and Esri 0.93m).
    """
    sun_elev_deg = np.linspace(20, 70, 51)   # solar elevation range
    gsd_vals_m = np.linspace(0.5, 10, 50)     # GSD range
    Sun, GSD = np.meshgrid(sun_elev_deg, gsd_vals_m)

    sigma_h = np.tan(np.radians(Sun)) * GSD / np.sqrt(12)

    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(Sun, GSD, sigma_h, levels=20, cmap="YlOrRd")
    cs = ax.contour(Sun, GSD, sigma_h, levels=[0.5, 1, 2, 3, 4.3, 6, 8, 10],
                    colors="black", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f m")

    # Mark MiTra A50 operating points
    ax.scatter([56.19], [10.0], c="blue", s=120, marker="o", edgecolors="white",
               zorder=5, label="MiTra A50 (Sentinel-2, 10 m GSD, σ=4.3 m)")
    ax.scatter([56.19], [0.9346], c="green", s=120, marker="s", edgecolors="white",
               zorder=5, label="MiTra A50 (Esri, 0.93 m GSD, σ=0.40 m)")

    cbar = fig.colorbar(cf, ax=ax, label=r"$\sigma_{\hat{h}}$ (m)")
    ax.set_xlabel("Solar Elevation β_sun (°)")
    ax.set_ylabel("Ground Sampling Distance (m/px)")
    ax.set_title("Theoretical Height Uncertainty σ_ĥ = tan(β_sun) × GSD / √12\n"
                 "with MiTra A50 Operating Points")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / "sigma_heatmap_v5.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    

def _render_enu_topology(building_priors, road_priors, origin_e, origin_n, radius_m, output_dir):
    """Fig 2: ENU topology map."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0A0F1E")
    ax.set_facecolor("#0A0F1E")
    circle = plt.Circle((0, 0), radius_m, fill=False, color="#4A6FA5", linewidth=1, linestyle="--", alpha=0.6)
    ax.add_patch(circle)
    for b in building_priors:
        style = POI_STYLE.get(b.semantic_type, DEFAULT_STYLE)
        e, n = b.geometry.centroid_enu_e, b.geometry.centroid_enu_n
        size = max(15, b.geometry.height_m * 4)
        ax.scatter(e, n, s=size, c=style["color"], alpha=0.85, edgecolors="white", linewidth=0.3, zorder=5)
    lane_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(road_priors))))
    for idx, road in enumerate(road_priors):
        if len(road.polyline_enu) < 2:
            continue
        xs = [pt[0] for pt in road.polyline_enu]
        ys = [pt[1] for pt in road.polyline_enu]
        ax.plot(xs, ys, color=lane_colors[idx % 10], linewidth=1.5, alpha=0.7,
                label="A50 Highway" if idx == 0 else None)
    ax.scatter([0], [0], c="cyan", s=300, marker="*", zorder=10, edgecolors="white", linewidths=0.5,
               label="Local observation center")
    handles = []
    for sem, style in POI_STYLE.items():
        if any(b.semantic_type == sem for b in building_priors):
            handles.append(mpatches.Patch(color=style["color"], label=style["label"]))
    ax.legend(handles=handles, loc="upper right", fontsize=7, facecolor="#111C2A",
              edgecolor="#2A3D55", labelcolor="white")
    ax.set_xlabel("East (m)", color="#8899BB"); ax.set_ylabel("North (m)", color="#8899BB")
    ax.tick_params(colors="#556677")
    ax.set_xlim(-radius_m * 1.1, radius_m * 1.1); ax.set_ylim(-radius_m * 1.1, radius_m * 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"ENU Topology Map ({len(building_priors)} buildings)", color="white", fontsize=11)
    ax.grid(True, color="#1A2B3C", linewidth=0.5, alpha=0.5)
    out_path = output_dir / "enu_topology_map_v5.png"
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0A0F1E")
    plt.close(fig)
    logger.info("Saved ENU topology map (Fig 2): %s", out_path)


def _render_height_statistics(buildings_df, output_dir):
    """Fig 5: Height statistics 4-panel."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    heights = buildings_df["height_m"]
    ax = axes[0, 0]
    ax.hist(heights, bins=12, color="#4A90D9", edgecolor="white", alpha=0.8)
    ax.axvline(heights.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={heights.mean():.1f}m")
    ax.axvline(heights.median(), color="green", linestyle=":", linewidth=1.5, label=f"Median={heights.median():.1f}m")
    ax.set_xlabel("Height (m)"); ax.set_ylabel("Count")
    ax.set_title(f"(a) Height Distribution (n={len(heights)})"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax = axes[0, 1]
    sem_types = list(buildings_df["sem_type"].unique())
    box_data = [buildings_df[buildings_df["sem_type"] == s]["height_m"].values for s in sem_types]
    cmap = {"industrial": "#8E44AD", "residential": "#27AE60", "mid_rise": "#E67E22", "high_rise": "#E74C3C"}
    bp = ax.boxplot(box_data, tick_labels=sem_types, patch_artist=True)
    for patch, sem in zip(bp["boxes"], sem_types):
        patch.set_facecolor(cmap.get(sem, "#BDC3C7")); patch.set_alpha(0.6)
    ax.set_ylabel("Height (m)"); ax.set_title("(b) Height by Semantic Type"); ax.grid(axis="y", alpha=0.3)
    ax = axes[1, 0]
    counts = buildings_df["sem_type"].value_counts()
    pie_colors = [cmap.get(s, "#BDC3C7") for s in counts.index]
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=pie_colors, startangle=90)
    ax.set_title(f"(c) Semantic Distribution (n={len(heights)})")
    ax = axes[1, 1]
    for sem in sem_types:
        sub = buildings_df[buildings_df["sem_type"] == sem]
        ax.scatter(sub["height_m"], sub["height_conf"], c=cmap.get(sem, "#BDC3C7"),
                   label=sem, alpha=0.6, s=30, edgecolors="white", linewidth=0.2)
    ax.set_xlabel("Height (m)"); ax.set_ylabel("Confidence")
    ax.set_title("(d) Height vs Confidence"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"MiTra A50 Building Height Statistics (n={len(heights)})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = output_dir / "height_statistics_v5.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved height statistics (Fig 5): %s", out_path)



def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-gated urban location prior generation pipeline.")
    parser.add_argument("--buffer-m", type=float, default=800.0, help="ROI radius around local observation center in meters.")
    parser.add_argument("--skip-3d", action="store_true", help="Skip 3D rendering.")
    args = parser.parse_args()

    generator = LocationPriorGenerator()
    artifacts = generator.generate(buffer_m=args.buffer_m, render_3d=not args.skip_3d)
    logger.info("Buildings CSV: %s", artifacts.paths.buildings_csv)
    logger.info("Prior JSON: %s", artifacts.paths.prior_json)
    logger.info("3D scene PNG: %s", artifacts.paths.prior_3d_png)
    logger.info("Summary TXT: %s", artifacts.paths.summary_txt)


if __name__ == "__main__":
    main()
