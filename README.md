# LPGF: Location Prior Generation Framework

A multi-source data fusion pipeline for generating 3D building height priors in low-altitude urban airspace. Integrates Sentinel-2 satellite imagery, UAV telemetry, vehicle GPS trajectories, and OpenStreetMap footprints.

> **Paper**: "Location Prior Generation Framework: A Multi-Source Data Fusion Pipeline for Low-Altitude Urban Air Mobility"  
> **Dataset**: MiTra A50 (Milan, Italy) — [doi:10.1038/s41597-025-05472-0](https://doi.org/10.1038/s41597-025-05472-0)

## Pipeline Overview

1. **Region Delimitation** — 800m radius around UAV hover point
2. **GIS Skeleton** — OSM building footprints + SVD-extracted lane centerlines from GPS
3. **Height Inference** — Structured 3-tier fallback + optional shadow-based estimation (SHEM)
4. **Prior Assembly** — Buildings with height, confidence, semantics, occlusion flags
5. **Export** — LPSP-v0 JSON format

## Quick Start

```bash
pip install -r requirements.txt
python pipeline.py --data-root /path/to/mitra_a50/
```

Output: `lpsp_v0_prior.json` + 3D visualization + summary report.

## Requirements

- Python 3.8+
- See `requirements.txt`

## License

MIT
