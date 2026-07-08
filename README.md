# LPGF: Quality-Gated Open Geospatial Data Fusion for Reusable 3D Urban Location Priors

A quality-gated open geospatial data fusion pipeline for generating reusable 3D urban location priors from incomplete urban data. LPGF integrates OpenStreetMap building footprints, satellite imagery, local telemetry, and vehicle GPS trajectories to produce footprint-level height estimates, uncertainty bounds, semantic classes, and provenance metadata.

> **Paper**: "Quality-Gated Open Geospatial Data Fusion for Reusable 3D Urban Location Priors"  
> **Dataset**: MiTra A50 (Milan, Italy) -- [doi:10.1038/s41597-025-05472-0](https://doi.org/10.1038/s41597-025-05472-0)

## Pipeline Overview

1. **Region Delimitation** -- fixed local urban context window
2. **GIS Skeleton** -- OSM building footprints + road-network context from GPS traces
3. **Height Inference** -- structured 3-tier fallback + optional shadow-based estimation (SHEM)
4. **Prior Assembly** -- buildings with height, confidence, semantics, uncertainty, and provenance
5. **Export** -- LPSP-v0 JSON format

## Quick Start

```bash
pip install -r requirements.txt
python pipeline.py --buffer-m 800
```

Output: `lpsp_v0_prior.json` + 3D visualization + summary report.

## Requirements

- Python 3.8+
- See `requirements.txt`

## License

MIT
