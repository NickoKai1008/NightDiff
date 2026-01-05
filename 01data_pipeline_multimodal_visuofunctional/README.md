# Isovist Analysis Script

This Python script performs visibility analysis (Isovist) based on building, POI, and road network data.

## Features

1. **Generate observation points along roads**: Generate observation points at 50-meter intervals along road networks
2. **Calculate Isovist geometry**: Calculate visible areas for each point using building edges as visual obstructions
3. **Expand visible areas**: Expand isovist geometry outward by 3 meters
4. **POI analysis**: Analyze dominant POI types and diversity within each isovist
5. **Output results**: Generate shapefile files containing analysis results

## Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

## Data Requirements

The script supports two input modes with corresponding file structures:

### Mode 1: Using pre-segmented point files
```
Current Directory/
├── ad/          # Building data
│   └── ad.shp
├── poi/         # POI data (requires fclass field)
│   └── poi.shp
├── pt/          # Pre-segmented observation points
│   └── pt.shp
└── isovist_analysis_optimized.py
```

### Mode 2: Generate points from roads (default)
```
Current Directory/
├── ad/          # Building data
│   └── ad.shp
├── poi/         # POI data (requires fclass field)
│   └── poi.shp
├── rd/          # Road data
│   └── rd.shp
└── isovist_analysis_optimized.py
```

## Usage

We provide two versions of the script:

### Basic version (isovist_analysis.py)
Suitable for small-scale data and learning purposes:

```bash
python isovist_analysis.py
```

### Optimized version (isovist_analysis_optimized.py) 
Recommended for production environments with better performance and error handling:

```bash
# Mode 1: Using pre-segmented point files (default parameters)
python isovist_analysis_optimized.py --mode 1

# Mode 2: Generate points from roads (default mode)
python isovist_analysis_optimized.py --mode 2 --interval 50

# Custom parameters example
python isovist_analysis_optimized.py --mode 2 --interval 30 --max-distance 150 --num-rays 36 --buffer 5
```

**Command line parameters:**
- `--mode`: Input mode (1=pre-segmented point file, 2=generate from roads, default 2)
- `--interval`: Observation point interval in meters (only valid for mode 2, default 50)
- `--max-distance`: Maximum visibility distance in meters (default 200)
- `--num-rays`: Number of rays (default 72)
- `--buffer`: Isovist expansion distance in meters (default 3)
- `--batch-size`: Batch processing size (default 100)

**Input mode details:**
- **Mode 1**: Directly use pre-segmented observation points from `pt/pt.shp`
- **Mode 2**: Automatically generate equally spaced observation points based on `rd/rd.shp` road network

## Output Files

The script generates the following output files:

1. **isovist_polygons.shp**: Isovist polygon data with the following fields:
   - `point_id`: Corresponding observation point ID
   - `dominant_poi`: Dominant POI type (most frequent type)
   - `shannon_entropy`: POI diversity (Shannon entropy)
   - `poi_count`: Total POI count within isovist
   - `unique_poi_types`: Number of unique POI types
   - `isovist_area`: Isovist area (square meters)

2. **observation_points.shp**: Observation point data containing all analysis result fields above

## Technical Specifications

- **Coordinate System**: Automatically unifies all data to EPSG:3857 (Mercator projection)
- **Visibility Range**: Default maximum visibility distance of 200 meters
- **Ray Count**: Uses 72 rays to calculate isovist (5-degree intervals)
- **POI Field Detection**: Automatically detects POI type fields (type, category, class, etc.), prioritizes `fclass`

## Parameter Adjustment

You can adjust the following parameters via command line:

- `--interval=50`: Observation point interval (meters)
- `--max-distance=200`: Maximum visibility distance (meters)
- `--num-rays=72`: Number of rays
- `--buffer=3`: Isovist expansion distance (meters)

## Notes

1. Ensure shapefile data integrity (includes .shp, .dbf, .shx, .prj files)
2. POI data must contain type classification field (preferably `fclass`)
3. Processing time depends on road network density and building quantity
4. Recommend batch processing for large-scale data

---

# Visuofunction Analysis with VLM

This complementary script (`visuofunction_sorting_withVLM.py`) uses Vision-Language Models (VLM) to analyze visuofunctional POI levels by combining street view imagery with isovist analysis results.

## Overview

The visuofunction analysis leverages both visual evidence from street-view photos and contextual information from dominant POI types (derived from isovist analysis) to assess urban intensity and development density on a 1-10 scale.

## POI Level Scale

- **10**: Very dense CBD / luxury shopping streets, landmark skyscrapers, heavy pedestrian and retail traffic
- **8–9**: Downtown core, mixed high-rise offices and retail establishments  
- **5–7**: Typical urban neighbourhoods with mid-rise residential buildings, local retail shops and moderate pedestrian activity
- **3–4**: Suburban fringes, characterised by lower-density development and sparse commercial activities
- **1–2**: Rural roads, open fields, industrial backlots, and sparsely built-up areas with minimal human activity

## Requirements

Additional dependencies for VLM analysis:

```bash
pip install openai httpx geopandas
```

## Input Data

The script requires two main inputs:

1. **Street view images**: Directory containing street view photos with filenames that include point_id numbers (e.g., `point_123.jpg`, `123.png`)
2. **POI analysis results**: Either:
   - `isovist_polygons.shp` (output from isovist analysis)
   - CSV file with `point_id` and `dominant_poi` columns

## Configuration

Edit the script configuration section:

```python
IMG_DIR = "path/to/street/view/images"           # Directory containing street view photos
POI_DATA_PATH = "isovist_polygons.shp"          # POI analysis results (.shp or .csv)
OUT_CSV = "visuofunction_poi_levels.csv"        # Output file path
```

## Usage

Run the visuofunction analysis:

```bash
python visuofunction_sorting_withVLM.py
```

## Data Matching

The script automatically matches images to POI data based on:
- **Point ID extraction**: Extracts numeric IDs from image filenames
- **POI mapping**: Links point IDs to dominant POI types from analysis results
- **Contextual analysis**: Combines visual and functional information for comprehensive assessment

## Output

The script generates a CSV file containing:

- `filename`: Original image filename
- `point_id`: Extracted point identifier  
- `dominant_poi`: Dominant POI type from isovist analysis
- `visuofunction_poi_level`: Assessed POI level (1-10)

## Example Workflow

1. **Run isovist analysis** to generate `isovist_polygons.shp` with dominant POI types
2. **Collect street view images** at corresponding observation points 
3. **Configure and run** visuofunction analysis script
4. **Analyze results** for urban planning and development assessment

## Performance Notes

- Processing time depends on number of images and VLM API response times
- Batch processing with progress tracking included
- Error handling for missing data or invalid responses
- Summary statistics provided upon completion