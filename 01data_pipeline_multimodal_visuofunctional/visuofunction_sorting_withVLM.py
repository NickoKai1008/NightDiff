import os, re, base64, httpx, pandas as pd, geopandas as gpd
from openai import OpenAI
from tqdm import tqdm

# ---------- helpers --------------------------------------------------------
def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

def img2dataurl(path):
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def load_poi_data(poi_data_path):
    """
    Load POI dominant type data from CSV or shapefile
    
    Parameters:
    poi_data_path: str - Path to CSV file or shapefile containing POI data
    
    Returns:
    dict - Mapping from point_id to dominant_poi type
    """
    poi_mapping = {}
    
    if poi_data_path.lower().endswith('.csv'):
        # Load from CSV file
        df = pd.read_csv(poi_data_path)
        for _, row in df.iterrows():
            poi_mapping[str(row['point_id'])] = row.get('dominant_poi', 'unknown')
    
    elif poi_data_path.lower().endswith('.shp'):
        # Load from shapefile
        gdf = gpd.read_file(poi_data_path)
        for _, row in gdf.iterrows():
            poi_mapping[str(row['point_id'])] = row.get('dominant_poi', 'unknown')
    
    else:
        raise ValueError("POI data file must be either .csv or .shp format")
    
    return poi_mapping

def extract_point_id_from_filename(filename):
    """
    Extract point_id from image filename
    Assumes filename format like: point_123.jpg, 123.jpg, or image_123.png
    
    Parameters:
    filename: str - Image filename
    
    Returns:
    str - Extracted point_id or None if not found
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Try to extract number from filename
    numbers = re.findall(r'\d+', name_without_ext)
    if numbers:
        return numbers[-1]  # Take the last number found
    
    return None

# ---------- OpenAI-compatible client (DashScope) ---------------------------
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-xxxx",
    http_client=httpx.Client(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                             follow_redirects=True),
)

# ---------- prompt ---------------------------------------------------------
PROMPT = (
    "You are an urban-geography expert specializing in visuofunctional analysis.\n"
    "Given a street-view photo and the dominant POI type visible from this location, "
    "evaluate the visuofunctional POI level (1-10) that represents the urban intensity and development density.\n\n"
    
    "Use this specific scale:\n"
    "• 10: Very dense CBD / luxury shopping streets, landmark skyscrapers, heavy pedestrian and retail traffic.\n"
    "• 8–9: Downtown core, mixed high-rise offices and retail establishments.\n"
    "• 5–7: Typical urban neighbourhoods with mid-rise residential buildings, local retail shops and moderate pedestrian activity.\n"
    "• 3–4: Suburban fringes, characterised by lower-density development and sparse commercial activities.\n"
    "• 1–2: Rural roads, open fields, industrial backlots, and sparsely built-up areas with minimal human activity.\n\n"
    
    "Consider both:\n"
    "1. Visual evidence from the street scene (building density, height, activity level, infrastructure quality)\n"
    "2. The dominant POI type context (commercial vs residential vs industrial functions)\n\n"
    
    "Reply with **exactly one number (1-10)** representing the visuofunctional POI level.\n"
    "Example: 7"
)

def infer_poi_level(img_path, dominant_poi):
    """
    Infer POI level based on street view image and dominant POI type
    
    Parameters:
    img_path: str - Path to street view image
    dominant_poi: str - Dominant POI type from isovist analysis
    
    Returns:
    int - POI level (1-10) or None if error
    """
    dataurl = img2dataurl(img_path)
    
    # Create context message with POI information
    poi_context = f"The dominant POI type visible from this location is: {dominant_poi}"
    
    try:
        rsp = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f"{poi_context}\n\nHere is the street view photo:"},
                    {"type": "image_url", "image_url": {"url": dataurl}}
                ]}
            ],
        )
        
        result = rsp.choices[0].message.content.strip()
        # Extract number from response
        poi_level = int(re.findall(r'\d+', result)[0])
        return poi_level if 1 <= poi_level <= 10 else None
        
    except Exception as e:
        print("Error on", os.path.basename(img_path), "→", e)
        return None

# ---------- batch loop -----------------------------------------------------
if __name__ == "__main__":
    # Configuration
    IMG_DIR = r"C:/Users/pc/Desktop/skill/depthanything/image"
    POI_DATA_PATH = "isovist_polygons.shp"  # Can be .shp or .csv file
    OUT_CSV = os.path.join(IMG_DIR, "visuofunction_poi_levels.csv")
    
    # Load POI dominant type data
    print("Loading POI data...")
    try:
        poi_mapping = load_poi_data(POI_DATA_PATH)
        print(f"Loaded POI data for {len(poi_mapping)} points")
    except Exception as e:
        print(f"Error loading POI data: {e}")
        exit(1)
    
    # Get image files
    files = sorted([f for f in os.listdir(IMG_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                   key=natural_key)
    
    print(f"Found {len(files)} image files")
    
    # Process each image
    rows = []
    processed_count = 0
    
    for fn in tqdm(files, desc="Analyzing visuofunction"):
        # Extract point_id from filename
        point_id = extract_point_id_from_filename(fn)
        
        if point_id is None:
            print(f"Warning: Could not extract point_id from filename {fn}")
            continue
        
        # Get corresponding dominant POI type
        dominant_poi = poi_mapping.get(point_id, "unknown")
        
        if dominant_poi == "unknown":
            print(f"Warning: No POI data found for point_id {point_id} (file: {fn})")
            continue
        
        # Analyze image with POI context
        poi_level = infer_poi_level(os.path.join(IMG_DIR, fn), dominant_poi)
        
        if poi_level is not None:
            rows.append({
                "filename": fn,
                "point_id": point_id,
                "dominant_poi": dominant_poi,
                "visuofunction_poi_level": poi_level
            })
            processed_count += 1
    
    # Save results
    if rows:
        df_results = pd.DataFrame(rows)
        df_results.to_csv(OUT_CSV, index=False, encoding="utf-8")
        print(f">> Successfully processed {processed_count} images")
        print(f">> Saved results to {OUT_CSV}")
        
        # Display summary statistics
        print(f"\nVisuofunction POI Level Distribution:")
        level_counts = df_results['visuofunction_poi_level'].value_counts().sort_index()
        for level, count in level_counts.items():
            percentage = (count / len(df_results)) * 100
            print(f"  Level {level}: {count} images ({percentage:.1f}%)")
        
        print(f"Average POI level: {df_results['visuofunction_poi_level'].mean():.2f}")
    else:
        print("No images were successfully processed. Please check:")
        print("1. Image filenames contain point_id numbers")
        print("2. POI data file contains matching point_id values")
        print("3. Image files are accessible and valid")