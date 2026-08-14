import os
import csv
import tempfile
import datetime
import util

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

CONF_THRESHOLD = 0.01       # Keep low for small objects

# Adjusted to match 4 quadrants of a 12768 x 9564 image
SLICE_WIDTH = 6384           
SLICE_HEIGHT = 4782
OVERLAP_RATIO = 0.2         # 20% overlap to catch seals on the edge of tiles
DEVICE = "cuda:0"           # Use 'cpu' if no GPU


def is_image_file(filename):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    return filename.lower().strip().endswith(valid_exts)


def run_inference(input_dir: str, output_file_path: str, config: dict):
    """
    Core inference logic for Ultralytics YOLO models.
    """
    print("[MODEL] Starting YOLO inference process...")
    print(f"[MODEL] config:{config}")
    
    # 1. Resolve Weights
    weights_path = "/workspace/model.pt" # Default baked-in weights
    custom_weights_uri = config.get("weights")
    print(f'[MODEL] custom_weights_uri:{custom_weights_uri}', flush=True)
    
    temp_dir = tempfile.TemporaryDirectory()

    if custom_weights_uri:
        custom_weights = custom_weights_uri.split("/")[-1]
        weights_path = os.path.join(temp_dir.name, custom_weights)
        util.download_gcs_uri(custom_weights_uri, weights_path)
    
    print(f"[MODEL] Loading YOLO model: {weights_path} into SAHI...", flush=True)

    conf_threshold = config.get("options", {}).get("conf", CONF_THRESHOLD)  
    slice_width = config.get("slice_width", SLICE_WIDTH)
    slice_height = config.get("slice_height", SLICE_HEIGHT)
    overlap_ratio = config.get("overlap_ratio", OVERLAP_RATIO)
    print(f"[CONFIG] conf_threshold={conf_threshold}, slice_width={slice_width}, slice_height={slice_height}, overlap_ratio={overlap_ratio}", flush=True)
    
    # Load Model with full error logging (do not suppress exceptions)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path= weights_path,
        confidence_threshold= conf_threshold,
        device=DEVICE, 
    )
    
    print(f"[MODEL] Finished loading the model")

  
    # 2. Discover Files with full paths
    image_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        if os.path.isfile(os.path.join(input_dir, f)) and is_image_file(f)
    ]
    
    if not image_paths:
        print("[MODEL] WARNING: No input files found in directory!")
        
    # 3. Prepare CSV Output
    print(f'[MODEL] Writing output to: {output_file_path}')
    with open(output_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # VIAME Headers
        writer.writerow([
            "# 1: Detection or Track-id", "2: Video or Image Identifier", 
            "3: Unique Frame Identifier", "4-7: Img-bbox(TL_x", "TL_y", "BR_x", "BR_y)", 
            "8: Detection or Length Confidence", "9: Target Length (0 or -1 if invalid)", 
            "10-11+: Repeated Species", "Confidence Pairs or Attributes"
        ])
        
        # Metadata Header
        current_time = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        writer.writerow([
            "# metadata", "exec_time: 0", "exported_by: python_sahi_script", 
            f"exported_at: {current_time}", "", "", "", "", "", "", ""
        ])

        detection_id = 1
        
        # 4. Iterate Through Images
        for i, full_img_path in enumerate(image_paths):
            filename = os.path.basename(full_img_path)
            print(f"[MODEL] [{i+1}/{len(image_paths)}] Slicing & Detecting: {filename}")
            
            try:
                result = get_sliced_prediction(
                    full_img_path,
                    detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_ratio,
                    overlap_width_ratio=overlap_ratio,
                    verbose=0
                )
                
                # Parse SAHI Results for VIAME CSV
                for object_prediction in result.object_prediction_list:
                    bbox = object_prediction.bbox
                    minx, miny, maxx, maxy = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                    
                    score = object_prediction.score.value
                    category_name = object_prediction.category.name
                    
                    row = [
                        detection_id,          # 1. Detection ID
                        filename,              # 2. Image Filename
                        i,                     # 3. Frame ID
                        f"{minx:.3f}",         # 4. TL_x
                        f"{miny:.3f}",         # 5. TL_y
                        f"{maxx:.3f}",         # 6. BR_x
                        f"{maxy:.3f}",         # 7. BR_y
                        f"{score:.5f}",        # 8. Confidence
                        0,                     # 9. Target Length
                        category_name,         # 10. Class
                        f"{score:.5f}"         # 11. Attribute
                    ]
                    
                    writer.writerow(row)
                    detection_id += 1
                    
            except Exception as e:
                print(f"[MODEL ERROR] Failed to process {filename}: {e}", flush=True)
                raise e

        print(f"[MODEL] Saved SAHI detections to: {output_file_path}")
        
    # Cleanup temp directory holding custom weights
    temp_dir.cleanup()
    print("[MODEL] Inference complete!")