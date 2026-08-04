import os
import json
import tempfile
from ultralytics import YOLO
import util

def run_inference(input_dir: str, output_file_path: str, config: dict):
    """
    Core inference logic for Ultralytics YOLO models.
    
    Parameters
    ----------
    input_dir : str
        Local directory where all input images/videos have been downloaded.
    output_file_path : str
        The exact local file path where final KWCOCO results MUST be saved.
    config : dict
        The configuration dictionary from the Airflow payload.
    """
    print("[MODEL] Starting YOLO inference process...")
    
    # 1. Resolve Weights
    weights_path = "/workspace/model.pt" # Default baked-in weights
    custom_weights_uri = config.get("weights")
    
    # Use a temporary directory for custom weights to avoid polluting the workspace
    # across multiple invocations in the same container.
    temp_dir = tempfile.TemporaryDirectory()
    
    if custom_weights_uri:
        print(f"[MODEL] Dynamic weights override detected. Downloading {custom_weights_uri}...")
        weights_path = os.path.join(temp_dir.name, "custom_model.pt")
        util.download_gcs_uri(custom_weights_uri, weights_path)
    
    print(f"[MODEL] Loading YOLO model from {weights_path}...")
    model = YOLO(weights_path)
    
    # 2. Extract YOLO Options
    # Any keys inside "options" are passed directly to YOLO's predict method.
    options = config.get("options", {})
    print(f"[MODEL] Using YOLO inference options: {options}")

    # 3. Setup Output Structure (KWCOCO format)
    kwcoco_output = {
        "info": {"description": "Ultralytics YOLO Output"},
        "categories": [],
        "videos": [],
        "images": [],
        "annotations": []
    }
    
    # Populate categories dynamically from the model's loaded names
    for class_id, class_name in model.names.items():
        kwcoco_output["categories"].append({
            "id": int(class_id),
            "name": str(class_name)
        })
        
    # 4. Discover and Process Files
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if not input_files:
        print("[MODEL] WARNING: No input files found in directory!")
        
    video_id_counter = 1
    image_id_counter = 1
    annotation_id_counter = 1
    
    for filename in input_files:
        filepath = os.path.join(input_dir, filename)
        is_video = filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        current_vid_id = None
        if is_video:
            kwcoco_output["videos"].append({
                "id": video_id_counter,
                "name": filename
            })
            current_vid_id = video_id_counter
            video_id_counter += 1
            
        print(f"[MODEL] Processing {filename}...")
        
        # Run YOLO inference
        # YOLO handles both images and videos seamlessly, yielding results frame-by-frame.
        results = model.predict(source=filepath, stream=True, **options)
        
        for frame_idx, result in enumerate(results):
            height, width = result.orig_shape
            
            # Register Image/Frame in KWCOCO
            image_entry = {
                "id": image_id_counter,
                "file_name": filename if not is_video else f"{filename}_frame_{frame_idx:06d}",
                "width": width,
                "height": height
            }
            
            if is_video:
                image_entry["video_id"] = current_vid_id
                image_entry["frame_index"] = frame_idx
                
            kwcoco_output["images"].append(image_entry)
            
            # Register Annotations
            for box in result.boxes:
                # YOLO outputs xyxy (top-left x, top-left y, bottom-right x, bottom-right y)
                # KWCOCO requires [top-left x, top-left y, width, height]
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                kwcoco_output["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": int(box.cls.cpu().numpy()[0]),
                    "bbox": coco_bbox,
                    "score": float(box.conf.cpu().numpy()[0])
                })
                annotation_id_counter += 1
                
            image_id_counter += 1
            
    # 5. Save the Output
    print(f"[MODEL] Writing KWCOCO results to {output_file_path}")
    with open(output_file_path, 'w') as f:
        json.dump(kwcoco_output, f, indent=4)
        
    # Cleanup temp directory holding custom weights
    temp_dir.cleanup()
    print("[MODEL] Inference complete!")
