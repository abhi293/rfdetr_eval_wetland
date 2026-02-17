# This script processes bounding box and species information from CSV files to generate COCO-style annotations for object detection tasks.

import pandas as pd
import json
import ast
import os

def generate_coco_annotations(base_path):
    # Define absolute paths based on your directory listing
    bbox_csv = os.path.join(base_path, 'bounding_boxes.csv')
    species_csv = os.path.join(base_path, 'species_ID.csv')
    output_json = os.path.join(base_path, '_annotations.coco.json')

    # Verify files exist
    if not os.path.exists(bbox_csv) or not os.path.exists(species_csv):
        print(f"Error: Required files not found in {base_path}")
        return

    # Load species for categories [cite: 1]
    species_df = pd.read_csv(species_csv)
    
    # Load bounding boxes 
    # Note: Using sep=';' as per the file structure
    bbox_df = pd.read_csv(bbox_csv, sep=';')

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 1. Create Categories [cite: 1]
    for _, row in species_df.iterrows():
        coco_data["categories"].append({
            "id": int(row['id']),
            "name": row['species'],
            "supercategory": "bird"
        })

    image_id_map = {}
    image_counter = 1
    annotation_counter = 1

    print("Processing all frames... this may take a moment.")

    # 2. Process every frame (removes downsampling) 
    for _, row in bbox_df.iterrows():
        video_name = str(row['video_name'])
        frame_idx = int(row['frame'])
        species_id = int(row['species_id'])
           
        # Standard filename format: video_name_000000.jpg
        file_name = f"{video_name}_{frame_idx:06d}.jpg"
        
        if file_name not in image_id_map:
            image_id_map[file_name] = image_counter
            coco_data["images"].append({
                "id": image_counter,
                "file_name": file_name,
                "width": 1280,   # Adjust if your video resolution differs
                "height": 720    
            })
            image_counter += 1

        curr_image_id = image_id_map[file_name]

        # 3. Parse and convert bounding boxes 
        try:
            # bbox string format: [(x1, y1, x2, y2, bird_id, behavior_id)]
            bboxes = ast.literal_eval(row['bounding_boxes'])
            for bbox in bboxes:
                x1, y1, x2, y2, bird_id, behavior_id = bbox
                
                # COCO format: [xmin, ymin, width, height]
                width = x2 - x1
                height = y2 - y1
                
                coco_data["annotations"].append({
                    "id": annotation_counter,
                    "image_id": curr_image_id,
                    "category_id": species_id,
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(width), 2), round(float(height), 2)],
                    "area": round(float(width * height), 2),
                    "iscrowd": 0,
                    "attributes": {
                        "bird_id": bird_id,
                        "behavior_id": behavior_id
                    }
                })
                annotation_counter += 1
        except Exception:
            continue

    # 4. Save the updated JSON to the specified directory
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Success! Saved updated COCO annotations to: {output_json}")
    print(f"Total Images: {len(coco_data['images'])}")
    print(f"Total Annotations: {len(coco_data['annotations'])}")

if __name__ == "__main__":
    # Your verified working directory
    target_dir = r'D:\Projects\RF_DETR_Wetland\rf_detr'
    generate_coco_annotations(target_dir)