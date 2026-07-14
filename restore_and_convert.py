import os
import zipfile
import shutil

def restore_and_convert():
    project_dir = r"d:\Final year project\Anomaly Detection in Jewellery.v1i.yolov8"
    zip_path = r"d:\Final year project\Anomaly Detection in Jewellery.v1i.yolov8.zip"
    
    if not os.path.exists(zip_path):
        print(f"Error: Zip file not found at {zip_path}")
        return False
        
    print(f"Opening zip file {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # We only want to extract the label files (txt)
        # Zip structure:
        # train/labels/xxx.txt, valid/labels/xxx.txt, test/labels/xxx.txt
        all_files = zip_ref.namelist()
        label_files = [f for f in all_files if f.endswith('.txt') and 'labels/' in f]
        
        print(f"Found {len(label_files)} label files in zip. Extracting and converting to bounding boxes...")
        
        converted_count = 0
        bbox_count = 0
        empty_count = 0
        
        for file_in_zip in label_files:
            # Read label content directly from zip
            with zip_ref.open(file_in_zip) as f:
                content = f.read().decode('utf-8')
                
            lines = content.strip().split('\n')
            new_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Check if it is a bounding box (5 elements: class x_center y_center w h)
                if len(parts) == 5:
                    # Clip coords
                    clipped = [max(0.0, min(1.0, x)) for x in coords]
                    new_lines.append(f"{class_id} " + " ".join([f"{x:.6f}" for x in clipped]) + "\n")
                    bbox_count += 1
                elif len(parts) > 5 and len(coords) % 2 == 0:
                    # Polygon format (segmentation)
                    x_coords = coords[0::2]
                    y_coords = coords[1::2]
                    
                    xmin = min(x_coords)
                    xmax = max(x_coords)
                    ymin = min(y_coords)
                    ymax = max(y_coords)
                    
                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    # Clip
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    converted_count += 1
                else:
                    print(f"Skipping unknown format line in {file_in_zip}: '{line}'")
            
            # Destination path inside workspace
            dest_path = os.path.join(project_dir, file_in_zip)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            if new_lines:
                with open(dest_path, 'w', encoding='utf-8') as df:
                    df.writelines(new_lines)
            else:
                # If label is empty, write an empty file (meaning background sample)
                with open(dest_path, 'w', encoding='utf-8') as df:
                    pass
                empty_count += 1
                
        print(f"Restoration and conversion summary:")
        print(f"  Total polygon labels converted to bounding boxes: {converted_count}")
        print(f"  Total bounding box labels verified: {bbox_count}")
        print(f"  Total empty label files: {empty_count}")
        return True

if __name__ == "__main__":
    restore_and_convert()
