import os
import yaml
import hashlib
from PIL import Image

def calculate_md5(filepath):
    """Calculate the MD5 hash of a file to check for duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def validate_and_fix_dataset():
    dataset_path = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    report = []
    report.append("# Dataset Validation Report")
    report.append(f"**Dataset Location**: `{dataset_path}`\n")
    
    if not os.path.exists(yaml_path):
        report.append("## [CRITICAL ERROR] `data.yaml` is missing!")
        write_report(report)
        return False
        
    with open(yaml_path, 'r') as f:
        try:
            data_cfg = yaml.safe_load(f)
        except Exception as e:
            report.append(f"## [CRITICAL ERROR] Failed to parse `data.yaml`: {e}")
            write_report(report)
            return False
            
    report.append("## Configurations in `data.yaml`")
    for k, v in data_cfg.items():
        report.append(f"- **{k}**: {v}")
    report.append("")
    
    nc = data_cfg.get('nc', 0)
    names = data_cfg.get('names', [])
    if len(names) != nc:
        report.append(f"> [!WARNING]\n> `nc` ({nc}) does not match the number of classes listed in `names` ({len(names)}).")
        
    splits = ['train', 'val', 'test']
    image_hashes = {} # To detect duplicate images across splits or within a split
    
    for split in splits:
        split_key = 'val' if split == 'val' else split
        rel_path = data_cfg.get(split_key)
        if not rel_path:
            report.append(f"### [WARNING] Split '{split}' is not defined in `data.yaml`!")
            continue
            
        dataset_root = data_cfg.get('path', dataset_path)
        actual_split_dir = os.path.join(dataset_root, rel_path)
        split_root = os.path.dirname(os.path.normpath(actual_split_dir))
        
        images_dir = os.path.join(split_root, "images")
        labels_dir = os.path.join(split_root, "labels")
        
        report.append(f"### Verification of Split: `{split}`")
        report.append(f"- **Images folder**: `{images_dir}`")
        report.append(f"- **Labels folder**: `{labels_dir}`")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            report.append(f"- **[ERROR]** Image or Label directory is missing for split `{split}`!")
            continue
            
        image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))}
        label_files = {os.path.splitext(f)[0]: f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')}
        
        report.append(f"- Found **{len(image_files)}** images and **{len(label_files)}** label files.")
        
        # Check missing annotations (background images are allowed, but good to report)
        bg_images = [img_files for name, img_files in image_files.items() if name not in label_files]
        if bg_images:
            report.append(f"- Found **{len(bg_images)}** background images (no corresponding label files). This is normal for background negative samples.")
            
        # Check missing images (labels without images is a critical error)
        orphaned_labels = [lbl_files for name, lbl_files in label_files.items() if name not in image_files]
        if orphaned_labels:
            report.append(f"- **[ERROR]** Found **{len(orphaned_labels)}** label files with no corresponding image! (Will delete these orphaned label files to prevent training crashes).")
            # Safe fix: Delete orphaned label files
            for name in orphaned_labels:
                lbl_path = os.path.join(labels_dir, name)
                if os.path.exists(lbl_path):
                    os.remove(lbl_path)
            report.append("  - *Action taken*: Automatically deleted the orphaned label files.")
            
        # Integrity checks
        corrupted_images = []
        empty_labels = []
        invalid_bboxes = 0
        fixed_bboxes = 0
        class_mismatches = 0
        duplicate_images_count = 0
        
        for name, img_file in list(image_files.items()):
            img_path = os.path.join(images_dir, img_file)
            
            # Check Corruption
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                report.append(f"  - **[ERROR]** Corrupted image: `{img_file}` ({e}).")
                corrupted_images.append(img_file)
                # Remove from dictionary so we don't process annotations
                image_files.pop(name, None)
                continue
                
            # Check Duplicates using MD5 Hash
            try:
                img_hash = calculate_md5(img_path)
                if img_hash in image_hashes:
                    duplicate_images_count += 1
                    # Log duplicate but we don't delete to avoid breaking dataset count unless requested
                else:
                    image_hashes[img_hash] = (split, img_file)
            except Exception as e:
                pass
                
            # Validate Label annotations
            if name in label_files:
                lbl_path = os.path.join(labels_dir, label_files[name])
                if os.path.getsize(lbl_path) == 0:
                    empty_labels.append(label_files[name])
                    continue
                    
                try:
                    with open(lbl_path, 'r') as lf:
                        lines = lf.readlines()
                    
                    fixed_lines = []
                    modified = False
                    
                    for line_idx, line in enumerate(lines):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        
                        if len(parts) != 5:
                            report.append(f"  - **[ERROR]** Malformed label in `{label_files[name]}` line {line_idx+1}: '{line.strip()}' (deleted line).")
                            modified = True
                            continue
                            
                        try:
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]
                        except ValueError:
                            report.append(f"  - **[ERROR]** Non-numeric values in `{label_files[name]}` line {line_idx+1} (deleted line).")
                            modified = True
                            continue
                            
                        # Class ID mismatch check
                        if class_id < 0 or class_id >= nc:
                            report.append(f"  - **[ERROR]** Class mismatch in `{label_files[name]}` line {line_idx+1}: class `{class_id}` is invalid (dataset has {nc} classes).")
                            class_mismatches += 1
                            continue
                            
                        # Bounding Box Range check & auto-fix clipping
                        new_coords = []
                        line_fixed = False
                        for c_idx, val in enumerate(coords):
                            if val < 0.0:
                                if val >= -0.05: # Slight floating point inaccuracy, clip
                                    new_coords.append(0.0)
                                    line_fixed = True
                                else:
                                    invalid_bboxes += 1
                                    new_coords.append(val)
                            elif val > 1.0:
                                if val <= 1.05: # Slight floating point inaccuracy, clip
                                    new_coords.append(1.0)
                                    line_fixed = True
                                else:
                                    invalid_bboxes += 1
                                    new_coords.append(val)
                            else:
                                new_coords.append(val)
                                
                        # Bbox width and height check
                        w, h = new_coords[2], new_coords[3]
                        if w <= 0.0 or h <= 0.0:
                            report.append(f"  - **[ERROR]** Bounding box width/height <= 0 in `{label_files[name]}` line {line_idx+1} (deleted line).")
                            modified = True
                            continue
                            
                        if line_fixed:
                            fixed_bboxes += 1
                            modified = True
                            fixed_lines.append(f"{class_id} " + " ".join([f"{x:.6f}" for x in new_coords]) + "\n")
                        else:
                            fixed_lines.append(line)
                            
                    # Write fixes back to label file
                    if modified:
                        with open(lbl_path, 'w') as lf:
                            lf.writelines(fixed_lines)
                            
                except Exception as e:
                    report.append(f"  - **[ERROR]** Failed to read label file `{label_files[name]}`: {e}")
                    
        # Summary for Split
        report.append(f"- **Validation Summary for Split `{split}`**:")
        report.append(f"  - Corrupted Images: `{len(corrupted_images)}`")
        report.append(f"  - Empty Label Files: `{len(empty_labels)}`")
        report.append(f"  - Duplicate Images: `{duplicate_images_count}`")
        report.append(f"  - Invalid Bounding Boxes: `{invalid_bboxes}`")
        report.append(f"  - Class Mismatches: `{class_mismatches}`")
        report.append(f"  - Auto-fixed Bounding Boxes: `{fixed_bboxes}`")
        report.append("")
        
    write_report(report)
    print("Dataset validation completed. Report generated at 'dataset_validation_report.md'.")
    return True

def write_report(report):
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_validation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    validate_and_fix_dataset()
