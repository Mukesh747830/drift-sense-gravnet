import sys
import os
import glob
import numpy as np
import cv2

def restore_image(img_array):
    """
    Baseline Image Restoration.
    Since no trained weights are available for this track, this uses traditional 
    computer vision techniques to restore the image, ensuring 100% compliance 
    with the hackathon constraints.
    """
    # 1. Handle NaNs and Infs
    if np.isnan(img_array).any() or np.isinf(img_array).any():
        img_array = np.nan_to_num(img_array, nan=np.nanmedian(img_array), posinf=1.0, neginf=0.0)

    # 2. Ensure it's a 2D grayscale array (H, W)
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
    elif img_array.ndim > 2:
        img_array = img_array[:, :, 0] # Fallback if multi-channel

    # 3. Normalize to [0, 1] range safely
    min_val = img_array.min()
    max_val = img_array.max()
    if max_val > min_val:
        img_array = (img_array - min_val) / (max_val - min_val)
    else:
        img_array = np.zeros_like(img_array)

    # 4. Apply a Median Filter to remove salt-and-pepper / random noise
    # OpenCV requires float32 for medianBlur
    img_array_f32 = img_array.astype(np.float32)
    restored = cv2.medianBlur(img_array_f32, 3)

    # 5. Ensure strict [0, 1] clipping just in case
    restored = np.clip(restored, 0.0, 1.0)
    
    return restored

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read all .npy files from the input directory
    input_files = glob.glob(os.path.join(input_dir, "*.npy"))
    if not input_files:
        print(f"Warning: No .npy files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(input_files)} .npy files. Starting restoration...")

    for file_path in input_files:
        filename = os.path.basename(file_path)
        
        try:
            # Load degraded image
            img_array = np.load(file_path)
            
            # Process and restore
            restored_array = restore_image(img_array)
            
            # Save restored image with identical filename
            output_path = os.path.join(output_dir, filename)
            np.save(output_path, restored_array)
            
            print(f"Successfully processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Restoration complete.")

if __name__ == "__main__":
    main()
