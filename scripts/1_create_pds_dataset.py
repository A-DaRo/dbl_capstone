# Poisson Disk Sampling (PDS) for Patch Extraction in Coral-MTL
# This script creates a dataset of image patches sampled using Poisson Disk Sampling (PDS)

import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import random
from tqdm import tqdm

# Imports from your project files
from coral_mtl.data.dataset import TASK_DEFINITIONS, create_lookup_table

# --- Core Poisson Disk Sampling Algorithm (Unchanged) ---
def poisson_disk_sampling(
    width: int, height: int, radius: int, foreground_mask: np.ndarray, k: int = 30
) -> list[tuple[int, int]]:
    """Generates sample points using Bridson's Poisson Disk Sampling algorithm."""
    # (Implementation is unchanged)
    cell_size = radius / np.sqrt(2)
    grid_width, grid_height = int(np.ceil(width / cell_size)), int(np.ceil(height / cell_size))
    grid = [None] * (grid_width * grid_height)
    foreground_coords = np.argwhere(foreground_mask)
    if len(foreground_coords) == 0:
        print("Warning: Foreground mask is empty. No points will be sampled.")
        return []
    start_idx = random.randint(0, len(foreground_coords) - 1)
    p0 = (foreground_coords[start_idx][1], foreground_coords[start_idx][0])
    samples, active_list = [p0], [p0]
    grid_x, grid_y = int(p0[0] / cell_size), int(p0[1] / cell_size)
    grid[grid_x + grid_y * grid_width] = p0
    while active_list:
        idx = random.randrange(len(active_list))
        p = active_list[idx]; found = False
        for _ in range(k):
            angle, dist = 2 * np.pi * random.random(), random.uniform(radius, 2 * radius)
            px_new, py_new = int(p[0] + dist * np.cos(angle)), int(p[1] + dist * np.sin(angle))
            if not (0 <= px_new < width and 0 <= py_new < height and foreground_mask[py_new, py_new]): continue
            gx, gy = int(px_new / cell_size), int(py_new / cell_size); too_close = False
            for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
                for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                    s = grid[i + j * grid_width]
                    if s:
                        dx, dy = s[0] - px_new, s[1] - py_new
                        if dx**2 + dy**2 < radius**2: too_close = True; break
                if too_close: break
            if not too_close:
                p_new = (px_new, py_new); samples.append(p_new); active_list.append(p_new)
                grid[gx + gy * grid_width] = p_new; found = True; break
        if not found: active_list.pop(idx)
    return samples

# --- Helper function to create a lookup table for masks ---
def create_mask_lookup(mask_base_dir: Path) -> dict[str, Path]:
    """
    Scans the mask directory recursively and creates a dictionary mapping
    a unique image identifier to its full mask file path.
    """
    print(f"Building mask lookup table from: {mask_base_dir}")
    lookup = {}
    # --- [MODIFIED] Corrected the filename pattern to match your dataset ---
    for mask_path in mask_base_dir.glob("**/*_gtFine.png"):
        # The unique identifier is the filename without the suffix
        identifier = mask_path.name.replace('_gtFine.png', '')
        lookup[identifier] = mask_path
    print(f"Found {len(lookup)} mask files.")
    return lookup

# --- Main Processing Logic ---
def create_pds_dataset(args):
    """Main function to orchestrate the dataset creation process."""
    dataset_root = Path(args.dataset_root)
    image_base_dir = dataset_root / "leftImg8bit" / "train"
    mask_base_dir = dataset_root / "gtFine" / "train"
    
    output_dir = Path(args.output_dir)
    output_img_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    mask_lookup = create_mask_lookup(mask_base_dir)

    print(f"Recursively searching for training images in: {image_base_dir}")
    image_files = sorted(list(image_base_dir.glob("**/*.png")))
    print(f"Found {len(image_files)} training images to process.")

    if not image_files or not mask_lookup:
        print("\nError: Could not find images or masks. Please check the following:")
        print(f"  - Is --dataset_root ('{dataset_root}') correct?")
        print(f"  - Does '{image_base_dir}' contain image files?")
        print(f"  - Does '{mask_base_dir}' contain mask files ending in '_gtFine.png'?")
        return

    lookup_tables = {name: create_lookup_table(info["mapping"]) for name, info in TASK_DEFINITIONS.items()}
    total_patches, processed_images_count = 0, 0
    patch_radius = args.patch_size // 2

    for img_path in tqdm(image_files, desc="Processing Images"):
        image_identifier = img_path.name.replace('_leftImg8bit.png', '')
        mask_path = mask_lookup.get(image_identifier)
        
        if not mask_path:
            continue

        processed_images_count += 1
        image, raw_mask = Image.open(img_path).convert("RGB"), np.array(Image.open(mask_path))
        foreground_mask = (raw_mask != 0) & (raw_mask != 5)
        width, height = image.size
        sample_points = poisson_disk_sampling(width, height, args.pds_radius, foreground_mask)

        for i, (x, y) in enumerate(sample_points):
            left, upper = max(0, x - patch_radius), max(0, y - patch_radius)
            right, lower = min(width, x + patch_radius), min(height, y + patch_radius)
            if (right - left) != args.patch_size or (lower - upper) != args.patch_size: continue

            img_patch = image.crop((left, upper, right, lower))
            mask_patch_raw = raw_mask[upper:lower, left:right]
            task_masks = {name: table[mask_patch_raw] for name, table in lookup_tables.items()}

            patch_filename = f"{image_identifier}_patch_{i:04d}"
            img_patch.save(output_img_dir / f"{patch_filename}.png")
            np.savez_compressed(output_mask_dir / f"{patch_filename}.npz", **task_masks)
            total_patches += 1
            
    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed {processed_images_count} images with corresponding masks.")
    print(f"Generated {total_patches} patches.")
    print(f"Dataset saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a PDS-sampled patch dataset from the CoralScapes dataset."
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="Path to the root of the 'coralscapes' directory (which contains 'leftImg8bit' and 'gtFine')."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path where the new preprocessed patch dataset will be saved."
    )
    parser.add_argument(
        "--patch_size", type=int, default=512,
        help="Width and height of the square patches to extract."
    )
    # --- [THE FIX IS HERE] ---
    parser.add_argument(
        "--pds_radius", type=int, default=300, # Changed from --p_radius
        help="Minimum distance between the centers of any two patches."
    )
    
    args = parser.parse_args()
    # No changes needed here, as the variable name is based on the 'dest' which is inferred correctly.
    create_pds_dataset(args)