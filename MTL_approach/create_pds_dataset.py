# Poisson Disk Sampling (PDS) for Patch Extraction in Coral-MTL
# This script creates a dataset of image patches sampled using Poisson Disk Sampling (PDS)

import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import random
from tqdm import tqdm

# --- [MODIFIED] Simplified Imports ---
# Since this script now resides in the same directory as 'dataset.py',
# we can import directly from it without any path manipulation.
from dataset import TASK_DEFINITIONS, create_lookup_table

# --- 1. The Core Poisson Disk Sampling Algorithm (Unchanged) ---
# This is a pure function that implements Bridson's algorithm for fast PDS.

def poisson_disk_sampling(
    width: int,
    height: int,
    radius: int,
    foreground_mask: np.ndarray,
    k: int = 30
) -> list[tuple[int, int]]:
    """
    Generates sample points using Bridson's Poisson Disk Sampling algorithm,
    seeded only on the foreground areas of an image.
    """
    cell_size = radius / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = [None] * (grid_width * grid_height)

    foreground_coords = np.argwhere(foreground_mask)
    if len(foreground_coords) == 0:
        print("Warning: Foreground mask is empty. No points will be sampled.")
        return []

    # Choose a random starting point from the valid coordinates
    start_coord_idx = random.randint(0, len(foreground_coords) - 1)
    # Note: np.argwhere returns (row, col) which corresponds to (y, x)
    p0 = (foreground_coords[start_coord_idx][1], foreground_coords[start_coord_idx][0])

    samples = [p0]
    active_list = [p0]

    grid_x, grid_y = int(p0[0] / cell_size), int(p0[1] / cell_size)
    grid[grid_x + grid_y * grid_width] = p0

    while active_list:
        idx = random.randrange(len(active_list))
        p = active_list[idx]
        found = False

        for _ in range(k):
            angle = 2 * np.pi * random.random()
            dist = random.uniform(radius, 2 * radius)
            px_new, py_new = int(p[0] + dist * np.cos(angle)), int(p[1] + dist * np.sin(angle))

            # Continue if point is out of bounds or not in the foreground
            if not (0 <= px_new < width and 0 <= py_new < height and foreground_mask[py_new, px_new]):
                continue

            gx, gy = int(px_new / cell_size), int(py_new / cell_size)
            too_close = False
            # Check neighborhood in the grid
            for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
                for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                    s = grid[i + j * grid_width]
                    if s:
                        dx, dy = s[0] - px_new, s[1] - py_new
                        if dx**2 + dy**2 < radius**2:
                            too_close = True
                            break
                if too_close:
                    break

            if not too_close:
                p_new = (px_new, py_new)
                samples.append(p_new)
                active_list.append(p_new)
                grid[gx + gy * grid_width] = p_new
                found = True
                break

        if not found:
            active_list.pop(idx)

    return samples

# --- 2. Main Processing Logic ---

def create_pds_dataset(args):
    """
    Main function to orchestrate the dataset creation process.
    """
    raw_img_dir = Path(args.raw_image_dir)
    raw_mask_dir = Path(args.raw_mask_dir)
    output_dir = Path(args.output_dir)

    output_img_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for images in: {raw_img_dir}")
    image_files = sorted(list(raw_img_dir.glob("*.tif"))) # Assuming .tif format
    print(f"Found {len(image_files)} images.")

    # Use TASK_DEFINITIONS directly from your dataset.py
    # Create efficient lookup tables once
    lookup_tables = {
        task_name: create_lookup_table(task_info["mapping"])
        for task_name, task_info in TASK_DEFINITIONS.items()
    }

    total_patches = 0
    patch_radius = args.patch_size // 2

    for img_path in tqdm(image_files, desc="Processing Orthomosaics"):
        mask_path = raw_mask_dir / f"{img_path.stem}_label.png" # Adjust naming as needed
        if not mask_path.exists():
            print(f"Warning: Mask not found for {img_path.name}, skipping.")
            continue

        image = Image.open(img_path).convert("RGB")
        raw_mask = np.array(Image.open(mask_path))

        # --- Context-Aware Seeding ---
        # A foreground pixel is any pixel that is not 'unlabeled' (0) or 'sand' (5)
        # This prevents sampling from uninformative background areas.
        foreground_mask = (raw_mask != 0) & (raw_mask != 5)

        width, height = image.size
        sample_points = poisson_disk_sampling(width, height, args.pds_radius, foreground_mask)

        print(f"  - Generated {len(sample_points)} sample points for {img_path.name}")

        for i, (x, y) in enumerate(tqdm(sample_points, desc="  Extracting Patches", leave=False)):
            left, upper = max(0, x - patch_radius), max(0, y - patch_radius)
            right, lower = min(width, x + patch_radius), min(height, y + patch_radius)

            # Skip patches at the edges that are not the full size
            if (right - left) != args.patch_size or (lower - upper) != args.patch_size:
                continue

            img_patch = image.crop((left, upper, right, lower))
            mask_patch_raw = raw_mask[upper:lower, left:right]

            # Transform raw mask patch into multi-task masks using the imported logic
            task_masks = {
                task_name: table[mask_patch_raw]
                for task_name, table in lookup_tables.items()
            }

            patch_filename = f"{img_path.stem}_patch_{i:04d}"
            img_patch.save(output_img_dir / f"{patch_filename}.png")
            # Save all task masks into a single compressed .npz file for efficiency
            np.savez_compressed(output_mask_dir / f"{patch_filename}.npz", **task_masks)

            total_patches += 1

    print(f"\n--- Processing Complete ---")
    print(f"Total patches generated: {total_patches}")
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a PDS-sampled patch dataset for the Coral-MTL project."
    )
    parser.add_argument("--raw_image_dir", type=str, required=True, help="Path to high-resolution orthomosaic images.")
    parser.add_argument("--raw_mask_dir", type=str, required=True, help="Path to corresponding high-resolution raw label masks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path where preprocessed patches will be saved.")
    parser.add_argument("--patch_size", type=int, default=512, help="Width and height of square patches to extract.")
    parser.add_argument("--pds_radius", type=int, default=300, help="Minimum distance between patch centers.")

    args = parser.parse_args()
    create_pds_dataset(args)