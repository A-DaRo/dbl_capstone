# This script implements the Context-Aware Spatial Sampling strategy as detailed in
# Section 4 and the "Final Technical Implementation Guide" of project_specification.md.
# It has been refactored to be a high-performance, parallelized, and configurable
# library function that generates a dataset with flattened, remapped class masks.
import argparse
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import concurrent.futures
import os
from functools import partial
from tqdm import tqdm
from numba import jit
import json
import random
from typing import Optional, Dict, Tuple, List, Set
from .id2labels_labels2colors_coralscapes import get_coralscapes_mappings

@jit(nopython=True)
def poisson_disk_sampling(
    width: int, height: int, radius: int, foreground_mask: np.ndarray, k: int = 30
) -> List[Tuple[int, int]]:
    """
    Generates sample points using Bridson's Poisson Disk Sampling algorithm.
    This function is JIT-compiled with Numba for performance, as per Spec Section 4.3.
    """
    cell_size = radius / np.sqrt(2)
    grid_width, grid_height = int(np.ceil(width / cell_size)), int(np.ceil(height / cell_size))
    grid = [(-1, -1)] * (grid_width * grid_height) # Use a sentinel value for Numba
    
    foreground_coords = np.argwhere(foreground_mask)
    if len(foreground_coords) == 0:
        # Numba cannot infer the type of an empty list literal [].
        # We must explicitly create a list of the correct type (a list of tuples of ints)
        # and ensure it's empty. This is a common pattern to solve this issue.
        typed_empty_list = [(0, 0)]
        return typed_empty_list[:0]
        
    start_idx = random.randint(0, len(foreground_coords) - 1)
    p0_y, p0_x = foreground_coords[start_idx]
    p0 = (p0_x, p0_y)
    
    samples = [p0]
    active_list = [p0]
    grid_x, grid_y = int(p0[0] / cell_size), int(p0[1] / cell_size)
    grid[grid_x + grid_y * grid_width] = p0

    while len(active_list) > 0:
        idx = random.randint(0, len(active_list) - 1)
        p = active_list[idx]
        found = False
        for _ in range(k):
            angle = 2 * np.pi * random.random()
            dist = random.uniform(radius, 2 * radius)
            px_new = int(p[0] + dist * np.cos(angle))
            py_new = int(p[1] + dist * np.sin(angle))

            if not (0 <= px_new < width and 0 <= py_new < height and foreground_mask[py_new, px_new]):
                continue

            gx, gy = int(px_new / cell_size), int(py_new / cell_size)
            too_close = False
            for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
                for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                    s = grid[i + j * grid_width]
                    if s[0] != -1: # Check if cell is occupied
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

def _create_remapping_assets(
    task_definition_path: Optional[str],
    num_original_classes: int,
    original_id2label: Dict[int, str],
    original_label2color: Dict[str, Tuple[int, int, int]]
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, Tuple[int, int, int]]]:
    """
    Builds all necessary assets for remapping as defined in the "Final Technical Implementation Guide".
    This includes the lookup table (LUT) for flattening classes, new ID-to-label mappings,
    and a new label-to-color mapping with intelligent, non-colliding color assignment.
    """
    if task_definition_path is None:
        print("No task definition file provided. Using original labels and colors.")
        remapping_lut = np.arange(num_original_classes, dtype=np.uint8)
        return remapping_lut, {k:v for k,v in original_id2label.items()}, original_label2color

    print(f"Loading task definitions from: {task_definition_path}")
    with open(task_definition_path, 'r') as f:
        task_definitions = yaml.safe_load(f)

    remapping_lut = np.zeros(num_original_classes, dtype=np.uint8)
    new_id2label = {0: "unlabeled"}
    new_label2color = {"unlabeled": (0, 0, 0)}
    used_colors: Set[Tuple[int, int, int]] = {(0, 0, 0)}
    
    all_new_ids = {0}
    for task_group, details in task_definitions.items():
        for new_id_str, new_label_name in details.get('id2label', {}).items():
            new_id = int(new_id_str)
            if new_id == 0: continue
            if new_id in all_new_ids:
                raise ValueError(f"Duplicate new class ID '{new_id}' found in id2label section.")
            all_new_ids.add(new_id)
            new_id2label[new_id] = new_label_name

    for task_group, details in task_definitions.items():
        for new_id_str, old_ids in details.get('mapping', {}).items():
            new_id = int(new_id_str)
            new_label_name = new_id2label.get(new_id)
            if not new_label_name:
                raise ValueError(f"Mapping for new ID '{new_id}' has no corresponding label in id2label.")
            
            for old_id in old_ids:
                if old_id < num_original_classes:
                    remapping_lut[old_id] = new_id

            chosen_color = None
            for old_id in old_ids:
                original_label = original_id2label.get(old_id)
                if original_label:
                    candidate_color = original_label2color.get(original_label)
                    if candidate_color and candidate_color not in used_colors:
                        chosen_color = candidate_color
                        break
            
            if chosen_color is None:
                while True:
                    r, g, b = random.randint(30, 225), random.randint(30, 225), random.randint(30, 225)
                    random_color = (r, g, b)
                    if random_color not in used_colors:
                        chosen_color = random_color
                        print(f"Warning: Could not find a unique source color for '{new_label_name}'. Assigning random color {chosen_color}.")
                        break
            
            new_label2color[new_label_name] = chosen_color
            used_colors.add(chosen_color)

    print("Successfully created remapping assets.")
    return remapping_lut, new_id2label, new_label2color

def process_image(
    img_mask_path_tuple: Tuple[Path, Path],
    output_dir_str: str,
    patch_size: int,
    pds_radius: int,
    remapping_lut: np.ndarray
) -> int:
    """
    Worker function for parallel processing. It samples points, extracts patches,
    remaps labels using the provided LUT, and saves the results.
    """
    img_path, mask_path = img_mask_path_tuple
    output_dir = Path(output_dir_str)
    output_img_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"
    patch_radius = patch_size // 2
    
    try:
        image = Image.open(img_path).convert("RGB")
        raw_mask = np.array(Image.open(mask_path))
        
        # As per Spec Section 4.2.1, create a foreground mask to seed sampling.
        # This assumes coral classes are non-zero. A more robust implementation
        # would use the YAML config to define foreground classes.
        foreground_mask = (raw_mask != 0) 
        width, height = image.size
        
        sample_points = poisson_disk_sampling(width, height, pds_radius, foreground_mask)

        patch_count = 0
        for i, (x, y) in enumerate(sample_points):
            left, upper = x - patch_radius, y - patch_radius
            right, lower = x + patch_radius, y + patch_radius

            if not (left >= 0 and upper >= 0 and right <= width and lower <= height):
                continue
            
            img_patch = image.crop((left, upper, right, lower))
            mask_patch_raw = raw_mask[upper:lower, left:right]

            # This is the key step from the Final Guide: flatten the mask
            remapped_mask_patch = remapping_lut[mask_patch_raw]
            remapped_mask_img = Image.fromarray(remapped_mask_patch.astype(np.uint8))
            
            patch_filename = f"{img_path.stem.replace('_leftImg8bit','')}_patch_{i:04d}"
            img_patch.save(output_img_dir / f"{patch_filename}.png")
            # Save the single flattened mask as a PNG for easy visualization
            remapped_mask_img.save(output_mask_dir / f"{patch_filename}.png")
            patch_count += 1
            
        return patch_count
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        return 0

def create_pds_dataset(
    dataset_root: str,
    output_dir: str,
    patch_size: int = 512,
    pds_radius: int = 300,
    num_workers: Optional[int] = None,
    task_definition_path: Optional[str] = None,
    num_original_classes: int = 40
):
    """
    Creates a pre-processed patch dataset using Context-Aware Spatial Sampling.
    This is the main orchestrator function as per the Final Guide.
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    print(f"Starting dataset creation. Output will be saved to: {output_dir}")

    # 1. Setup paths and directories
    image_base_dir = dataset_root / "leftImg8bit" / "train"
    mask_base_dir = dataset_root / "gtFine" / "train"
    output_img_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # 2. Fetch original mappings to use as a color source
    original_id2label, original_label2color = get_coralscapes_mappings()
    if not original_id2label:
        raise RuntimeError("Failed to fetch original CoralScapes mappings. Aborting.")

    # 3. Generate remapping assets based on the YAML config
    remapping_lut, new_id2label, new_label2color = _create_remapping_assets(
        task_definition_path, num_original_classes, original_id2label, original_label2color
    )

    # 4. Save self-describing metadata to the output directory
    with open(output_dir / "id2label.json", "w") as f:
        json.dump({str(k): v for k, v in new_id2label.items()}, f, indent=4)
    with open(output_dir / "label2color.json", "w") as f:
        json.dump(new_label2color, f, indent=4)
    print(f"Saved new label and color mappings to {output_dir}")

    # 5. Gather all image/mask pairs for processing
    image_files = sorted(list(image_base_dir.glob("**/*_leftImg8bit.png")))
    img_mask_pairs = []
    for img_path in image_files:
        identifier = img_path.name.replace('_leftImg8bit.png', '')
        mask_path = mask_base_dir / img_path.relative_to(image_base_dir).parent / f"{identifier}_gtFine.png"
        if mask_path.exists():
            img_mask_pairs.append((img_path, mask_path))
    
    print(f"Found {len(img_mask_pairs)} image/mask pairs to process.")
    if not img_mask_pairs: return

    # 6. Run patch generation in parallel
    total_patches = 0
    worker_fn = partial(
        process_image,
        output_dir_str=str(output_dir),
        patch_size=patch_size,
        pds_radius=pds_radius,
        remapping_lut=remapping_lut
    )
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(img_mask_pairs), desc="Processing Images") as pbar:
            for patch_count in executor.map(worker_fn, img_mask_pairs):
                total_patches += patch_count
                pbar.update(1)
                pbar.set_postfix({"Total Patches": total_patches})
    
    print(f"\n--- Processing Complete ---")
    print(f"Generated {total_patches} patches.")
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a PDS-sampled patch dataset from the CoralScapes dataset."
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the root of the raw dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the new preprocessed patch dataset.")
    parser.add_argument("--task_definition_path", type=str, default=None, help="Path to a YAML file defining the class remapping.")
    parser.add_argument("--patch_size", type=int, default=512, help="Width and height of the square patches.")
    parser.add_argument("--pds_radius", type=int, default=300, help="Minimum distance between patch centers.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of CPU cores to use. Defaults to all available.")
    
    args = parser.parse_args()
    create_pds_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        pds_radius=args.pds_radius,
        num_workers=args.num_workers,
        task_definition_path=args.task_definition_path
    )