from datasets import load_dataset
import numpy as np
from huggingface_hub import hf_hub_download
import json
from PIL import Image # Pillow is used for image objects
import itertools # Used for a robust demonstration loop

# --- Dynamic Mapping Creation ---
# Define the repository ID for the dataset on the Hugging Face Hub
repo_id = "EPFL-ECEO/coralscapes"

print(f"--- Fetching mapping files from repository: {repo_id} ---")

# Download and load the id-to-label mapping file
id2label_path = hf_hub_download(repo_id=repo_id, filename="id2label.json", repo_type="dataset")
with open(id2label_path, "r") as f:
    id2label_str_keys = json.load(f)
    # The keys in the JSON are strings, so we convert them to integers
    id2label = {int(k): v for k, v in id2label_str_keys.items()}

# Download and load the label-to-color mapping file
label2color_path = hf_hub_download(repo_id=repo_id, filename="label2color.json", repo_type="dataset")
with open(label2color_path, "r") as f:
    label2color_list_values = json.load(f)
    # Convert the color lists [R, G, B] to tuples (R, G, B)
    label2color = {k: tuple(v) for k, v in label2color_list_values.items()}

# --- FIX: Manually add the mapping for ID 0 ---
# The label IDs in the JSON start from 1. ID 0 is conventionally used for an
# "unlabeled" or "ignore" class.
id2label[0] = "unlabeled"
# We can assign a color for it, black is a common choice.
label2color["unlabeled"] = (0, 0, 0)


print("\n--- ID to Label Mapping (id2label) ---")
# A more robust way to print the first 5 items
for id_val, label_name in itertools.islice(id2label.items(), 5):
    print(f"ID {id_val}: {label_name}")


print("\n--- Label to Color Mapping (label2color) ---")
# Print a few examples
for label_name in list(label2color.keys())[1:6]: # Slicing to show the first few actual classes
    print(f"Label '{label_name}': {label2color[label_name]}")


# --- Original Code Integrated with Mappings ---

# 1. Load the dataset from the Hugging Face Hub
print("\n--- Loading a small part of the dataset (streaming mode) ---")
# Using streaming=True to avoid downloading the full 5.85 GB dataset
dataset = load_dataset("EPFL-ECEO/coralscapes", split="train", streaming=True)

# 2. Get the first example from the streaming dataset
print("\n--- First Example from Training Set ---")
first_example = next(iter(dataset))
label_mask = np.array(first_example["label"])  # convert to numpy for inspection
image = first_example["image"]

print("Image size:", image.size, "mode:", image.mode)

# 3. Use the mappings to interpret the label mask
unique_ids = np.unique(label_mask)
print("Label mask shape:", label_mask.shape)
print("Unique label IDs found in this mask:", unique_ids)

print("\n--- Labels present in the first example mask (decoded) ---")
for label_id in unique_ids:
    if label_id in id2label:
        label_name = id2label[label_id]
        color = label2color.get(label_name, "N/A") # Use .get() for safety
        print(f"ID: {label_id:<3} | Name: {label_name:<25} | Color: {color}")
    else:
        # This will catch any other unexpected IDs
        print(f"ID: {label_id:<3} | Name: Not Found in id2label map")