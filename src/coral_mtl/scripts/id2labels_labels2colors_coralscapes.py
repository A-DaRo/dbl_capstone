from datasets import load_dataset
import numpy as np
from huggingface_hub import hf_hub_download
import json
from PIL import Image, ImageDraw, ImageFont
import itertools

def get_coralscapes_mappings(repo_id="EPFL-ECEO/coralscapes"):
    """
    Fetches, loads, and processes id2label and label2color mappings for the CoralScapes dataset.

    This function downloads the necessary JSON mapping files from the Hugging Face Hub,
    converts string keys to integers for id2label, and ensures the 'unlabeled'
    class (ID 0) is correctly handled.

    Args:
        repo_id (str): The repository ID for the dataset on the Hugging Face Hub.

    Returns:
        tuple: A tuple containing:
            - dict: id2label mapping (int ID -> str label).
            - dict: label2color mapping (str label -> tuple(R, G, B) color).
    """
    print(f"--- Fetching mapping files from repository: {repo_id} ---")

    try:
        # Download and load the id-to-label mapping file
        id2label_path = hf_hub_download(repo_id=repo_id, filename="id2label.json", repo_type="dataset")
        with open(id2label_path, "r") as f:
            id2label_str_keys = json.load(f)
            id2label = {int(k): v for k, v in id2label_str_keys.items()}

        # Download and load the label-to-color mapping file
        label2color_path = hf_hub_download(repo_id=repo_id, filename="label2color.json", repo_type="dataset")
        with open(label2color_path, "r") as f:
            label2color_list_values = json.load(f)
            label2color = {k: tuple(v) for k, v in label2color_list_values.items()}

        # Manually add the mapping for ID 0 if it's missing
        if 0 not in id2label:
            id2label[0] = "unlabeled"
        if "unlabeled" not in label2color:
            label2color["unlabeled"] = (0, 0, 0)

        print("--- Mappings successfully loaded and processed. ---")
        return id2label, label2color

    except Exception as e:
        print(f"An error occurred while fetching or processing mappings: {e}")
        return None, None

def display_mappings(id2label, label2color, num_examples=5):
    """
    Prints a sample of the id2label and label2color mappings.

    Args:
        id2label (dict): Mapping from ID to label name.
        label2color (dict): Mapping from label name to color tuple.
        num_examples (int): The number of example entries to print.
    """
    if not id2label or not label2color:
        print("Mappings are not available.")
        return

    print("\n--- ID to Label Mapping (Sample) ---")
    for id_val, label_name in itertools.islice(id2label.items(), num_examples):
        print(f"ID {id_val}: {label_name}")

    print("\n--- Label to Color Mapping (Sample) ---")
    # Exclude 'unlabeled' from the sample if it exists, for a more informative sample
    filtered_labels = [lbl for lbl in label2color.keys() if lbl != "unlabeled"]
    for label_name in itertools.islice(filtered_labels, num_examples):
        print(f"Label '{label_name}': {label2color[label_name]}")

def decode_and_display_labels(label_mask, id2label, label2color):
    """
    Decodes a label mask to show which labels are present.

    Args:
        label_mask (np.ndarray): The segmentation mask with integer label IDs.
        id2label (dict): Mapping from ID to label name.
        label2color (dict): Mapping from label name to color tuple.
    """
    if not id2label or not label2color:
        print("Mappings are not available for decoding.")
        return

    unique_ids = np.unique(label_mask)
    print("\n--- Labels present in the mask ---")
    print(f"Unique label IDs found: {unique_ids}")
    for label_id in unique_ids:
        label_name = id2label.get(label_id, "Unknown ID")
        color = label2color.get(label_name, "N/A")
        print(f"ID: {label_id:<3} | Name: {label_name:<25} | Color: {color}")

def create_segmentation_legend(id2label, label2color, font_path=None, font_size=15):
    """
    Generates an image representing the legend for the segmentation classes.

    Args:
        id2label (dict): Mapping from ID to label name.
        label2color (dict): Mapping from label name to color tuple.
        font_path (str, optional): Path to a .ttf font file. Defaults to Pillow's basic font.
        font_size (int): The font size to use for labels.

    Returns:
        PIL.Image.Image: An image object containing the visual legend.
    """
    if not id2label or not label2color:
        print("Cannot create legend, mappings are not available.")
        return None

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except IOError:
        print(f"Warning: Font at {font_path} not found. Using default font.")
        font = ImageFont.load_default()

    box_size = 25
    padding = 10
    text_offset = 5
    
    # Sort items by ID, excluding 'unlabeled' if we don't want it in the legend
    sorted_labels = sorted([item for item in id2label.items() if item[1] != 'unlabeled'], key=lambda item: item[0])

    # Calculate image dimensions
    img_width = 400
    img_height = len(sorted_labels) * (box_size + padding) + padding

    legend_img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(legend_img)

    y_pos = padding
    for label_id, label_name in sorted_labels:
        color = label2color.get(label_name, (0, 0, 0))
        
        # Draw color box
        draw.rectangle([padding, y_pos, padding + box_size, y_pos + box_size], fill=color, outline="black")
        
        # Draw text
        text = f"{label_id}: {label_name}"
        draw.text((padding + box_size + text_offset, y_pos), text, fill="black", font=font)
        
        y_pos += box_size + padding
        
    return legend_img

# --- Main execution block for demonstration ---
if __name__ == "__main__":
    from datasets import load_dataset

    # 1. Get the mappings
    id2label, label2color = get_coralscapes_mappings()

    if id2label and label2color:
        # 2. Display a sample of the mappings
        display_mappings(id2label, label2color)

        # 3. Load a dataset example
        print("\n--- Loading a single example from the dataset (streaming) ---")
        try:
            dataset = load_dataset("EPFL-ECEO/coralscapes", split="train", streaming=True)
            first_example = next(iter(dataset))
            image = first_example["image"]
            label_mask = np.array(first_example["label"])

            print("Original Image size:", image.size)
            print("Label mask shape:", label_mask.shape)

            # 4. Decode and display the labels found in the example mask
            decode_and_display_labels(label_mask, id2label, label2color)

            # 5. Generate and display the segmentation legend
            print("\n--- Generating segmentation legend ---")
            legend = create_segmentation_legend(id2label, label2color)
            if legend:
                print("Legend created. Showing image...")
                legend.show(title="CoralScapes Segmentation Legend")
                # You can also save it:
                # legend.save("coralscapes_legend.png")

        except Exception as e:
            print(f"Failed to load or process dataset example: {e}")