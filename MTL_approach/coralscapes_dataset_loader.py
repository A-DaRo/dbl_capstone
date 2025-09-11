from time import time
from datasets import load_dataset
import numpy as np
# 1. Load the dataset from the Hugging Face Hub
# This creates a DatasetDict object containing all data splits.
dataset = load_dataset("EPFL-ECEO/coralscapes")

# 2. Explore the dataset structure
# This will print the structure, including splits (train, validation, test)
# and features ('image', 'label').
print("--- Dataset Structure ---")
print(dataset)

# 3. Access a specific split
# You can access each split of the data like a dictionary.
train_dataset = dataset["train"]
print("\n--- Train Split Info ---")
print(train_dataset)

print("\n--- First Example from Training Set ---")
first_example = train_dataset[0]
label_mask = np.array(first_example["label"])  # convert to numpy for inspection
image = first_example["image"]
print("Image:", image.size, image.mode)
print("Label mask shape:", label_mask.shape, "unique values:", np.unique(label_mask))


print(image)

