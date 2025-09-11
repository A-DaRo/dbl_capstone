from datasets import load_dataset

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

# 4. Access a single example from the training split
# You can index the dataset to get a specific data point.
first_example = train_dataset[0]
print("\n--- First Example from Training Set ---")
print(first_example)

# 5. Inspect the data
# The 'image' feature is a PIL Image object, and 'label' is an integer.
image = first_example['image']
label = first_example['label']

print(f"\nImage Mode: {image.mode}, Image Size: {image.size}")
print(f"Label: {label}")

# You can also inspect the class names for the labels
class_names = train_dataset.features['label'].names
print(f"The name for label {label} is: '{class_names[label]}'")