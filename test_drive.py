from drive_manager import Drive_Manager
import google.auth.exceptions
import pandas as pd
import io
import random
from typing import Optional, Dict


def find_item_by_path(drive_manager: Drive_Manager, path: str) -> Optional[Dict]:
    """
    Traverses a path string to find a specific file or folder in Google Drive.
    
    Args:
        drive_manager: An initialized Drive_Manager instance.
        path: A string path like "folderA/folderB/file.txt".
        
    Returns:
        The Drive API item dictionary (with id, name, mimeType) or None if not found.
    """
    print(f"Searching for path: '{path}'")
    # Clean up path, removing leading/trailing slashes and splitting
    parts = [part for part in path.split('/') if part]
    
    current_id = drive_manager.root_id
    current_item = None
    
    for i, part in enumerate(parts):
        print(f"  -> Searching for '{part}'...")
        current_item = drive_manager.get_item_in_folder(part, current_id)
        
        if not current_item:
            print(f"Error: Could not find '{part}' in the path.")
            return None
        
        current_id = current_item.get('id')
        print(f"     Found '{part}' with ID: {current_id}")

        # If it's not the last part, it must be a folder to continue
        is_last_part = (i == len(parts) - 1)
        if not is_last_part and current_item.get('mimeType') != 'application/vnd.google-apps.folder':
            print(f"Error: '{part}' is a file, but the path continues. Invalid path.")
            return None
            
    return current_item


if __name__ == '__main__':
    # IMPORTANT: 
    # 1. Update ROOT_FOLDER_ID with your actual folder ID.
    # 2. Make sure to install pandas: pip install pandas
    ROOT_FOLDER_ID = "1mOuhlo0y-b65eo8QzlyUYLQMpwmvJYXF"
    
    if ROOT_FOLDER_ID == "1aBcDeFgHiJkLmNoPqRsTuVwXyZ_12345":
        print("CRITICAL: Please update the 'ROOT_FOLDER_ID' variable in the script with your folder's actual ID.")
    else:
        try:
            drive = Drive_Manager(root_folder_id=ROOT_FOLDER_ID)
            
            # --- Define the relative paths to the items we want to find ---
            images_dir_path = "benthic_datasets/point_labels/SEAVIEW/ATL"
            points_csv_path = "benthic_datasets/point_labels/SEAVIEW/tabular-data/annotations_ATL.csv"

            # --- Find the folder and file using our helper function ---
            images_folder_item = find_item_by_path(drive, images_dir_path)
            points_csv_item = find_item_by_path(drive, points_csv_path)

            # --- DEMO 1: Read a random image name and its content from the found folder ---
            if images_folder_item and images_folder_item.get('mimeType') == 'application/vnd.google-apps.folder':
                print("\n--- Demo 1: Reading a random image file into memory ---")
                image_folder_id = images_folder_item.get('id')
                
                # List files inside the specific folder we found
                images_in_dir = drive.list_files(folder_id=image_folder_id)

                if not images_in_dir:
                    print(f"No images found in the directory: '{images_dir_path}'")
                else:
                    random_image_meta = random.choice(images_in_dir)
                    random_image_id = random_image_meta.get('id')
                    random_image_name = random_image_meta.get('name')
                    
                    print(f"Selected random image: '{random_image_name}' (ID: {random_image_id})")
                    print("Reading its content into memory...")
                    
                    image_bytes = drive.read_file_content(random_image_id)
                    if image_bytes:
                        print(f"Successfully read {len(image_bytes)} bytes from '{random_image_name}'.")
                        # You could now use these bytes with a library like Pillow:
                        # from PIL import Image
                        # img = Image.open(io.BytesIO(image_bytes))
                        # img.show()
                    else:
                        print(f"Failed to read content for '{random_image_name}'.")
            else:
                print(f"\nCould not find or access the image directory at '{images_dir_path}'. Skipping Demo 1.")

            # --- DEMO 2: Stream the CSV file and load it as a Pandas DataFrame ---
            if points_csv_item and points_csv_item.get('mimeType') != 'application/vnd.google-apps.folder':
                print("\n--- Demo 2: Streaming a CSV and loading with Pandas ---")
                csv_file_id = points_csv_item.get('id')
                csv_file_name = points_csv_item.get('name')
                
                print(f"Streaming '{csv_file_name}' (ID: {csv_file_id})...")
                
                # The stream_file_content returns a generator of byte chunks
                stream_generator = drive.stream_file_content(csv_file_id)
                
                # We join the chunks together to form one complete bytes object
                csv_bytes = b''.join(stream_generator)
                
                print("Stream complete. Loading into Pandas DataFrame...")
                
                # Use io.BytesIO to treat the bytes object as a file for pandas
                df = pd.read_csv(io.BytesIO(csv_bytes))
                
                print(f"Successfully loaded '{csv_file_name}' into a DataFrame.")
                print("DataFrame Head:")
                print(df.head())
            else:
                print(f"\nCould not find or access the CSV file at '{points_csv_path}'. Skipping Demo 2.")

        except (ValueError, IOError, google.auth.exceptions.DefaultCredentialsError) as e:
            print(f"\nInitialization failed: {e}")
        except Exception as e:
            print(f"\nAn unexpected runtime error occurred: {e}")