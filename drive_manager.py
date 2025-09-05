import os
import io
import google.auth
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from typing import Dict, List, Iterator, Tuple, Optional

# This scope is sufficient for all read-only operations.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class Drive_Manager:
    """
    A class to manage read-only navigation and operations within a shared Google Drive folder.
    
    Designed to handle very large folders efficiently by using generators for content listing
    and providing methods for in-memory file reading and streaming.
    """

    def __init__(self, root_folder_id: str):
        """
        Initializes the Drive_Manager and authenticates with Google Drive API.

        Args:
            root_folder_id (str): The ID of the public "anyone with the link" folder
                                  that will serve as the root for navigation.
        
        Raises:
            ValueError: If the root_folder_id is inaccessible or not found.
            google.auth.exceptions.DefaultCredentialsError: If authentication fails.
        """
        self.service = self._initialize_service()
        self.root_id = root_folder_id
        
        print(f"Verifying access to root folder ID: {self.root_id}...")
        try:
            # Verify the service account can access the root folder and get its name
            root_folder_info = self.service.files().get(
                fileId=self.root_id, 
                fields='id, name',
                supportsAllDrives=True  # Important for Shared Drive compatibility
            ).execute()
            self.root_name = root_folder_info.get('name')
            print(f"Successfully accessed root folder: '{self.root_name}'")
        except HttpError as error:
            if error.resp.status == 404:
                raise ValueError(f"Error: The folder with ID '{self.root_id}' was not found.") from error
            else:
                raise IOError(f"An API error occurred while accessing the root folder: {error}") from error
        
        # Navigation state
        self.current_folder_id: str = self.root_id
        # Path stored as a list of (name, id) tuples
        self.current_path: List[Tuple[str, str]] = [(self.root_name, self.root_id)]

    def _initialize_service(self):
        """Handles authentication and builds the Drive API service object."""
        try:
            creds, _ = google.auth.default(scopes=SCOPES)
            service = build('drive', 'v3', credentials=creds)
            print("Authentication successful.")
            return service
        except google.auth.exceptions.DefaultCredentialsError as e:
            print("Authentication failed. Please check the 'GOOGLE_APPLICATION_CREDENTIALS' environment variable.")
            raise e
        except Exception as e:
            print(f"An unexpected authentication error occurred: {e}")
            raise e

    def _paginated_list_query(self, query: str) -> Iterator[Dict]:
        """A generator that handles pagination for a Drive API file list query."""
        page_token = None
        while True:
            try:
                response = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, size, modifiedTime)',
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    pageToken=page_token
                ).execute()
                
                for item in response.get('files', []):
                    yield item
                    
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
            except HttpError as error:
                print(f'An API error occurred during query execution: {error}')
                break

    def pwd(self) -> str:
        """Returns the current working directory path as a string."""
        return '/' + '/'.join([name for name, id in self.current_path])

    def change_directory(self, target: str) -> None:
        """Changes the current working directory ('..', '/', 'subdirectory_name')."""
        if target == '..':
            if len(self.current_path) > 1:
                self.current_path.pop()
                self.current_folder_id = self.current_path[-1][1]
                print(f"Moved up. Current path: {self.pwd()}")
            else:
                print("Already at the root directory.")
            return

        if target == '/':
            self.current_path = [self.current_path[0]]
            self.current_folder_id = self.root_id
            print(f"Returned to root. Current path: {self.pwd()}")
            return

        # Search for the target subdirectory within the current directory
        # Preprocess the target to escape single quotes
        escaped_target = target.replace("'", "\\'")
        query = (f"'{self.current_folder_id}' in parents and "
                f"name = '{escaped_target}' and "
                f"mimeType = 'application/vnd.google-apps.folder'")
        
        try:
            response = self.service.files().list(q=query, fields='files(id, name)', supportsAllDrives=True).execute()
            folders = response.get('files', [])
            if not folders:
                print(f"Error: Subdirectory '{target}' not found in '{self.pwd()}'.")
            else:
                folder = folders[0]
                self.current_folder_id = folder['id']
                self.current_path.append((folder['name'], folder['id']))
                print(f"Changed directory. Current path: {self.pwd()}")
        except HttpError as error:
            print(f'An API error occurred while changing directory: {error}')

    def iter_subdirectories(self, folder_id: Optional[str] = None) -> Iterator[Dict]:
        """Generator that yields all subdirectories in a given folder."""
        target_id = folder_id or self.current_folder_id
        query = f"'{target_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"
        yield from self._paginated_list_query(query)

    def iter_files(self, folder_id: Optional[str] = None) -> Iterator[Dict]:
        """Generator that yields all files (non-folders) in a given folder."""
        target_id = folder_id or self.current_folder_id
        query = f"'{target_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
        yield from self._paginated_list_query(query)

    def list_subdirectories(self, folder_id: Optional[str] = None) -> List[Dict]:
        """Returns a list of all subdirectories. Use with caution on folders with many items."""
        return list(self.iter_subdirectories(folder_id))

    def list_files(self, folder_id: Optional[str] = None) -> List[Dict]:
        """Returns a list of all files. Use with caution on folders with many items."""
        target_id = folder_id or self.current_folder_id
        query = f"'{target_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
        return list(self._paginated_list_query(query))
    
    def get_item_in_folder(self, name: str, folder_id: str) -> Optional[Dict]:
        """Finds a specific item (file or folder) by name inside a given folder."""
        escaped_name = name.replace("'", "\\'")
        query = (
            f"'{folder_id}' in parents and "
            f"name = '{escaped_name}' and "
            f"trashed = false"
        )
        try:
            response = self.service.files().list(
                q=query, fields='files(id, name, mimeType)', supportsAllDrives=True, pageSize=1
            ).execute()
            items = response.get('files', [])
            return items[0] if items else None
        except HttpError as error:
            print(f'API error while searching for "{name}": {error}')
            return None

    def find_item_by_path(self, path: str) -> Optional[Dict]:
        """
        Traverses a path string from the root to find a specific file or folder.
        
        Args:
            path: A string path like "folderA/folderB/file.txt". The path is
                  relative to the manager's root folder.
            
        Returns:
            The Drive API item dictionary (with id, name, mimeType) or None if not found.
        """
        print(f"Searching for path from root: '{path}'")
        # Clean up path, removing leading/trailing slashes and splitting
        parts = [part for part in path.split('/') if part]
        
        # Start the search from the manager's root folder
        current_id = self.root_id
        current_item = None
        
        for i, part in enumerate(parts):
            print(f"  -> Searching for '{part}'...")
            # Use the instance's own method to find the item
            current_item = self.get_item_in_folder(part, current_id)
            
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

    def read_file_content(self, file_id: str) -> Optional[bytes]:
        """
        Reads the entire content of a file into memory.
        Ideal for small to medium-sized files.

        Args:
            file_id (str): The ID of the file to read.

        Returns:
            Optional[bytes]: The file content as a bytes object, or None if an error occurs.
        """
        print(f"Reading content of file ID: {file_id}")
        try:
            request = self.service.files().get_media(fileId=file_id)
            # Use an in-memory binary stream
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download progress: {int(status.progress() * 100)}%")

            print("Content read successfully.")
            return fh.getvalue()
        except HttpError as error:
            print(f'An API error occurred while reading file content: {error}')
            return None
        except Exception as e:
            print(f"An unexpected error occurred during content read: {e}")
            return None


    def stream_file_content(self, file_id: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        """
        Yields the content of a file in chunks.
        This is memory-efficient and suitable for very large files.

        Args:
            file_id (str): The ID of the file to stream.
            chunk_size (int): The size of each chunk in bytes.

        Yields:
            Iterator[bytes]: A generator that yields chunks of the file content.
        """
        print(f"Streaming content of file ID: {file_id} in {chunk_size} byte chunks")
        try:
            request = self.service.files().get_media(fileId=file_id)
            # Use an in-memory binary stream as a buffer
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
            
            done = False
            while not done:
                # next_chunk() downloads a chunk and writes it to fh
                status, done = downloader.next_chunk()
                # Rewind the buffer to the beginning to read what was just downloaded
                fh.seek(0)
                # Yield the chunk
                yield fh.read()
                
                # Clear the buffer for the next chunk
                fh.seek(0)
                fh.truncate(0)

        except HttpError as error:
            print(f'An API error occurred while streaming file content: {error}')
            # The generator will simply stop if an error occurs
        except Exception as e:
            print(f"An unexpected error occurred during content stream: {e}")

    def download_file(self, file_id: str, destination_folder: str) -> None:
        """Downloads a file from Google Drive to the local filesystem."""
        try:
            file_metadata = self.service.files().get(fileId=file_id, fields='name', supportsAllDrives=True).execute()
            file_name = file_metadata.get('name')
            destination_path = os.path.join(destination_folder, file_name)

            print(f"Downloading '{file_name}' (ID: {file_id}) to '{destination_path}'...")
            os.makedirs(destination_folder, exist_ok=True)

            request = self.service.files().get_media(fileId=file_id)
            fh = io.FileIO(destination_path, 'wb')
            
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download progress: {int(status.progress() * 100)}%")
            print("Download complete.")

        except HttpError as error:
            print(f'An API error occurred during download: {error}')
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")

    def download_directory(self, local_destination_path: str, drive_folder_id: Optional[str] = None, max_workers: int = 8, overwrite: bool = False) -> None:
        """
        Downloads the entire content of a Google Drive folder using multiple concurrent workers.

        This method first scans the entire directory structure, then uses a thread pool
        to download files concurrently, displaying a single progress bar for the operation.

        Args:
            local_destination_path (str): The path to the local directory where the
                                          content will be saved. It will be created
                                          if it doesn't exist.
            drive_folder_id (Optional[str], optional): The ID of the Drive folder to download.
                                                       If None, the current working directory
                                                       of the manager is used. Defaults to None.
            max_workers (int, optional): The maximum number of concurrent download threads.
                                         Defaults to 8.
            overwrite (bool, optional): If False (default), files that already exist in the
                                        destination path will be skipped. If True, existing
                                        files will be overwritten.
        """
        target_folder_id = drive_folder_id or self.current_folder_id

        try:
            # --- PASS 1: Scan the directory to get a complete list of all files ---
            print("Scanning directory structure...")
            all_files, _ = self._get_directory_contents_recursive(target_folder_id)

            if not all_files:
                print("No files found in the specified Drive directory.")
                return
            
            # --- FILTERING STEP: Decide which files actually need to be downloaded ---
            files_to_download = []
            if not overwrite:
                print("Overwrite is False. Checking for existing files to skip...")
                for f in all_files:
                    destination_path = os.path.join(local_destination_path, f['local_path'])
                    if not os.path.exists(destination_path):
                        files_to_download.append(f)
                
                skipped_count = len(all_files) - len(files_to_download)
                if skipped_count > 0:
                    print(f"Skipped {skipped_count} files that already exist locally.")
            else:
                print("Overwrite is True. All files will be downloaded.")
                files_to_download = all_files
            
            # --- Check if there's anything left to do ---
            if not files_to_download:
                print("All files already exist locally. Nothing to download.")
                return

            # --- PREPARATION for PASS 2: Calculate size and create directories ---
            total_size_to_download = sum(f['size'] for f in files_to_download)
            
            print(f"Starting download of {len(files_to_download)} files (Total size: {total_size_to_download / (1024*1024):.2f} MB) using up to {max_workers} workers.")
            print(f"Destination: '{os.path.abspath(local_destination_path)}'")

            # --- PASS 2: Download files concurrently with a single progress bar ---
            with tqdm(total=total_size_to_download, unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading") as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Create all local directories beforehand to prevent race conditions.
                    # This is based on the filtered list of files to download.
                    all_local_dirs = {os.path.dirname(os.path.join(local_destination_path, f['local_path'])) for f in files_to_download}
                    for d in all_local_dirs:
                        os.makedirs(d, exist_ok=True)
                    
                    # Submit only the filtered download tasks to the thread pool
                    futures = [
                        executor.submit(
                            self._download_file_and_update_progress,
                            file_info['id'],
                            os.path.join(local_destination_path, file_info['local_path']),
                            pbar
                        )
                        for file_info in files_to_download
                    ]
                    # The 'with' block implicitly waits for all futures to complete.

            print("\nDirectory download completed successfully.")

        except HttpError as error:
            print(f'An API error occurred: Could not process folder ID {target_folder_id}. {error}')
        except Exception as e:
            print(f"An unexpected error occurred during directory download: {e}")

    def _get_directory_contents_recursive(self, folder_id: str, current_path: str = "") -> Tuple[List[Dict], int]:
        """
        Recursively scans a Drive folder to get a flat list of all files and the total size.
        This remains a sequential operation to safely build the file list.
        """
        all_files = []
        total_size = 0

        # Use a list to hold iterators to avoid exhausting them prematurely
        iterators = [self.iter_files(folder_id), self.iter_subdirectories(folder_id)]
        
        # Get files in the current directory
        for file_item in iterators[0]:
            file_size = int(file_item.get('size', 0))
            all_files.append({
                'id': file_item['id'],
                'name': file_item['name'],
                'size': file_size,
                'local_path': os.path.join(current_path, file_item['name'])
            })
            total_size += file_size

        # Recurse into subdirectories
        for subdir_item in iterators[1]:
            new_path = os.path.join(current_path, subdir_item['name'])
            sub_files, sub_size = self._get_directory_contents_recursive(subdir_item['id'], new_path)
            all_files.extend(sub_files)
            total_size += sub_size
            
        return all_files, total_size

    def _download_file_and_update_progress(self, file_id: str, destination_path: str, pbar: tqdm) -> None:
        """
        Downloads a single file and updates the provided thread-safe tqdm progress bar.
        This function is designed to be called from multiple threads.
        """
        try:
            # Each thread needs its own authorized service instance to be thread-safe.
            # Building it is cheap as credentials are cached.
            service = self._initialize_service()
            request = service.files().get_media(fileId=file_id)

            with io.FileIO(destination_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    last_pos = fh.tell()
                    _, done = downloader.next_chunk()
                    bytes_downloaded = fh.tell() - last_pos
                    pbar.update(bytes_downloaded)

        except HttpError as error:
            pbar.write(f"Warning: API error downloading file ID {file_id} to '{destination_path}'. Error: {error}")
        except Exception as e:
            pbar.write(f"Warning: Unexpected error downloading file ID {file_id}. Error: {e}")