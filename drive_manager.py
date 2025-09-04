import os
import io
import google.auth
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

    