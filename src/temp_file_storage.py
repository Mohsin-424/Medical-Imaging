# temp_file_storage.py

import os
import tempfile
import shutil

def create_temp_file(content: bytes, suffix: str = '') -> str:
    """Create a temporary file and write content to it.
    
    Args:
        content (bytes): The content to write to the file.
        suffix (str): The file extension (e.g., '.pdf').
    
    Returns:
        str: The path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        return tmp_file.name

def clean_up_temp_files(temp_dir: str):
    """Remove temporary files in the specified directory.
    
    Args:
        temp_dir (str): The directory to clean up.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def create_temp_directory() -> str:
    """Create a temporary directory for storing files.
    
    Returns:
        str: The path to the temporary directory.
    """
    temp_dir = tempfile.mkdtemp()
    return temp_dir
