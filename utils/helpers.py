import os
import json
import uuid
import re

def save_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> str:
    """
    Saves a Streamlit uploaded file to a specified directory with a unique UUID prefix
    to avoid name collisions.
    
    Args:
        uploaded_file: The Streamlit UploadedFile object.
        upload_dir (str): Relative or absolute path to the directory to save files in.
        
    Returns:
        str: Absolute or relative path to the saved file.
    """
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
        
    # Extract extension
    file_ext = os.path.splitext(uploaded_file.name)[1]
    # Create unique filename
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    
    # Save file
    file_path = os.path.join(upload_dir, unique_filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path

def clean_ocr_text(text: str) -> str:
    """
    Helper function to clean raw OCR text. Normalizes spaces, symbols, and line breaks.
    
    Args:
        text (str): Input text block.
        
    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', text)
    # Remove control characters and non-printable text
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
    return cleaned.strip()

def save_json_output(data: dict, filename: str, output_dir: str = "outputs") -> str:
    """
    Saves parsed JSON dict to the outputs directory.
    
    Args:
        data (dict): Parsed structured data dictionary.
        filename (str): Base name for the JSON output file.
        output_dir (str): Directory where JSON outputs are stored.
        
    Returns:
        str: Path where JSON is saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # If the file has an image extension, replace it with .json
    base_name = os.path.splitext(os.path.basename(filename))[0]
    json_filename = f"{base_name}_parsed.json"
    
    save_path = os.path.join(output_dir, json_filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    return save_path

def to_downloadable_json(data: dict) -> str:
    """
    Converts data dictionary into a pretty-printed JSON string suitable for downloads.
    
    Args:
        data (dict): Data dictionary.
        
    Returns:
        str: JSON formatted string.
    """
    return json.dumps(data, indent=4, ensure_ascii=False)
