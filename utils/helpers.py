import os
import re
import json
import uuid

def save_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> str:

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
    if not text:
        return ""
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', text)
    # Remove control characters and non-printable text
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
    return cleaned.strip()

def save_json_output(data: dict, filename: str, output_dir: str = "outputs") -> str:
    
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
    return json.dumps(data, indent=4, ensure_ascii=False)
