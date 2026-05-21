import os
import shutil
from PIL import Image

# Ensure directories exist
os.makedirs("D:/ocr-emirates-id/uae_ocr_project/assets", exist_ok=True)
os.makedirs("D:/ocr-emirates-id/uae_ocr_project/samples", exist_ok=True)
os.makedirs("D:/ocr-emirates-id/uae_ocr_project/uploads", exist_ok=True)
os.makedirs("D:/ocr-emirates-id/uae_ocr_project/outputs", exist_ok=True)

# Sources
logo_src = r"C:\Users\Admin\.gemini\antigravity\brain\2a4e8266-45fa-466f-91fc-3f1a32af2f05\ocr_logo_1779341483171.png"
eid_src = r"C:\Users\Admin\.gemini\antigravity\brain\2a4e8266-45fa-466f-91fc-3f1a32af2f05\emirates_id_mock_1779341763510.png"
dl_src = r"C:\Users\Admin\.gemini\antigravity\brain\2a4e8266-45fa-466f-91fc-3f1a32af2f05\driving_lic_mock_1779341782230.png"

# Destinations
logo_dst = "D:/ocr-emirates-id/uae_ocr_project/assets/logo.png"
eid_dst = "D:/ocr-emirates-id/uae_ocr_project/samples/emirates_id_sample.jpg"
dl_dst = "D:/ocr-emirates-id/uae_ocr_project/samples/driving_license_sample.jpg"

try:
    # Copy logo (keep as PNG)
    shutil.copy(logo_src, logo_dst)
    print("Successfully copied logo.png")
    
    # Load and save EID mockup as JPEG
    img_eid = Image.open(eid_src).convert("RGB")
    img_eid.save(eid_dst, "JPEG", quality=95)
    print("Successfully converted and saved emirates_id_sample.jpg")
    
    # Load and save DL mockup as JPEG
    img_dl = Image.open(dl_src).convert("RGB")
    img_dl.save(dl_dst, "JPEG", quality=95)
    print("Successfully converted and saved driving_license_sample.jpg")
except Exception as e:
    print(f"Error during copy/conversion: {str(e)}")
