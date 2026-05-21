import os
import sys
import cv2
import numpy as np
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile


# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import extract_text, parse_ocr_text, preprocess_for_ocr

app = FastAPI(title="UAE Document OCR API", version="1.0")

@app.post("/extract-text/")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """Extract text from uploaded document image"""
    try:
        # Read file bytes
        file_bytes = np.asarray(bytearray(await file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse({"error": "Invalid image file"}, status_code=400)
        
        # Process image
        processed = preprocess_for_ocr(image)
        ocr_result = extract_text(processed)
        parsed = parse_ocr_text(ocr_result)
        
        return {"success": True, "data": parsed}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)