import os

# Disable oneDNN and MKLDNN CPU acceleration to prevent PIR framework attribute translation failures on Windows
# These must be set BEFORE importing PaddleOCR/PaddlePaddle
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only inference

import logging
import numpy as np
from paddleocr import PaddleOCR

# Suppress ppocr log spam to keep the console/terminal neat and clean
logging.getLogger("ppocr").setLevel(logging.ERROR)

class PaddleOCRSingleton:
    """
    A singleton class to handle the PaddleOCR instance initialization.
    Initializing PaddleOCR loads deep learning weights and takes time/memory, 
    so we ensure it only happens once.
    """
    _ocr_instance = None

    @classmethod
    def get_ocr_engine(cls) -> PaddleOCR:
        if cls._ocr_instance is None:
            # Initialize PaddleOCR with speed optimizations
            # Using only valid parameters for PaddleOCR v3.3+
            cls._ocr_instance = PaddleOCR(
                use_angle_cls=False,  # Disable angle detection for speed
                lang='en',
                enable_mkldnn=False
            )
        return cls._ocr_instance

def extract_text(image_path_or_arr) -> list:
    try:
        ocr_engine = PaddleOCRSingleton.get_ocr_engine()
        result = ocr_engine.ocr(image_path_or_arr)
        
        extracted_lines = []
        
        # Handle the NEW PaddleOCR v3.3+ result format
        # Result is a list of page results, each is a dict with 'rec_texts', 'rec_scores', etc.
        if result and isinstance(result, list):
            for page_idx, page_result in enumerate(result):
                if page_result and isinstance(page_result, dict):
                    # Extract recognized texts and scores from the dictionary
                    rec_texts = page_result.get('rec_texts', [])
                    rec_scores = page_result.get('rec_scores', [])
                    
                    # Pair texts with their confidence scores
                    if rec_texts:
                        for text_idx, text in enumerate(rec_texts):
                            try:
                                text_str = str(text).strip()
                                # Get corresponding score or default to 0.0
                                confidence = float(rec_scores[text_idx]) if text_idx < len(rec_scores) else 0.0
                                
                                # Only add non-empty text with reasonable confidence
                                if text_str and confidence > 0.1:
                                    extracted_lines.append({
                                        "text": text_str,
                                        "confidence": confidence
                                    })
                            except (IndexError, ValueError, TypeError) as e:
                                print(f"Warning: Could not parse text at index {text_idx}: {str(e)}")
                                continue
                elif page_result and isinstance(page_result, list):
                    # Handle LEGACY PaddleOCR format (lines as list of [[coords], (text, confidence)])
                    for line_idx, line in enumerate(page_result):
                        try:
                            if isinstance(line, (list, tuple)) and len(line) >= 2:
                                text_info = line[1]
                                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                    text = str(text_info[0]).strip()
                                    confidence = float(text_info[1])
                                    if text:
                                        extracted_lines.append({
                                            "text": text,
                                            "confidence": confidence
                                        })
                        except (IndexError, ValueError, TypeError) as e:
                            print(f"Warning: Could not parse legacy format line at {line_idx}: {str(e)}")
                            continue
        
        return extracted_lines
    except Exception as e:
        print(f"Error during OCR extraction: {str(e)}")
        raise e
