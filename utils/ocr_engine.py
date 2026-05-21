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
            # Initialize PaddleOCR with required parameters
            cls._ocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False
            )
        return cls._ocr_instance

def extract_text(image_path_or_arr) -> list:
    """
    Performs OCR on an image and returns a list of extracted clean text lines.
    
    Args:
        image_path_or_arr: A file path (str) or preprocessed numpy image array (np.ndarray).
        
    Returns:
        list: A list of dicts containing text and confidence, e.g. [{"text": "...", "confidence": 0.98}]
    """
    try:
        ocr_engine = PaddleOCRSingleton.get_ocr_engine()
        
        # Run PaddleOCR on the image
        # If input is a path, PaddleOCR handles it. If input is a numpy array, it also handles it.
        result = ocr_engine.ocr(image_path_or_arr, cls=True)
        
        extracted_lines = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0].strip()
                confidence = float(line[1][1])
                if text:
                    extracted_lines.append({
                        "text": text,
                        "confidence": confidence
                    })
        
        return extracted_lines
    except Exception as e:
        print(f"Error during OCR extraction: {str(e)}")
        raise e
