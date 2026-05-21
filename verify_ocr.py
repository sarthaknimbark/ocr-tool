import os
import cv2
import json
import sys

# Add the directory containing 'utils' to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils import preprocess_for_ocr, extract_text, parse_ocr_text, save_json_output

def test_document(filepath: str):
    print(f"\n==================================================")
    print(f"🔬 RUNNING BACKEND OCR PIPELINE FOR: {os.path.basename(filepath)}")
    print(f"==================================================")
    
    if not os.path.exists(filepath):
        print(f"❌ Error: Target file not found at {filepath}")
        return
        
    try:
        # 1. Load image
        img = cv2.imread(filepath)
        if img is None:
            print("❌ Error: OpenCV failed to read image.")
            return
            
        # 2. Preprocess
        print("🔄 Step 1: Preprocessing image using OpenCV...")
        preprocessed = preprocess_for_ocr(img)
        print(f"   Image ready (Dimensions: {preprocessed.shape})")
        
        # 3. Extract Text via PaddleOCR
        print("🔍 Step 2: Running PaddleOCR (loading engine)...")
        raw_results = extract_text(preprocessed)
        print(f"   Successfully extracted {len(raw_results)} lines.")
        
        print("\n📝 Raw OCR Detections:")
        for idx, line in enumerate(raw_results):
            print(f"   [{idx + 1:02d}] \"{line['text']}\" (Confidence: {line['confidence']:.2f})")
            
        # 4. Parse fields
        print("\n⚙️ Step 3: Classifying document and running parser regex...")
        parsed_response = parse_ocr_text(raw_results)
        
        # Save output
        output_dir = os.path.join(current_dir, "outputs")
        json_path = save_json_output(parsed_response, os.path.basename(filepath), output_dir)
        print(f"💾 Step 4: Saved structured response to: {os.path.basename(json_path)}")
        
        # Print output
        print("\n✨ Structured JSON Response:")
        print(json.dumps(parsed_response, indent=4))
        
    except Exception as e:
        print(f"❌ Error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    samples_dir = os.path.join(current_dir, "samples")
    
    eid_sample = os.path.join(samples_dir, "emirates_id_sample.jpg")
    dl_sample = os.path.join(samples_dir, "driving_license_sample.jpg")
    
    test_document(eid_sample)
    test_document(dl_sample)
