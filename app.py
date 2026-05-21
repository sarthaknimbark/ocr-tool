import os
import sys

os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode

import cv2
import json
import numpy as np
from PIL import Image
import streamlit as st

# Import package-level utility functions
from utils import (
    extract_text,
    parse_ocr_text,
    save_json_output,
    preprocess_for_ocr,
    save_uploaded_file,
    to_downloadable_json,
)

# Page configuration for a premium, wide layout
st.set_page_config(
    page_title="UAE Document OCR Extraction System",
    # page_icon="🇦🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern, premium CSS for a stunning look and feel
# We use custom HSL variables, clean typography (Inter), rounded cards, and gold/cyan accents
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* Elegant Title and Badges */
    .app-title {
        background: linear-gradient(135deg, #0284C7 0%, #0F766E 50%, #B45309 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 25px;
    }
    
    .badge {
        background: rgba(15, 118, 110, 0.1);
        color: #0F766E;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(15, 118, 110, 0.2);
    }
    
    /* Container/Card styles */
    .card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        border: 1px solid #F1F5F9;
        margin-bottom: 20px;
    }
    
    /* Status indicators */
    .field-label {
        font-weight: 600;
        color: #475569;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .field-value {
        font-size: 1.1rem;
        color: #0F172A;
        font-weight: 500;
        padding: 6px 0;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Layout: Main columns
col_header_left, col_header_right = st.columns([4, 1])

with col_header_left:
    st.markdown('<div class="app-title">UAE Document OCR Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Extract structured identity details from Emirates IDs and Driving Licenses in real-time. <span class="badge">PaddleOCR Powered</span></div>', unsafe_allow_html=True)

# with col_header_right:
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     logo_path = os.path.join(script_dir, "assets", "logo.png")
#     if os.path.exists(logo_path):
#         st.image(logo_path, width=120)

# Sidebar layout
st.sidebar.markdown("### 🇦🇪 System Control Panel")
# st.sidebar.info("Upload a document photo or choose a pre-loaded mock sample card to start processing.")

# Document Upload Type Selection
source_choice = st.sidebar.radio(
    "Select Document Source:",
    ["Upload Custom Document"]
)

uploaded_file = None
sample_selection = None

if source_choice == "Upload Custom Document":
    uploaded_file = st.sidebar.file_uploader(
        "Upload ID / License Image", 
        type=["jpg", "jpeg", "png",],
        help="Supports JPG, JPEG, and PNG images of Emirates ID or Driving License."
    )
# else:
#     sample_selection = st.sidebar.selectbox(
#         "Select a Sample Document:",
#         ["Emirates ID (Mock Sample)", "Driving License (Mock Sample)"]
#     )
#     st.sidebar.success("💡 Sample documents are pre-loaded to show full OCR capability without using real personal IDs.")

# # Sidebar Instructions & Tech Specs
# with st.sidebar.expander("🛠️ Technology Stack Specs", expanded=False):
#     st.markdown("""
#     - **Frontend:** Streamlit
#     - **OCR:** PaddleOCR (En)
#     - **Binarization:** OpenCV v4
#     - **Parsing:** Python Regular Expressions
#     - **Engine Core:** PaddlePaddle (CPU)
#     """)

# Workspace Directories Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(script_dir, "uploads")
OUTPUT_DIR = os.path.join(script_dir, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Core logic to load and resolve target image
image_to_process = None
image_name_label = ""

if source_choice == "Upload Custom Document" and uploaded_file is not None:
    # Save the file using the helper
    saved_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
    image_to_process = cv2.imread(saved_path)
    image_name_label = uploaded_file.name
    
# elif source_choice == "Use Pre-loaded Mock Samples":
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     if sample_selection == "Emirates ID (Mock Sample)":
#         sample_path = os.path.join(script_dir, "samples", "emirates_id_sample.jpg")
#     else:
#         sample_path = os.path.join(script_dir, "samples", "driving_license_sample.jpg")
        
#     if os.path.exists(sample_path):
#         image_to_process = cv2.imread(sample_path)
#         image_name_label = os.path.basename(sample_path)
#     else:
#         st.error(f"Sample asset not found at: {sample_path}")

# Main Layout split
if image_to_process is not None:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📷 Loaded Document Viewer")
        # Convert BGR image to RGB for displaying correctly in Streamlit
        rgb_preview = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
        st.image(rgb_preview, width='stretch', caption=f"Active Document: {image_name_label}")
        
        # Trigger Extraction Button
        extract_button = st.button("🚀 Extract Identity Details", type="primary", width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚡ Parsed Information Hub")
        
        if extract_button:
            # Step-by-step pipeline with loading spinner
            with st.spinner("Processing: Image optimization, running OCR engine & parsing text..."):
                try:
                    # 1. Preprocess the image
                    preprocessed_img = preprocess_for_ocr(image_to_process)
                    
                    # 2. Extract Text via PaddleOCR
                    raw_ocr_results = extract_text(preprocessed_img)
                    
                    # 3. Parse and structure the data
                    parsed_response = parse_ocr_text(raw_ocr_results)
                    
                    # 4. Save JSON response in outputs directory
                    saved_json_path = save_json_output(parsed_response, image_name_label, OUTPUT_DIR)
                    
                    # Success notifications
                    if parsed_response["success"]:
                        st.success(f"Success! Document identified as: **{parsed_response['document_type']}**")
                    else:
                        st.warning("OCR complete, but could not determine the exact document type. Showing raw text.")
                    
                    # 5. Display structured table/view of parsed information
                    doc_type = parsed_response["document_type"]
                    data_fields = parsed_response["data"]
                    
                    if doc_type == "Emirates ID" and parsed_response["success"]:
                        st.markdown("### 🇦🇪 Emirates ID Information")
                        
                        col_e1, col_e2 = st.columns(2)
                        with col_e1:
                            st.markdown(f'<div class="field-label">ID Number</div><div class="field-value">{data_fields.get("id_number") or "N/A"}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="field-label">Full Name</div><div class="field-value">{data_fields.get("name") or "N/A"}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="field-label">Nationality</div><div class="field-value">{data_fields.get("nationality") or "N/A"}</div>', unsafe_allow_html=True)
                        with col_e2:
                            st.markdown(f'<div class="field-label">Gender</div><div class="field-value">{data_fields.get("gender") or "N/A"}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="field-label">Date of Birth</div><div class="field-value">{data_fields.get("date_of_birth") or "N/A"}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="field-label">Expiry Date</div><div class="field-value">{data_fields.get("expiry_date") or "N/A"}</div>', unsafe_allow_html=True)
                            
                    elif doc_type == "Driving License" and parsed_response["success"]:
                        st.markdown("### 🚗 Driving License Information")
                        
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.markdown(f'<div class="field-label">License Number</div><div class="field-value">{data_fields.get("license_number") or "N/A"}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="field-label">Full Name</div><div class="field-value">{data_fields.get("name") or "N/A"}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="field-label">Nationality</div><div class="field-value">{data_fields.get("nationality") or "N/A"}</div>', unsafe_allow_html=True)
                        with col_d2:
                            st.markdown(f'<div class="field-label">Date of Birth</div><div class="field-value">{data_fields.get("date_of_birth") or "N/A"}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="field-label">Issue Date</div><div class="field-value">{data_fields.get("issue_date") or "N/A"}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="field-label">Expiry Date</div><div class="field-value">{data_fields.get("expiry_date") or "N/A"}</div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ Document identification inconclusive (confidence: {parsed_response.get('detection_confidence', 0):.1%})\n\n" + 
                                  "This image may not be an Emirates ID or Driving License. Please:\n" +
                                  "• Ensure the document is clearly visible\n" +
                                  "• Check image quality (not blurry or too small)\n" +
                                  "• Upload a clear photo of an **Emirates ID** or **Driving License**\n\n" +
                                  "Review the OCR text below to see what was detected from the image.")
                    
                    # 6. JSON results viewer
                    with st.expander("📦 Structured JSON Response Output", expanded=True):
                        st.json(parsed_response)
                        
                    # 7. Raw lines extracted (auto-open if detection failed)
                    # should_expand_raw = not parsed_response["success"]
                    # with st.expander("📝 Raw OCR Lines Extracted", expanded=should_expand_raw):
                    #     if parsed_response.get("raw_text"):
                    #         for i, text in enumerate(parsed_response["raw_text"], 1):
                    #             st.write(f"**Line {i}:** {text}")
                    #     else:
                    #         st.info("No text detected in the image")
                        
                    # 8. Download JSON button
                    json_str = to_downloadable_json(parsed_response)
                    st.download_button(
                        label="📥 Download Structured JSON",
                        data=json_str,
                        file_name=f"{os.path.splitext(image_name_label)[0]}_extracted.json",
                        mime="application/json",
                        width='stretch'
                    )
                    
                except Exception as ex:
                    st.error(f"An unexpected error occurred during processing: {str(ex)}")
                    st.exception(ex)
        else:
            st.info("Click the 'Extract Identity Details' button on the left to execute OCR & parsing.")
            
        st.markdown('</div>', unsafe_allow_html=True)
# else:
#     # If no file is loaded yet
#     st.warning("⚠️ No document active. Please select a sample card or upload your own file using the sidebar panel.")
    
    # Beautiful landing placeholder
    # st.markdown("""
    # <div style="background-color: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 16px; padding: 60px; text-align: center;">
    #     <h3 style="color: #64748B;">Ready to Extract UAE Documents</h3>
    #     <p style="color: #94A3B8; max-width: 500px; margin: 10px auto;">
    #         Our PaddleOCR-backed backend processes identity documents, cleans the visual layers, segments textual content, classifies headers, and formats matching key-value pairs into JSON schemas.
    #     </p>
    # </div>
    # """, unsafe_allow_html=True)
