import os
import sys
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

# Helper function to display document results
def _display_document_results(parsed_response, doc_name):
    """Display extraction results for a document"""
    # Success notifications
    if parsed_response["success"]:
        st.success(f"Success! Document identified as: **{parsed_response['document_type']}**")
    else:
        st.warning("OCR complete, but could not determine the exact document type. Showing raw text.")
    
    # Display structured information
    doc_type = parsed_response["document_type"]
    data_fields = parsed_response["data"]
    
    if doc_type == "Emirates ID" and parsed_response["success"]:
        st.markdown("###Emirates ID Information")
        
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
        st.markdown("###Driving License Information")
        
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
        st.warning(f" Document identification inconclusive (confidence: {parsed_response.get('detection_confidence', 0):.1%})\n\n" + 
                  "This image may not be an Emirates ID or Driving License. Please:\n" +
                  "• Ensure the document is clearly visible\n" +
                  "• Check image quality (not blurry or too small)\n" +
                  "• Upload a clear photo of an **Emirates ID** or **Driving License**\n\n" +
                  "Review the OCR text below to see what was detected from the image.")
    
    # JSON results viewer
    with st.expander(" Structured JSON Response Output", expanded=False):
        st.json(parsed_response)
    
    # Download JSON button
    json_str = to_downloadable_json(parsed_response)
    st.download_button(
        label=" Download Structured JSON",
        data=json_str,
        file_name=f"{os.path.splitext(doc_name)[0]}_extracted.json",
        mime="application/json",
        use_container_width=True
    )

def _combine_documents_response(all_results: list) -> dict:
    """Combine multiple document extraction results into a single response"""
    combined = {
        "success": all(r["success"] for r in all_results),
        "documents": all_results,
        "summary": {
            "total_documents": len(all_results),
            "documents_processed": sum(1 for r in all_results if r["success"]),
            "extraction_details": []
        }
    }
    
    # Build summary from all documents
    for result in all_results:
        summary_item = {
            "document_type": result["document_type"],
            "success": result["success"],
            "confidence": result["detection_confidence"],
            "data": result["data"]
        }
        combined["summary"]["extraction_details"].append(summary_item)
    
    return combined

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
st.sidebar.markdown("### System Control Panel")
# st.sidebar.info("Upload a document photo or choose a pre-loaded mock sample card to start processing.")

# Document Upload Type Selection
source_choice = st.sidebar.radio(
    "Select Document Source:",
    ["Upload Custom Document"]
)

uploaded_file = None
sample_selection = None

if source_choice == "Upload Custom Document":
    uploaded_files = st.sidebar.file_uploader(
        "Upload ID / License Images (Up to 2 documents)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload up to 2 documents (Emirates ID and/or Driving License). Supports JPG, JPEG, and PNG."
    )
    # Limit to 2 files
    if len(uploaded_files) > 2:
        st.sidebar.warning("Please upload maximum 2 documents")
        uploaded_files = uploaded_files[:2]

# Workspace Directories Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(script_dir, "uploads")
OUTPUT_DIR = os.path.join(script_dir, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Core logic to load and resolve target images
processed_documents = []

if source_choice == "Upload Custom Document" and uploaded_files:
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        saved_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
        image = cv2.imread(saved_path)
        processed_documents.append({
            "name": uploaded_file.name,
            "image": image,
            "path": saved_path
        })

# Display documents and extraction results
if processed_documents:
    # Show document previews and extract button
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"Loaded Documents ({len(processed_documents)} file{'s' if len(processed_documents) > 1 else ''})")
    
    preview_cols = st.columns(len(processed_documents))
    for idx, doc in enumerate(processed_documents):
        with preview_cols[idx]:
            rgb_preview = cv2.cvtColor(doc["image"], cv2.COLOR_BGR2RGB)
            st.image(rgb_preview, use_column_width=True, caption=doc["name"])
    
    # Extract button
    extract_button = st.button("Extract Details from All Documents", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process extraction if button clicked
    if extract_button:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("### Combined Extraction Results")
        
        all_extraction_results = []
        
        # Process all documents in parallel (sequential for now, can be parallelized)
        with st.spinner("Processing all documents..."):
            for doc in processed_documents:
                try:
                    preprocessed_img = preprocess_for_ocr(doc["image"])
                    raw_ocr_results = extract_text(preprocessed_img)
                    parsed_response = parse_ocr_text(raw_ocr_results)
                    saved_json_path = save_json_output(parsed_response, doc['name'], OUTPUT_DIR)
                    all_extraction_results.append(parsed_response)
                except Exception as ex:
                    st.error(f"Error processing {doc['name']}: {str(ex)}")
        
        # Combine all results into single response
        if all_extraction_results:
            combined_response = _combine_documents_response(all_extraction_results)
            
            # Display combined summary
            st.success(f"Successfully processed {combined_response['summary']['documents_processed']} out of {combined_response['summary']['total_documents']} document(s)")
            
            # Display extraction details for ALL documents in a SINGLE 4x2 table format (UNIQUE fields only)
            st.markdown("### Extraction Results")
            
            # Collect unique fields from all documents (no duplicates)
            all_fields = []
            seen_labels = set()
            for idx, detail in enumerate(combined_response["summary"]["extraction_details"], 1):
                doc_type = detail['document_type']
                if detail['data']:
                    for key, value in detail['data'].items():
                        if value:
                            label = key.replace('_', ' ').capitalize()
                            # Only add if we haven't seen this label before
                            if label not in seen_labels:
                                all_fields.append(f"{label}: {value}")
                                seen_labels.add(label)
            
            # Organize into 4 rows with 2 columns each (4x2 table)
            if all_fields:
                table_html = '<table style="width:100%; border-collapse: collapse; background: white;">'
                for i in range(0, len(all_fields), 2):
                    table_html += '<tr style="border-bottom: 1px solid #e0e0e0;">'
                    
                    # First column
                    table_html += f'<td style="padding: 12px; width: 50%; font-size: 0.95rem;">{all_fields[i]}</td>'
                    
                    # Second column (if exists)
                    if i + 1 < len(all_fields):
                        table_html += f'<td style="padding: 12px; width: 50%; font-size: 0.95rem;">{all_fields[i + 1]}</td>'
                    else:
                        table_html += '<td style="padding: 12px; width: 50%;"></td>'
                    
                    table_html += '</tr>'
                table_html += '</table>'
                st.markdown(table_html, unsafe_allow_html=True)
            
            # Display full JSON response
            st.markdown("### Complete JSON Response")
            with st.expander("View Full Response Structure", expanded=True):
                st.json(combined_response)
            
            # Download combined JSON
            combined_json_str = json.dumps(combined_response, indent=2)
            st.download_button(
                label="Download All Results (JSON)",
                data=combined_json_str,
                file_name="all_documents_extracted.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

