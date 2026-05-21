import re
import datetime

def clean_arabic_and_special_chars(text: str) -> str:
    """
    Cleans Arabic characters, special symbols, and extra whitespaces 
    to isolate English alphanumeric text.
    """
    if not text:
        return ""
    # Remove Arabic characters (Unicode range: U+0600 to U+06FF)
    text_no_arabic = re.sub(r'[\u0600-\u06FF]+', ' ', text)
    # Remove junk characters, keeping typical English alphanumeric and basic punctuation
    clean_text = re.sub(r'[^a-zA-Z0-9\s\-\/\:\.\,\(\)]', '', text_no_arabic)
    # Collapse multiple spaces into one
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def extract_dates(text: str) -> list:
    """
    Extracts all dates matching DD/MM/YYYY format from the text.
    """
    # Matches dates like 01/01/1990, 15-12-2025, 23.08.1994
    pattern = r'\b(\d{2}[/\-.]\d{2}[/\-.]\d{4})\b'
    matches = re.findall(pattern, text)
    
    # Normalize date separators to '/'
    normalized_dates = []
    for date_str in matches:
        normalized = date_str.replace('-', '/').replace('.', '/')
        normalized_dates.append(normalized)
        
    return normalized_dates

def detect_document_type(raw_lines: list) -> str:
    """
    Detects the document type based on key terms found in the raw text lines.
    
    Args:
        raw_lines (list): List of dictionaries/strings from OCR.
        
    Returns:
        str: "Emirates ID", "Driving License", or "Unknown"
    """
    combined_text = " ".join([line["text"].lower() for line in raw_lines])
    
    # Emirates ID keywords
    eid_keywords = [
        "emirates id", "identity card", "card number", "resident card", 
        "united arab emirates", "authority for identity"
    ]
    
    # Driving License keywords
    dl_keywords = [
        "driving license", "driver license", "driving lic", "lic. no", 
        "lic no", "traffic code", "license no"
    ]
    
    # Check for driving license first
    if any(keyword in combined_text for keyword in dl_keywords):
        return "Driving License"
        
    # Check for Emirates ID
    if any(keyword in combined_text for keyword in eid_keywords):
        return "Emirates ID"
        
    # Regex checks for EID number pattern
    eid_pattern = r'784[- ]?\d{4}[- ]?\d{7}[- ]?\d'
    if re.search(eid_pattern, combined_text):
        return "Emirates ID"
        
    # Default fallback
    return "Unknown"

def parse_emirates_id(raw_lines: list) -> dict:
    """
    Parses fields for a UAE Emirates ID.
    
    Expected output structure:
    {
      "document_type": "Emirates ID",
      "id_number": "",
      "name": "",
      "nationality": "",
      "gender": "",
      "date_of_birth": "",
      "expiry_date": ""
    }
    """
    data = {
        "document_type": "Emirates ID",
        "id_number": "",
        "name": "",
        "nationality": "",
        "gender": "",
        "date_of_birth": "",
        "expiry_date": ""
    }
    
    lines = [clean_arabic_and_special_chars(line["text"]) for line in raw_lines]
    # Filter out empty or whitespace-only lines
    lines = [l for l in lines if l]
    
    combined_text = " ".join(lines)
    
    # 1. Extract ID Number (784-YYYY-XXXXXXX-Z)
    id_pattern = r'\b(784[- ]?\d{4}[- ]?\d{7}[- ]?\d)\b'
    id_match = re.search(id_pattern, combined_text)
    if id_match:
        data["id_number"] = id_match.group(1).replace(" ", "-") # Standardize format with dashes
    else:
        # Fallback if hyphens are missing and it's a sequence of 15 digits
        fallback_pattern = r'\b(784\d{12})\b'
        fallback_match = re.search(fallback_pattern, combined_text)
        if fallback_match:
            raw_digits = fallback_match.group(1)
            data["id_number"] = f"{raw_digits[0:3]}-{raw_digits[3:7]}-{raw_digits[7:14]}-{raw_digits[14]}"
            
    # 2. Extract Dates (Birth and Expiry)
    all_dates = []
    for line in lines:
        all_dates.extend(extract_dates(line))
        
    # Remove duplicates but keep order
    seen = set()
    all_dates = [x for x in all_dates if not (x in seen or seen.add(x))]
    
    # Proximity matching for Date of Birth & Expiry Date
    dob_found = False
    expiry_found = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Find Date of Birth
        if "birth" in line_lower or "dob" in line_lower or "date of b" in line_lower:
            # Check same line first
            dates_on_line = extract_dates(line)
            if dates_on_line:
                data["date_of_birth"] = dates_on_line[0]
                dob_found = True
            # Check next line
            elif i + 1 < len(lines):
                dates_next_line = extract_dates(lines[i+1])
                if dates_next_line:
                    data["date_of_birth"] = dates_next_line[0]
                    dob_found = True
                    
        # Find Expiry Date
        if "expiry" in line_lower or "exp" in line_lower or "valid" in line_lower:
            dates_on_line = extract_dates(line)
            if dates_on_line:
                data["expiry_date"] = dates_on_line[0]
                expiry_found = True
            elif i + 1 < len(lines):
                dates_next_line = extract_dates(lines[i+1])
                if dates_next_line:
                    data["expiry_date"] = dates_next_line[0]
                    expiry_found = True

    # Smart fallback for Dates if proximity search failed
    if all_dates:
        # Sort dates chronologically to assign DOB (earliest) and Expiry (latest/future)
        try:
            parsed_dates = [datetime.datetime.strptime(d, "%d/%m/%Y") for d in all_dates]
            parsed_dates.sort()
            
            if not dob_found and parsed_dates:
                data["date_of_birth"] = parsed_dates[0].strftime("%d/%m/%Y")
            if not expiry_found and len(parsed_dates) > 1:
                # Typically the furthest date is the expiry date
                data["expiry_date"] = parsed_dates[-1].strftime("%d/%m/%Y")
        except Exception:
            # If date parsing fails, assign by order in text as a fallback
            if not dob_found and len(all_dates) >= 1:
                data["date_of_birth"] = all_dates[0]
            if not expiry_found and len(all_dates) >= 2:
                data["expiry_date"] = all_dates[1]
                
    # 3. Extract Name
    # EID names are usually uppercase lines. We check for keywords "Name" or "Name /"
    name_extracted = False
    for i, line in enumerate(lines):
        line_clean = re.sub(r'[^a-zA-Z\s\/]', '', line).strip()
        # Find where "Name" starts
        if any(keyword in line_clean.lower() for keyword in ["name", "full name", "nom"]):
            # Look at same line after the keyword "Name"
            # Remove the "Name" label
            name_candidate = re.sub(r'(?i)name|full|nom|[:/]', '', line_clean).strip()
            if len(name_candidate) > 4 and name_candidate.isupper():
                data["name"] = name_candidate
                name_extracted = True
                break
                
            # If not on same line, look at the next lines (usually 1st or 2nd line after "Name")
            for offset in [1, 2]:
                if i + offset < len(lines):
                    next_line = lines[i + offset].strip()
                    # Clean punctuation
                    next_line_clean = re.sub(r'[^a-zA-Z\s]', '', next_line).strip()
                    # Names are usually multiple words and UPPERCASE
                    if len(next_line_clean) > 5 and next_line_clean.isupper() and "NATIONALITY" not in next_line_clean and "IDENTITY" not in next_line_clean:
                        data["name"] = next_line_clean
                        name_extracted = True
                        break
            if name_extracted:
                break
                
    # Name Fallback: If not found through "Name" label, find the first multi-word all-caps line
    if not data["name"]:
        for line in lines:
            line_clean = re.sub(r'[^a-zA-Z\s]', '', line).strip()
            words = line_clean.split()
            if (len(words) >= 2 and 
                line_clean.isupper() and 
                not any(kwd in line_clean.lower() for kwd in ["authority", "identity", "united arab", "emirates", "card", "resident", "nationality", "sex"])):
                data["name"] = line_clean
                break

    # 4. Extract Nationality
    for i, line in enumerate(lines):
        if "nationality" in line.lower() or "national" in line.lower():
            # Check same line
            nat_candidate = re.sub(r'(?i)nationality|national|[:/]', '', line).strip()
            # Clean of digits/symbols
            nat_candidate = re.sub(r'[^a-zA-Z\s]', '', nat_candidate).strip()
            if len(nat_candidate) > 3 and not nat_candidate.lower() in ["card", "identity"]:
                data["nationality"] = nat_candidate
                break
            # Check next line
            elif i + 1 < len(lines):
                next_line_clean = re.sub(r'[^a-zA-Z\s]', '', lines[i+1]).strip()
                if len(next_line_clean) > 3 and not any(kwd in next_line_clean.lower() for kwd in ["sex", "gender", "expiry", "date"]):
                    data["nationality"] = next_line_clean
                    break
                    
    # 5. Extract Gender
    for line in lines:
        line_lower = line.lower()
        if "sex" in line_lower or "gender" in line_lower or "gnd" in line_lower:
            # Check for M or F
            if re.search(r'\b(M|Male|MALE)\b', line):
                data["gender"] = "Male"
                break
            elif re.search(r'\b(F|Female|FEMALE)\b', line):
                data["gender"] = "Female"
                break
        # Standalone gender terms
        if re.search(r'\b(Male|MALE)\b', line):
            data["gender"] = "Male"
            break
        elif re.search(r'\b(Female|FEMALE)\b', line):
            data["gender"] = "Female"
            break
            
    # Normalize gender
    if data["gender"] not in ["Male", "Female"]:
        # Standard default or empty
        data["gender"] = "Male" if "M" in combined_text else ("Female" if "F" in combined_text else "")

    return data

def parse_driving_license(raw_lines: list) -> dict:
    """
    Parses fields for a UAE Driving License.
    
    Expected output structure:
    {
      "document_type": "Driving License",
      "license_number": "",
      "name": "",
      "nationality": "",
      "date_of_birth": "",
      "issue_date": "",
      "expiry_date": ""
    }
    """
    data = {
        "document_type": "Driving License",
        "license_number": "",
        "name": "",
        "nationality": "",
        "date_of_birth": "",
        "issue_date": "",
        "expiry_date": ""
    }
    
    lines = [clean_arabic_and_special_chars(line["text"]) for line in raw_lines]
    lines = [l for l in lines if l]
    combined_text = " ".join(lines)
    
    # 1. Extract License Number
    # Typically listed next to "Lic. No." or "License No." or "No."
    license_pattern = r'(?i)(?:lic(?:ense)?\.?\s*no\.?|license|number)\s*[:\-\s]*([0-9]+)'
    license_match = re.search(license_pattern, combined_text)
    if license_match:
        data["license_number"] = license_match.group(1)
    else:
        # Fallback: search for a standalone digit sequence between 6 and 9 digits
        fallback_match = re.findall(r'\b(\d{7,9})\b', combined_text)
        if fallback_match:
            data["license_number"] = fallback_match[0]
            
    # 2. Extract Dates (Birth, Issue, Expiry)
    all_dates = []
    for line in lines:
        all_dates.extend(extract_dates(line))
        
    seen = set()
    all_dates = [x for x in all_dates if not (x in seen or seen.add(x))]
    
    dob_found = False
    issue_found = False
    expiry_found = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Find Date of Birth
        if "birth" in line_lower or "dob" in line_lower:
            dates = extract_dates(line)
            if dates:
                data["date_of_birth"] = dates[0]
                dob_found = True
            elif i + 1 < len(lines):
                dates_next = extract_dates(lines[i+1])
                if dates_next:
                    data["date_of_birth"] = dates_next[0]
                    dob_found = True
                    
        # Find Issue Date
        if "issue" in line_lower or "issued" in line_lower:
            dates = extract_dates(line)
            if dates:
                data["issue_date"] = dates[0]
                issue_found = True
            elif i + 1 < len(lines):
                dates_next = extract_dates(lines[i+1])
                if dates_next:
                    data["issue_date"] = dates_next[0]
                    issue_found = True
                    
        # Find Expiry Date
        if "expiry" in line_lower or "exp" in line_lower or "valid" in line_lower or "expires" in line_lower:
            dates = extract_dates(line)
            if dates:
                data["expiry_date"] = dates[0]
                expiry_found = True
            elif i + 1 < len(lines):
                dates_next = extract_dates(lines[i+1])
                if dates_next:
                    data["expiry_date"] = dates_next[0]
                    expiry_found = True

    # Chronological sort fallback for DL Dates (DOB is earliest, Issue is middle, Expiry is latest)
    if all_dates and (not dob_found or not issue_found or not expiry_found):
        try:
            parsed_dates = [datetime.datetime.strptime(d, "%d/%m/%Y") for d in all_dates]
            parsed_dates.sort()
            
            if not dob_found and parsed_dates:
                data["date_of_birth"] = parsed_dates[0].strftime("%d/%m/%Y")
            if not expiry_found and len(parsed_dates) > 1:
                data["expiry_date"] = parsed_dates[-1].strftime("%d/%m/%Y")
            if not issue_found:
                # Find a date in the middle
                middle_dates = [d for d in parsed_dates if d != parsed_dates[0] and d != parsed_dates[-1]]
                if middle_dates:
                    data["issue_date"] = middle_dates[0].strftime("%d/%m/%Y")
                elif len(parsed_dates) >= 3:
                    data["issue_date"] = parsed_dates[1].strftime("%d/%m/%Y")
        except Exception:
            if not dob_found and len(all_dates) >= 1:
                data["date_of_birth"] = all_dates[0]
            if not issue_found and len(all_dates) >= 2:
                data["issue_date"] = all_dates[1]
            if not expiry_found and len(all_dates) >= 3:
                data["expiry_date"] = all_dates[2]

    # 3. Extract Name
    name_extracted = False
    for i, line in enumerate(lines):
        line_clean = re.sub(r'[^a-zA-Z\s\/]', '', line).strip()
        if any(keyword in line_clean.lower() for keyword in ["name", "full name", "holder"]):
            name_candidate = re.sub(r'(?i)name|full|holder|[:/]', '', line_clean).strip()
            if len(name_candidate) > 4 and name_candidate.isupper():
                data["name"] = name_candidate
                name_extracted = True
                break
            
            for offset in [1, 2]:
                if i + offset < len(lines):
                    next_line_clean = re.sub(r'[^a-zA-Z\s]', '', lines[i + offset]).strip()
                    if len(next_line_clean) > 5 and next_line_clean.isupper() and "DRIVING" not in next_line_clean and "LICENSE" not in next_line_clean:
                        data["name"] = next_line_clean
                        name_extracted = True
                        break
            if name_extracted:
                break
                
    # Fallback name search
    if not data["name"]:
        for line in lines:
            line_clean = re.sub(r'[^a-zA-Z\s]', '', line).strip()
            words = line_clean.split()
            if (len(words) >= 2 and 
                line_clean.isupper() and 
                not any(kwd in line_clean.lower() for kwd in ["driving", "license", "united arab", "emirates", "authority", "national", "birth", "issue"])):
                data["name"] = line_clean
                break

    # 4. Extract Nationality
    for i, line in enumerate(lines):
        if "nationality" in line.lower() or "national" in line.lower():
            nat_candidate = re.sub(r'(?i)nationality|national|[:/]', '', line).strip()
            nat_candidate = re.sub(r'[^a-zA-Z\s]', '', nat_candidate).strip()
            if len(nat_candidate) > 3 and not nat_candidate.lower() in ["card", "license"]:
                data["nationality"] = nat_candidate
                break
            elif i + 1 < len(lines):
                next_line_clean = re.sub(r'[^a-zA-Z\s]', '', lines[i+1]).strip()
                if len(next_line_clean) > 3 and not any(kwd in next_line_clean.lower() for kwd in ["issue", "expiry", "date"]):
                    data["nationality"] = next_line_clean
                    break

    return data

def parse_ocr_text(raw_lines: list) -> dict:
    """
    Main entry point for parsing extracted OCR text.
    Detects document type and parses fields accordingly.
    
    Returns standard structured JSON format:
    {
      "success": bool,
      "document_type": str,
      "data": dict,
      "raw_text": list
    }
    """
    if not raw_lines:
        return {
            "success": False,
            "document_type": "Unknown",
            "data": {},
            "raw_text": []
        }
        
    doc_type = detect_document_type(raw_lines)
    raw_texts_only = [line["text"] for line in raw_lines]
    
    try:
        if doc_type == "Emirates ID":
            parsed_data = parse_emirates_id(raw_lines)
            success = True
        elif doc_type == "Driving License":
            parsed_data = parse_driving_license(raw_lines)
            success = True
        else:
            # We don't know the document type, return as empty but capture raw text
            parsed_data = {}
            success = False
            
        return {
            "success": success,
            "document_type": doc_type,
            "data": parsed_data,
            "raw_text": raw_texts_only
        }
        
    except Exception as e:
        print(f"Error during parsing raw OCR text: {str(e)}")
        return {
            "success": False,
            "document_type": doc_type,
            "data": {},
            "raw_text": raw_texts_only,
            "error": str(e)
        }
