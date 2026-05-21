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

def _normalize_for_detection(text: str) -> str:
    if not text:
        return ""
    normalized = text.lower().replace("<", " ").replace("|", " ")
    normalized = re.sub(r'[^a-z0-9\s\-\/\.:]', ' ', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()

def _unique_preserving_order(values: list) -> list:
    seen = set()
    unique_values = []
    for value in values:
        if value and value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values

def _extract_mrz_name(raw_text: str) -> str:
    if not raw_text or "<<" not in raw_text:
        return ""
    cleaned = re.sub(r'<+', ' ', raw_text)
    cleaned = re.sub(r'[^A-Za-z\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = cleaned.split()
    if len(words) >= 2:
        return " ".join(words[:4]).upper()
    return ""

def _extract_long_digit_sequence(text: str) -> str:
    digits_only = re.sub(r'\D', '', text)
    if len(digits_only) >= 15:
        return digits_only
    candidates = re.findall(r'\b\d{15,16}\b', text)
    if candidates:
        return max(candidates, key=len)
    return ""

def extract_dates(text: str) -> list:
    """
    Extracts all dates matching DD/MM/YYYY format from the text.
    Handles dates with or without spaces/separators.
    """
    # Match dates like 01/01/1990, 15-12-2025, 23.08.1994, and concatenated like "Birth04/10/1989"
    pattern = r'(\d{2}[/\-.]\d{2}[/\-.]\d{4})'
    matches = re.findall(pattern, text)
    
    # Normalize date separators to '/'
    normalized_dates = []
    for date_str in matches:
        normalized = date_str.replace('-', '/').replace('.', '/')
        if normalized not in normalized_dates:  # Avoid duplicates
            normalized_dates.append(normalized)
        
    return normalized_dates

def extract_date_after_label(text: str, label: str) -> str:
    """
    Extract date that appears right after a specific label.
    E.g., "Date of Birth04/10/1989" -> "04/10/1989"
    """
    # Look for label followed by optional spaces/punctuation, then a date
    pattern = f'{re.escape(label)}[\\s/:\\-.:,]*?(\\d{{2}}[/\\-.:]\\d{{2}}[/\\-.:]\\d{{4}})'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        date_str = match.group(1)
        return date_str.replace('-', '/').replace('.', '/').replace(':', '/')
    return ""

def classify_dates(dates: list, line_labels: list = None) -> dict:
    """
    Classifies dates into DOB, Issue, and Expiry based on context and proximity to labels.
    
    Args:
        dates: List of date strings (DD/MM/YYYY format)
        line_labels: List of lowercase line text for label matching
    
    Returns:
        dict with keys: 'date_of_birth', 'issue_date', 'expiry_date'
    """
    result = {
        'date_of_birth': '',
        'issue_date': '',
        'expiry_date': '',
        
        
    }
    
    if not dates:
        return result
    
    # Try to parse dates chronologically
    try:
        parsed = []
        for d in dates:
            try:
                dt = datetime.datetime.strptime(d, "%d/%m/%Y")
                parsed.append((d, dt))
            except:
                pass
        
        if not parsed:
            return result
            
        # Sort by datetime
        parsed.sort(key=lambda x: x[1])
        
        # Heuristic: DOB is oldest, Expiry is most recent
        if len(parsed) >= 1:
            result['date_of_birth'] = parsed[0][0]
        if len(parsed) >= 2:
            result['expiry_date'] = parsed[-1][0]
        if len(parsed) >= 3:
            # Issue is typically between DOB and Expiry
            result['issue_date'] = parsed[1][0]
        elif len(parsed) == 2:
            # Only 2 dates: DOB and Expiry, no issue date
            result['date_of_birth'] = parsed[0][0]
            result['expiry_date'] = parsed[1][0]
    except Exception:
        pass
    
    return result

def detect_document_type(raw_lines: list) -> tuple:
    """
    Detects the document type based on key terms found in the raw text lines.
    
    Args:
        raw_lines (list): List of dictionaries/strings from OCR.
        
    Returns:
        tuple: (document_type, confidence_score) where confidence is 0.0-1.0
    """
    normalized_lines = [_normalize_for_detection(line["text"]) for line in raw_lines if line.get("text")]
    combined_text = " ".join(normalized_lines)
    
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
    
    # Check for driving license first, including noisy OCR variants like "driving lie"
    if any(keyword in combined_text for keyword in dl_keywords) or re.search(r'\bdriving\b.{0,20}\b(?:lic(?:en(?:se)?)?|lie|licence)\b', combined_text):
        return ("Driving License", 0.95)
        
    # Check for Emirates ID
    if any(keyword in combined_text for keyword in eid_keywords) or re.search(r'\b(?:united arab emirates|resident identity card|emirates identity|identity card)\b', combined_text):
        return ("Emirates ID", 0.95)
        
    # Regex checks for EID number pattern
    eid_pattern = r'784[- ]?\d{4}[- ]?\d{7}[- ]?\d'
    if re.search(eid_pattern, combined_text):
        return ("Emirates ID", 0.90)

    # Fallbacks for heavily mangled OCR text
    if re.search(r'\bdriving\b', combined_text) and re.search(r'\b(?:lic|lie|no|issue|expiry)\b', combined_text):
        return ("Driving License", 0.78)

    if re.search(r'\b(?:resident|identity|emirates|united)\b', combined_text):
        return ("Emirates ID", 0.72)
        
    # Default fallback
    return ("Unknown", 0.0)

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
    
    raw_text_lines = [line.get("text", "") for line in raw_lines]
    lines = [clean_arabic_and_special_chars(text) for text in raw_text_lines]
    lines = [l for l in lines if l]
    
    combined_text = " ".join(lines)
    
    # 1. Extract ID Number (784-YYYY-XXXXXXX-Z)
    id_patterns = [
        r'\b(784[- ]?\d{4}[- ]?\d{7}[- ]?\d)\b',
        r'\b(78\d{13,14})\b',
        r'\b(7\d{14,15})\b',
    ]
    for pattern in id_patterns:
        id_match = re.search(pattern, combined_text)
        if id_match:
            raw_digits = re.sub(r'\D', '', id_match.group(1))
            if len(raw_digits) == 15:
                data["id_number"] = f"{raw_digits[0:3]}-{raw_digits[3:7]}-{raw_digits[7:14]}-{raw_digits[14]}"
            else:
                data["id_number"] = id_match.group(1).replace(" ", "-")
            break

    if not data["id_number"]:
        for raw_text in raw_text_lines:
            digits = _extract_long_digit_sequence(raw_text)
            if digits:
                data["id_number"] = digits
                break
            
    # 2. Extract Dates (Birth and Expiry)
    all_dates = []
    for line in lines:
        all_dates.extend(extract_dates(line))
        
    all_dates = _unique_preserving_order(all_dates)
    
    # Classify dates based on chronological order
    date_classification = classify_dates(all_dates)
    data["date_of_birth"] = date_classification.get('date_of_birth', '')
    data["expiry_date"] = date_classification.get('expiry_date', '')
    
    # Override with label-based matches if labels found
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Find Date of Birth
        if "birth" in line_lower or "dob" in line_lower or "date of b" in line_lower:
            dates_on_line = extract_dates(line)
            if dates_on_line:
                data["date_of_birth"] = dates_on_line[0]
            elif i + 1 < len(lines):
                dates_next_line = extract_dates(lines[i+1])
                if dates_next_line:
                    data["date_of_birth"] = dates_next_line[0]
                    
        # Find Expiry Date
        if "expiry" in line_lower or "exp" in line_lower or "valid" in line_lower:
            dates_on_line = extract_dates(line)
            if dates_on_line:
                data["expiry_date"] = dates_on_line[0]
            elif i + 1 < len(lines):
                dates_next_line = extract_dates(lines[i+1])
                if dates_next_line:
                    data["expiry_date"] = dates_next_line[0]
                
    # 3. Extract Name
    # EID names are usually after "Name:" or "Name" label
    name_extracted = False
    
    # First, look for "Name" label (with or without colon)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Check for "Name:", "Name", "Name /" patterns
        if re.search(r'\bname\s*[:/]?', line_lower):
            # Extract text after "Name" label (remove the label itself)
            name_candidate = re.sub(r'(?i)name\s*[:/]?', '', line).strip()
            name_candidate = re.sub(r'[^a-zA-Z\s]', '', name_candidate).strip()
            
            # If name is on same line after "Name:"
            if name_candidate and len(name_candidate) > 3:
                data["name"] = name_candidate
                name_extracted = True
                break
            
            # Otherwise check next line for the actual name
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                next_clean = re.sub(r'[^a-zA-Z\s]', '', next_line).strip()
                
                # Check if next line is a name (2+ words, UPPERCASE, not keywords)
                words = next_clean.split()
                if (len(words) >= 2 and 
                    next_clean.isupper() and 
                    not any(kwd in next_clean for kwd in ["NATIONALITY", "IDENTITY", "UNITED", "ARAB", "EMIRATES", "CARD", "RESIDENT"])):
                    data["name"] = next_clean
                    name_extracted = True
                    break
    
    # Fallback: Find uppercase multi-word lines that look like names
    if not data["name"]:
        for line in lines:
            line_clean = re.sub(r'[^a-zA-Z\s]', '', line).strip()
            words = line_clean.split()
            
            # Filter out keyword-only lines
            excluded_keywords = ["NATIONALITY", "IDENTITY", "UNITED", "ARAB", "EMIRATES", "CARD", "RESIDENT", 
                               "AUTHORITY", "DUBAI", "ABU", "DHABI", "SEX", "GENDER", "DATE", "BIRTH", "EXPIRY", "NUMBER"]
            
            is_name = (len(words) >= 2 and 
                      line_clean.isupper() and 
                      not any(kwd in line_clean for kwd in excluded_keywords))
            
            if is_name:
                data["name"] = line_clean
                break

    if not data["name"]:
        for raw_text in raw_text_lines:
            mrz_name = _extract_mrz_name(raw_text)
            if mrz_name:
                data["name"] = mrz_name
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
            trailing_match = re.search(r'([A-Z]{3,})\s*$', line)
            if trailing_match:
                data["nationality"] = trailing_match.group(1)
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
    
    raw_text_lines = [line.get("text", "") for line in raw_lines]
    lines = [clean_arabic_and_special_chars(text) for text in raw_text_lines]
    lines = [l for l in lines if l]
    combined_text = " ".join(lines)
    
    # 1. Extract License Number - More aggressive search
    for raw_text in raw_text_lines:
        code_match = re.search(r'\b([A-Z]{2,}\d{3,}|\d{3,}[A-Z]{1,}\d{2,})\b', raw_text)
        if code_match:
            data["license_number"] = code_match.group(1)
            break

    if not data["license_number"]:
        all_numbers = re.findall(r'\b(\d{6,10})\b', combined_text)
        if all_numbers:
            data["license_number"] = all_numbers[0]
            
    # 2. Extract Dates (Birth, Issue, Expiry)
    all_dates = []
    for line in lines:
        all_dates.extend(extract_dates(line))
        
    all_dates = _unique_preserving_order(all_dates)
    
    # Classify dates chronologically
    date_classification = classify_dates(all_dates)
    data["date_of_birth"] = date_classification.get('date_of_birth', '')
    data["issue_date"] = date_classification.get('issue_date', '')
    data["expiry_date"] = date_classification.get('expiry_date', '')
    
    # Override with label-based matches for better accuracy
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Find Date of Birth
        if "birth" in line_lower or "dob" in line_lower:
            dates = extract_dates(line)
            if dates:
                data["date_of_birth"] = dates[0]
            elif i + 1 < len(lines):
                dates_next = extract_dates(lines[i+1])
                if dates_next:
                    data["date_of_birth"] = dates_next[0]
                    
        # Find Issue Date
        if "issue" in line_lower or "issued" in line_lower:
            dates = extract_dates(line)
            if dates:
                data["issue_date"] = dates[0]
            elif i + 1 < len(lines):
                dates_next = extract_dates(lines[i+1])
                if dates_next:
                    data["issue_date"] = dates_next[0]
                    
        # Find Expiry Date
        if "expiry" in line_lower or "exp" in line_lower or "valid" in line_lower or "expires" in line_lower:
            dates = extract_dates(line)
            if dates:
                data["expiry_date"] = dates[0]
            elif i + 1 < len(lines):
                dates_next = extract_dates(lines[i+1])
                if dates_next:
                    data["expiry_date"] = dates_next[0]

    # 3. Extract Name - Check for "Name" or "Holder" label (with or without colon)
    name_extracted = False
    
    # Look for "Name" or "Holder" patterns (with or without colon)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Check for "Name:", "Name", "Holder:", "Holder" patterns
        if re.search(r'\b(name|holder)\s*[:/]?', line_lower):
            # Try to extract name from same line after the label
            name_candidate = re.sub(r'(?i)(name|holder)\s*[:/]?', '', line).strip()
            name_candidate = re.sub(r'[^a-zA-Z\s]', '', name_candidate).strip()
            
            if name_candidate and len(name_candidate) > 3:
                data["name"] = name_candidate
                name_extracted = True
                break
            
            # If not on same line, check next 2 lines for name
            for offset in [1, 2]:
                if i + offset < len(lines):
                    next_line_clean = re.sub(r'[^a-zA-Z\s]', '', lines[i + offset]).strip()
                    # Names are typically 2+ words, uppercase, not keywords
                    words = next_line_clean.split()
                    excluded = ["DRIVING", "LICENSE", "VALIDITY", "ISSUED", "DATE", "EXPIRY", "DOB"]
                    
                    if (len(words) >= 2 and 
                        next_line_clean.isupper() and 
                        not any(kwd in next_line_clean for kwd in excluded)):
                        data["name"] = next_line_clean
                        name_extracted = True
                        break
            if name_extracted:
                break
                
    # Fallback name search: Find first uppercase multi-word line that's not a keyword
    if not data["name"]:
        for line in lines:
            line_clean = re.sub(r'[^a-zA-Z\s]', '', line).strip()
            words = line_clean.split()
            if (len(words) >= 2 and 
                line_clean.isupper() and 
                not any(kwd in line_clean.lower() for kwd in ["driving", "license", "united arab", "emirates", "authority", "national", "birth", "issue"])):
                data["name"] = line_clean
                break

    if not data["name"]:
        for raw_text in raw_text_lines:
            mrz_name = _extract_mrz_name(raw_text)
            if mrz_name:
                data["name"] = mrz_name
                break

    # 4. Extract Nationality
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "nationality" in line_lower or "national" in line_lower or re.search(r'\bnati\w*', line_lower):
            nat_candidate = re.sub(r'(?i)nationality|national|[:/]', '', line).strip()
            nat_candidate = re.sub(r'(?i)nati\w*', '', nat_candidate).strip()
            nat_candidate = re.sub(r'[^a-zA-Z\s]', '', nat_candidate).strip()
            if len(nat_candidate) > 3 and not nat_candidate.lower() in ["card", "license"]:
                data["nationality"] = nat_candidate
                break
            trailing_match = re.search(r'([A-Z]{3,})\s*$', line)
            if trailing_match:
                data["nationality"] = trailing_match.group(1)
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
            "raw_text": [],
            "detection_confidence": 0.0
        }
        
    doc_type, confidence = detect_document_type(raw_lines)
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
            "raw_text": raw_texts_only,
            "detection_confidence": confidence
        }
        
    except Exception as e:
        print(f"Error during parsing raw OCR text: {str(e)}")
        return {
            "success": False,
            "document_type": doc_type,
            "data": {},
            "raw_text": raw_texts_only,
            "detection_confidence": confidence,
            "error": str(e)
        }
