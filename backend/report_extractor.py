"""
Medical Report Extraction & Parsing Engine
Extracts clinical parameters, units, reference ranges, and flags from medical lab reports (PDF/images).
"""
import re
import io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from pypdf import PdfReader
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

PARAMETER_DEFINITIONS = {
    "age": {
        "name": "Age",
        "aliases": [r"\bage\b", r"\bpatient\s*age\b", r"\byears\s*old\b"],
        "unit": "years",
        "min": 0, "max": 120,
        "default_ref": "18 - 85 years",
        "category": "demographic"
    },
    "gender": {
        "name": "Gender",
        "aliases": [r"\bgender\b", r"\bsex\b"],
        "unit": "category",
        "min": 0, "max": 1,
        "default_ref": "M / F",
        "category": "demographic"
    },
    "bmi": {
        "name": "Body Mass Index (BMI)",
        "aliases": [r"\bbmi\b", r"\bbody\s*mass\s*index\b"],
        "unit": "kg/m²",
        "min": 10.0, "max": 70.0,
        "default_ref": "18.5 - 24.9 kg/m²",
        "category": "anthropometric"
    },
    "alt": {
        "name": "ALT (Alanine Aminotransferase)",
        "aliases": [r"\balt\b", r"\bsgpt\b", r"\balanine\s*aminotransferase\b"],
        "unit": "U/L",
        "min": 0.0, "max": 1000.0,
        "default_ref": "7 - 56 U/L",
        "category": "enzymes"
    },
    "ast": {
        "name": "AST (Aspartate Aminotransferase)",
        "aliases": [r"\bast\b", r"\bsgot\b", r"\baspartate\s*aminotransferase\b"],
        "unit": "U/L",
        "min": 0.0, "max": 1000.0,
        "default_ref": "10 - 40 U/L",
        "category": "enzymes"
    },
    "ggt": {
        "name": "GGT (Gamma-Glutamyl Transferase)",
        "aliases": [r"\bggt\b", r"\bgamma[- ]?gt\b", r"\bgamma[- ]?glutamyl\b"],
        "unit": "U/L",
        "min": 0.0, "max": 1000.0,
        "default_ref": "9 - 48 U/L",
        "category": "enzymes"
    },
    "bilirubin_total": {
        "name": "Total Bilirubin",
        "aliases": [r"\btotal\s*bilirubin\b", r"\bbilirubin[,\s]*total\b", r"\bt[- ]?bili\b", r"\bbilirubin\b"],
        "unit": "mg/dL",
        "min": 0.0, "max": 30.0,
        "default_ref": "0.1 - 1.2 mg/dL",
        "category": "enzymes"
    },
    "platelet_count": {
        "name": "Platelet Count",
        "aliases": [r"\bplatelet\s*count\b", r"\bplatelets\b", r"\bplt\b"],
        "unit": "×10³/µL",
        "min": 0.0, "max": 1000.0,
        "default_ref": "150 - 450 ×10³/µL",
        "category": "hematology"
    },
    "triglycerides": {
        "name": "Triglycerides",
        "aliases": [r"\btriglycerides\b", r"\btg\b", r"\bserum\s*triglycerides\b"],
        "unit": "mg/dL",
        "min": 0.0, "max": 2000.0,
        "default_ref": "< 150 mg/dL",
        "category": "lipids"
    },
    "hdl_cholestrol": {
        "name": "HDL Cholesterol",
        "aliases": [r"\bhdl[- ]?cholesterol\b", r"\bhdl[- ]?c\b", r"\bhdl\b", r"\bhigh[- ]density\s*lipoprotein\b"],
        "unit": "mg/dL",
        "min": 0.0, "max": 200.0,
        "default_ref": "> 40 mg/dL",
        "category": "lipids"
    },
    "ldl_cholestrol": {
        "name": "LDL Cholesterol",
        "aliases": [r"\bldl[- ]?cholesterol\b", r"\bldl[- ]?c\b", r"\bldl\b", r"\blow[- ]density\s*lipoprotein\b"],
        "unit": "mg/dL",
        "min": 0.0, "max": 400.0,
        "default_ref": "< 100 mg/dL",
        "category": "lipids"
    },
    "fasting_blood_sugar": {
        "name": "Fasting Blood Sugar",
        "aliases": [r"\bfasting\s*blood\s*sugar\b", r"\bfasting\s*glucose\b", r"\bfbs\b", r"\bglucose[,\s]*fasting\b", r"\bblood\s*sugar\b"],
        "unit": "mg/dL",
        "min": 0.0, "max": 600.0,
        "default_ref": "70 - 99 mg/dL",
        "category": "metabolic"
    },
    "tsh": {
        "name": "TSH (Thyroid Stimulating Hormone)",
        "aliases": [r"\btsh\b", r"\bthyroid\s*stimulating\s*hormone\b", r"\bultra\s*tsh\b"],
        "unit": "µIU/mL",
        "min": 0.0, "max": 100.0,
        "default_ref": "0.4 - 4.5 µIU/mL",
        "category": "endocrine"
    },
    "diabetes_type_2": {
        "name": "Type 2 Diabetes History",
        "aliases": [r"\btype\s*2\s*diabetes\b", r"\bt2d\b", r"\bt2dm\b", r"\bdiabetic\b", r"\bdiabetes\s*mellitus\b"],
        "unit": "binary (0/1)",
        "min": 0, "max": 1,
        "default_ref": "No (0) / Yes (1)",
        "category": "history"
    },
    "hypertension": {
        "name": "Hypertension History",
        "aliases": [r"\bhypertension\b", r"\bhtn\b", r"\bhigh\s*blood\s*pressure\b"],
        "unit": "binary (0/1)",
        "min": 0, "max": 1,
        "default_ref": "No (0) / Yes (1)",
        "category": "history"
    },
    "alcohol_status": {
        "name": "Alcohol Consumption",
        "aliases": [r"\balcohol\s*consumption\b", r"\balcohol\s*status\b", r"\balcohol\s*intake\b", r"\balcohol\b"],
        "unit": "binary (0/1)",
        "min": 0, "max": 1,
        "default_ref": "No (0) / Yes (1)",
        "category": "history"
    },
    "smoking_status": {
        "name": "Smoking Status",
        "aliases": [r"\bsmoking\s*status\b", r"\bsmoker\b", r"\btobacco\s*use\b", r"\bcigarette\b"],
        "unit": "binary (0/1)",
        "min": 0, "max": 1,
        "default_ref": "No (0) / Yes (1)",
        "category": "history"
    },
    "pcos": {
        "name": "PCOS Status",
        "aliases": [r"\bpcos\b", r"\bpolycystic\s*ovary\b", r"\bpcod\b"],
        "unit": "category (0/1/2)",
        "min": 0, "max": 2,
        "default_ref": "0=No, 1=Yes, 2=N/A(M)",
        "category": "history"
    }
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract selectable text from PDF using pypdf."""
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"pypdf extraction error: {e}")
    return text.strip()


def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from image using pytesseract if available."""
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"pytesseract extraction error: {e}")
        return ""


def extract_raw_text(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """Determine method and extract text from bytes."""
    ext = Path(filename).suffix.lower()
    method = "direct_pdf"
    raw_text = ""

    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_bytes)
        if not raw_text and PYTESSERACT_AVAILABLE:
            method = "ocr_scanned"
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        method = "ocr_image"
        raw_text = extract_text_from_image(file_bytes)
    
    return raw_text, method


def parse_clinical_parameters(raw_text: str) -> Dict[str, Any]:
    """Parse text lines to identify clinical parameters, values, units, reference ranges, and flags."""
    results = {}
    lines = raw_text.splitlines()

    for key, spec in PARAMETER_DEFINITIONS.items():
        results[key] = {
            "key": key,
            "name": spec["name"],
            "value": None,
            "unit": spec["unit"],
            "reference_range": spec["default_ref"],
            "confidence": 0.0,
            "source_text": None,
            "status": "missing",
            "is_abnormal": False,
            "category": spec["category"],
            "min": spec["min"],
            "max": spec["max"]
        }

    # Line-by-line parsing
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        line_lower = cleaned_line.lower()

        # Check gender explicitly
        if results["gender"]["value"] is None:
            if re.search(r"\b(gender|sex)\b", line_lower):
                if re.search(r"\b(female|woman|f)\b", line_lower):
                    results["gender"].update({
                        "value": 1,
                        "source_text": cleaned_line,
                        "confidence": 0.95,
                        "status": "extracted"
                    })
                elif re.search(r"\b(male|man|m)\b", line_lower):
                    results["gender"].update({
                        "value": 0,
                        "source_text": cleaned_line,
                        "confidence": 0.95,
                        "status": "extracted"
                    })

        # Check binary history flags (Diabetes, Hypertension, Smoking, Alcohol, PCOS)
        for history_key in ["diabetes_type_2", "hypertension", "smoking_status", "alcohol_status", "pcos"]:
            if results[history_key]["value"] is None:
                aliases = PARAMETER_DEFINITIONS[history_key]["aliases"]
                matched = any(re.search(alias, line_lower) for alias in aliases)
                if matched:
                    # Detect presence / absence
                    has_positive = re.search(r"\b(yes|positive|present|diagnosed|active|smoker|drinker|history of)\b", line_lower)
                    has_negative = re.search(r"\b(no|negative|absent|none|denies|non[- ]?smoker|non[- ]?drinker|normal)\b", line_lower)
                    if has_positive and not has_negative:
                        val = 1
                        results[history_key].update({
                            "value": val,
                            "source_text": cleaned_line,
                            "confidence": 0.85,
                            "status": "extracted"
                        })
                    elif has_negative:
                        val = 0
                        results[history_key].update({
                            "value": val,
                            "source_text": cleaned_line,
                            "confidence": 0.90,
                            "status": "extracted"
                        })

        # Check numerical laboratory parameters
        for key, spec in PARAMETER_DEFINITIONS.items():
            if key in ["gender", "diabetes_type_2", "hypertension", "smoking_status", "alcohol_status"]:
                continue
            if results[key]["value"] is not None:
                continue

            matched_alias = None
            for alias in spec["aliases"]:
                if re.search(alias, line_lower):
                    matched_alias = alias
                    break

            if matched_alias:
                # Find numerical values in this line
                # Match numbers like 45, 120.5, 0.85
                num_matches = list(re.finditer(r"([><]?\s*(\d+(?:\.\d+)?))", cleaned_line))
                if not num_matches:
                    continue

                extracted_num = None
                extracted_unit = spec["unit"]
                extracted_ref = spec["default_ref"]

                # Extract reference ranges like (10-40), 7-56, 70 - 100, < 150
                ref_match = re.search(r"(\(?\s*\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*\)?|[><]\s*\d+(?:\.\d+)?)", cleaned_line)
                if ref_match:
                    extracted_ref = ref_match.group(1).strip("()")

                # Pick the measurement value (usually first number after alias or first number in row)
                for m in num_matches:
                    raw_val_str = m.group(2)
                    try:
                        val_float = float(raw_val_str)
                        if spec["min"] <= val_float <= spec["max"]:
                            extracted_num = val_float
                            break
                    except ValueError:
                        continue

                if extracted_num is not None:
                    # Detect abnormal indicators
                    is_abnormal = bool(re.search(r"\b(high|low|abnormal|h\b|l\b|\*|\+)\b", line_lower))
                    
                    # Also assess based on clinical thresholds
                    if key == "alt" and extracted_num > 45: is_abnormal = True
                    if key == "ast" and extracted_num > 40: is_abnormal = True
                    if key == "ggt" and extracted_num > 50: is_abnormal = True
                    if key == "triglycerides" and extracted_num >= 150: is_abnormal = True
                    if key == "fasting_blood_sugar" and extracted_num >= 100: is_abnormal = True
                    if key == "bmi" and extracted_num >= 25: is_abnormal = True

                    results[key].update({
                        "value": extracted_num,
                        "unit": extracted_unit,
                        "reference_range": extracted_ref,
                        "confidence": 0.90,
                        "source_text": cleaned_line,
                        "status": "extracted",
                        "is_abnormal": is_abnormal
                    })

    # Adjust PCOS if gender is Male
    if results["gender"]["value"] == 0:  # Male
        results["pcos"]["value"] = 2
        results["pcos"]["status"] = "extracted"
        results["pcos"]["source_text"] = "Automated clinical rule: PCOS is Not Applicable (2) for male patients"
        results["pcos"]["confidence"] = 1.0

    return results
