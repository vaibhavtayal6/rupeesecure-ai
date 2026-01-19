import re
from typing import Dict, Tuple, Optional, List
import logging
from config import Config

logger = logging.getLogger(__name__)

def validate_serial_format(left_serial: str, right_serial: str, denomination: int) -> Dict:
    """
    Validate serial number format based on denomination
    
    Args:
        left_serial: Left serial number string
        right_serial: Right serial number string
        denomination: Banknote denomination
        
    Returns:
        Validation result dictionary
    """
    logger.info(f"🔢 Validating serial numbers for ₹{denomination} note")
    logger.info(f"   Left: '{left_serial}', Right: '{right_serial}'")
    
    # Get format pattern for denomination
    format_info = Config.get_serial_format(denomination)
    pattern = format_info["pattern"]
    format_desc = format_info["description"]
    
    # Clean serial numbers - remove extra spaces and normalize
    left_clean = clean_serial_number(left_serial) if left_serial and left_serial != "UNREADABLE" else ""
    right_clean = clean_serial_number(right_serial) if right_serial and right_serial != "UNREADABLE" else ""
    
    validation_result = {
        'pass': False,
        'left_serial': left_clean,
        'right_serial': right_clean,
        'format': format_desc,
        'explanation': '',
        'parsed_left': None,
        'parsed_right': None,
        'errors': []
    }
    
    # Check if serial numbers are provided
    if not left_clean or not right_clean:
        validation_result['errors'].append("One or both serial numbers are empty or unreadable")
        validation_result['explanation'] = "Missing or unreadable serial numbers"
        return validation_result
    
    # Use dedicated validation functions for specific denominations
    if denomination == 500:
        return validate_500_serial_format(left_clean, right_clean)
    elif denomination in [100, 200]:
        return validate_100_200_serial_format(left_clean, right_clean, denomination)
    
    # Validate format using regex for other denominations (₹10, ₹20, ₹50)
    left_match = re.match(pattern, left_clean)
    right_match = re.match(pattern, right_clean)
    
    if not left_match:
        validation_result['errors'].append(f"Left serial '{left_clean}' doesn't match format {format_desc}")
    
    if not right_match:
        validation_result['errors'].append(f"Right serial '{right_clean}' doesn't match format {format_desc}")
    
    if not left_match or not right_match:
        validation_result['explanation'] = "Serial numbers don't match expected format"
        return validation_result
    
    # Parse components based on denomination
    if denomination in [10, 20]:
        # Format: NNL NNNNNN (2 digits, 1 letter, 6 digits)
        left_digits1, left_letter, left_digits2 = left_match.groups()
        right_digits1, right_letter, right_digits2 = right_match.groups()
        
        left_parsed = f"{left_digits1}{left_letter} {left_digits2}"
        right_parsed = f"{right_digits1}{right_letter} {right_digits2}"
        
        # Validate letters
        if not validate_serial_letters([left_letter]):
            validation_result['errors'].append(f"Left serial contains invalid letter '{left_letter}'")
        
        if not validate_serial_letters([right_letter]):
            validation_result['errors'].append(f"Right serial contains invalid letter '{right_letter}'")
    
    else:
        # For ₹50 - Format: NLL NNNNNN (1 digit, 2 letters, 6 digits)
        left_digit, left_letters, left_digits2 = left_match.groups()
        right_digit, right_letters, right_digits2 = right_match.groups()
        
        left_parsed = f"{left_digit}{left_letters} {left_digits2}"
        right_parsed = f"{right_digit}{right_letters} {right_digits2}"
        
        # Validate letters
        if not validate_serial_letters(list(left_letters)):
            validation_result['errors'].append(f"Left serial contains invalid letters '{left_letters}'")
        
        if not validate_serial_letters(list(right_letters)):
            validation_result['errors'].append(f"Right serial contains invalid letters '{right_letters}'")
    
    validation_result['parsed_left'] = left_parsed
    validation_result['parsed_right'] = right_parsed
    
    # Check if serial numbers are consistent
    consistency, consistency_msg = check_serial_consistency(left_parsed, right_parsed)
    if not consistency:
        validation_result['errors'].append(consistency_msg)
    
    # Check for specific patterns that might indicate counterfeiting
    if is_suspicious_serial(left_parsed) or is_suspicious_serial(right_parsed):
        validation_result['errors'].append("Serial number pattern appears suspicious")
    
    # Final validation
    if not validation_result['errors']:
        validation_result['pass'] = True
        validation_result['explanation'] = "Serial numbers validated successfully"
        logger.info("✅ Serial number validation passed")
    else:
        validation_result['explanation'] = "; ".join(validation_result['errors'])
        logger.warning(f"❌ Serial number validation failed: {validation_result['explanation']}")
    
    return validation_result

def validate_100_200_serial_format(left_serial: str, right_serial: str, denomination: int) -> Dict:
    """
    Special validation for ₹100 and ₹200 notes - format: NLL NNNNNN (1 digit, 2 letters, 6 digits)
    """
    validation_result = {
        'pass': False,
        'left_serial': left_serial,
        'right_serial': right_serial,
        'format': "NLL NNNNNN",
        'explanation': '',
        'parsed_left': None,
        'parsed_right': None,
        'errors': []
    }
    
    # Pattern for ₹100 and ₹200: 1 digit, 2 letters, space, 6 digits
    pattern = r'^(\d)([A-Z]{2})\s(\d{6})$'
    
    left_match = re.match(pattern, left_serial.upper())
    right_match = re.match(pattern, right_serial.upper())
    
    # Try to fix common OCR errors if initial match fails
    if not left_match:
        left_serial_fixed = fix_common_serial_errors(left_serial)
        left_match = re.match(pattern, left_serial_fixed.upper())
        if left_match:
            validation_result['left_serial'] = left_serial_fixed
            left_serial = left_serial_fixed
        else:
            # Try more aggressive cleaning for ₹100/₹200
            left_serial_fixed = aggressive_serial_clean(left_serial, denomination)
            left_match = re.match(pattern, left_serial_fixed.upper())
            if left_match:
                validation_result['left_serial'] = left_serial_fixed
                left_serial = left_serial_fixed
    
    if not right_match:
        right_serial_fixed = fix_common_serial_errors(right_serial)
        right_match = re.match(pattern, right_serial_fixed.upper())
        if right_match:
            validation_result['right_serial'] = right_serial_fixed
            right_serial = right_serial_fixed
        else:
            right_serial_fixed = aggressive_serial_clean(right_serial, denomination)
            right_match = re.match(pattern, right_serial_fixed.upper())
            if right_match:
                validation_result['right_serial'] = right_serial_fixed
                right_serial = right_serial_fixed
    
    if not left_match:
        validation_result['errors'].append(f"Left serial '{left_serial}' doesn't match format NLL NNNNNN")
        # Provide detailed error analysis
        error_analysis = analyze_serial_error(left_serial, denomination)
        if error_analysis:
            validation_result['errors'].append(error_analysis)
    
    if not right_match:
        validation_result['errors'].append(f"Right serial '{right_serial}' doesn't match format NLL NNNNNN")
        error_analysis = analyze_serial_error(right_serial, denomination)
        if error_analysis:
            validation_result['errors'].append(error_analysis)
    
    if not left_match or not right_match:
        validation_result['explanation'] = f"Serial numbers don't match expected format for ₹{denomination}"
        return validation_result
    
    # Extract components
    left_digit, left_letters, left_digits2 = left_match.groups()
    right_digit, right_letters, right_digits2 = right_match.groups()
    
    left_parsed = f"{left_digit}{left_letters} {left_digits2}"
    right_parsed = f"{right_digit}{right_letters} {right_digits2}"
    
    validation_result['parsed_left'] = left_parsed
    validation_result['parsed_right'] = right_parsed
    
    # Validate letters are from allowed set
    if not validate_serial_letters(list(left_letters)):
        validation_result['errors'].append(f"Left serial contains invalid letters '{left_letters}'. Allowed: {''.join(sorted(Config.ALLOWED_LETTERS))}")
    
    if not validate_serial_letters(list(right_letters)):
        validation_result['errors'].append(f"Right serial contains invalid letters '{right_letters}'. Allowed: {''.join(sorted(Config.ALLOWED_LETTERS))}")
    
    # Check consistency
    consistency, consistency_msg = check_serial_consistency(left_parsed, right_parsed)
    if not consistency:
        validation_result['errors'].append(consistency_msg)
    
    # Final validation
    if not validation_result['errors']:
        validation_result['pass'] = True
        validation_result['explanation'] = f"₹{denomination} serial numbers validated successfully"
        logger.info(f"✅ ₹{denomination} Serial validation passed: {left_parsed}")
    else:
        validation_result['explanation'] = "; ".join(validation_result['errors'])
        logger.warning(f"❌ ₹{denomination} Serial validation failed: {validation_result['explanation']}")
    
    return validation_result

def validate_500_serial_format(left_serial: str, right_serial: str) -> Dict:
    """
    Special validation for ₹500 notes - format: NLL NNNNNN (1 digit, 2 letters, 6 digits)
    """
    validation_result = {
        'pass': False,
        'left_serial': left_serial,
        'right_serial': right_serial,
        'format': "NLL NNNNNN",
        'explanation': '',
        'parsed_left': None,
        'parsed_right': None,
        'errors': []
    }
    
    # Pattern for ₹500: 1 digit, 2 letters, space, 6 digits
    pattern = r'^(\d)([A-Z]{2})\s(\d{6})$'
    
    left_match = re.match(pattern, left_serial.upper())
    right_match = re.match(pattern, right_serial.upper())
    
    if not left_match:
        # Try alternative patterns - sometimes OCR misreads
        left_serial_fixed = fix_common_serial_errors(left_serial)
        left_match = re.match(pattern, left_serial_fixed.upper())
        if left_match:
            validation_result['left_serial'] = left_serial_fixed
            left_serial = left_serial_fixed
    
    if not right_match:
        right_serial_fixed = fix_common_serial_errors(right_serial)
        right_match = re.match(pattern, right_serial_fixed.upper())
        if right_match:
            validation_result['right_serial'] = right_serial_fixed
            right_serial = right_serial_fixed
    
    if not left_match:
        validation_result['errors'].append(f"Left serial '{left_serial}' doesn't match format NLL NNNNNN")
        # Provide helpful suggestions
        if len(left_serial) == 9 and ' ' in left_serial:
            parts = left_serial.split(' ')
            if len(parts[0]) == 3 and len(parts[1]) == 5:
                validation_result['errors'].append(f"Format appears correct but validation failed. Check if '{parts[0][0]}' is digit and '{parts[0][1:3]}' are valid letters")
    
    if not right_match:
        validation_result['errors'].append(f"Right serial '{right_serial}' doesn't match format NLL NNNNNN")
    
    if not left_match or not right_match:
        validation_result['explanation'] = "Serial numbers don't match expected format for ₹500"
        return validation_result
    
    # Extract components
    left_digit, left_letters, left_digits2 = left_match.groups()
    right_digit, right_letters, right_digits2 = right_match.groups()
    
    left_parsed = f"{left_digit}{left_letters} {left_digits2}"
    right_parsed = f"{right_digit}{right_letters} {right_digits2}"
    
    validation_result['parsed_left'] = left_parsed
    validation_result['parsed_right'] = right_parsed
    
    # Validate letters are from allowed set
    if not validate_serial_letters(list(left_letters)):
        validation_result['errors'].append(f"Left serial contains invalid letters '{left_letters}'. Allowed: {''.join(sorted(Config.ALLOWED_LETTERS))}")
    
    if not validate_serial_letters(list(right_letters)):
        validation_result['errors'].append(f"Right serial contains invalid letters '{right_letters}'. Allowed: {''.join(sorted(Config.ALLOWED_LETTERS))}")
    
    # Check consistency
    consistency, consistency_msg = check_serial_consistency(left_parsed, right_parsed)
    if not consistency:
        validation_result['errors'].append(consistency_msg)
    
    # Final validation
    if not validation_result['errors']:
        validation_result['pass'] = True
        validation_result['explanation'] = "₹500 serial numbers validated successfully"
        logger.info(f"✅ ₹500 Serial validation passed: {left_parsed}")
    else:
        validation_result['explanation'] = "; ".join(validation_result['errors'])
        logger.warning(f"❌ ₹500 Serial validation failed: {validation_result['explanation']}")
    
    return validation_result

def clean_serial_number(serial: str) -> str:
    """
    Clean and normalize serial number string
    """
    if not serial or serial == "UNREADABLE":
        return ""
    
    # Remove extra spaces and normalize
    cleaned = ' '.join(serial.upper().split())

    # Heuristic: try to split into prefix (3 chars) and suffix (rest) when possible
    # Common formats:
    #  - ₹500:  NLL NNNNNN  -> prefix length = 3, suffix length = 6
    #  - ₹10/20: NNL NNNNNN -> prefix length = 3, suffix length = 6
    # We'll remove spaces and re-insert a space after the first 3 characters.
    t = re.sub(r"\s+", "", cleaned)
    if len(t) >= 6:
        prefix = t[:3]
        suffix = t[3:]

        # Fix common OCR confusions only in the numeric suffix (not letters)
        # Map likely misreads to digits: O->0, I->1, S->5, Z->2, B->8, Q->0
        trans = str.maketrans({'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8', 'Q': '0'})
        suffix_fixed = suffix.translate(trans)

        # Return normalized format with a space between prefix and suffix
        return f"{prefix} {suffix_fixed}"

    # Fallback: return the cleaned, uppercased value
    return cleaned

def aggressive_serial_clean(serial: str, denomination: int) -> str:
    """
    More aggressive cleaning for problematic serial numbers
    """
    if not serial:
        return ""
    
    # Remove all non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^A-Z0-9\s]', '', serial.upper())
    
    # Remove extra spaces
    cleaned = ' '.join(cleaned.split())
    
    # If no spaces, try to auto-insert space after 3 characters
    if ' ' not in cleaned and len(cleaned) == 9:
        cleaned = cleaned[:3] + ' ' + cleaned[3:]
    
    # Fix common character confusions
    char_map = {
        'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2',
        'B': '8', 'Q': '0', 'G': '6', 'T': '7'
    }
    
    # Apply character mapping more carefully
    parts = cleaned.split(' ')
    if len(parts) == 2:
        prefix, suffix = parts
        # Only fix suffix (numeric part) for character confusions
        fixed_suffix = ''
        for char in suffix:
            fixed_suffix += char_map.get(char, char)
        cleaned = f"{prefix} {fixed_suffix}"
    
    return cleaned

def fix_common_serial_errors(serial: str) -> str:
    """
    Fix common serial number OCR errors
    """
    if not serial:
        return ""
    
    fixed = serial.upper()
    
    # Common fixes for ₹500 format
    if len(fixed) == 9 and ' ' not in fixed:
        # If no space but correct length, add space after 3rd character
        fixed = fixed[:3] + ' ' + fixed[3:]
    elif len(fixed) == 9 and fixed[3] != ' ':
        # If wrong character at position 3, replace with space
        fixed = fixed[:3] + ' ' + fixed[4:]
    
    return fixed

def analyze_serial_error(serial: str, denomination: int) -> str:
    """
    Analyze why a serial number failed validation and provide helpful feedback
    """
    if not serial:
        return "Serial number is empty"
    
    serial_upper = serial.upper()
    
    # Check length
    if len(serial_upper.replace(' ', '')) != 9:
        return f"Expected 9 characters but got {len(serial_upper.replace(' ', ''))}"
    
    # Check for invalid characters
    invalid_chars = re.findall(r'[^A-Z0-9\s]', serial_upper)
    if invalid_chars:
        return f"Invalid characters found: {', '.join(set(invalid_chars))}"
    
    # Check space position
    if ' ' in serial_upper:
        parts = serial_upper.split(' ')
        if len(parts) != 2:
            return "Multiple spaces found - should be only one space"
        if len(parts[0]) != 3:
            return f"Prefix should be 3 characters before space, found {len(parts[0])}"
        if len(parts[1]) != 6:
            return f"Suffix should be 6 characters after space, found {len(parts[1])}"
    
    return "Format appears correct but validation failed - check individual characters"

def check_serial_consistency(serial1: str, serial2: str) -> Tuple[bool, str]:
    """
    Check if two serial numbers are consistent
    
    Args:
        serial1: First serial number
        serial2: Second serial number
        
    Returns:
        Tuple of (is_consistent, message)
    """
    if serial1 == serial2:
        return True, "Serial numbers match"
    else:
        return False, f"Serial mismatch: '{serial1}' vs '{serial2}'"

def parse_serial_components(serial: str, denomination: int) -> Optional[Dict]:
    """
    Parse serial number into its components
    
    Args:
        serial: Serial number string
        denomination: Banknote denomination
        
    Returns:
        Dictionary with parsed components or None if invalid
    """
    format_info = Config.get_serial_format(denomination)
    pattern = format_info["pattern"]
    
    # Clean the serial first
    serial_clean = clean_serial_number(serial)
    
    match = re.match(pattern, serial_clean.upper())
    if not match:
        return None
    
    if denomination in [10, 20]:
        digits1, letter, digits2 = match.groups()
        return {
            'prefix_digits': digits1,
            'letter': letter,
            'suffix_digits': digits2,
            'full_serial': f"{digits1}{letter} {digits2}"
        }
    else:
        digit, letters, digits2 = match.groups()
        return {
            'prefix_digit': digit,
            'letters': letters,
            'suffix_digits': digits2,
            'full_serial': f"{digit}{letters} {digits2}"
        }

def validate_serial_letters(letters: List[str]) -> bool:
    """
    Validate that all letters are from the allowed set
    
    Args:
        letters: List of letters to validate
        
    Returns:
        True if all letters are valid
    """
    for letter in letters:
        if letter not in Config.ALLOWED_LETTERS:
            return False
    return True

def get_denomination_format(denomination: int) -> Dict:
    """
    Get serial number format description for denomination
    
    Args:
        denomination: Banknote denomination
        
    Returns:
        Dictionary with format information
    """
    return Config.get_serial_format(denomination)

def is_suspicious_serial(serial: str) -> bool:
    """
    Check for suspicious serial number patterns
    
    Args:
        serial: Serial number to check
        
    Returns:
        True if serial appears suspicious
    """
    suspicious_patterns = [
        r'^0+',  # All zeros prefix
        r'(\d)\1{2,}',  # Repeated digits 3+ times
        r'(\w)\1{2,}',  # Repeated characters 3+ times
        r'^123456',  # Sequential numbers
        r'^000000',  # All zeros
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, serial):
            return True
    
    return False

def generate_serial_examples(denomination: int) -> List[str]:
    """
    Generate example serial numbers for a denomination
    
    Args:
        denomination: Banknote denomination
        
    Returns:
        List of example serial numbers
    """
    if denomination in [10, 20]:
        return ["45M 123456", "78H 654321", "23K 987654"]
    else:
        return ["5AB 123456", "9XY 543217", "2CD 678907", "4PB 787890"]

def extract_serial_from_text(text: str, denomination: int) -> List[str]:
    """
    Extract potential serial numbers from text using multiple patterns
    
    Args:
        text: Text to search for serial numbers
        denomination: Banknote denomination
        
    Returns:
        List of potential serial numbers found
    """
    potential_serials = []
    
    # Common serial number patterns
    patterns = [
        r'\b\d[A-Z]{2}\s\d{6}\b',  # NLL NNNNNN (₹500 format)
        r'\b\d{2}[A-Z]\s\d{6}\b',   # NNL NNNNNN (₹10/20 format)
        r'\b[A-Z]{3}\s\d{6}\b',     # LLL NNNNNN (alternative)
        r'\b\d{3}\s\d{5}\b',        # NNN NNNNN (digits only)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        potential_serials.extend(matches)
    
    # Also look for serials without spaces
    no_space_patterns = [
        r'\b\d[A-Z]{2}\d{6}\b',  # NLLNNNNNN
        r'\b\d{2}[A-Z]\d{6}\b',   # NNLNNNNNN
    ]
    
    for pattern in no_space_patterns:
        matches = re.findall(pattern, text.upper())
        # Add space for proper formatting
        for match in matches:
            if len(match) == 9:  # NLLNNNNNN
                formatted = match[:3] + ' ' + match[3:]
                potential_serials.append(formatted)
            elif len(match) == 9:  # NNLNNNNNN
                formatted = match[:3] + ' ' + match[3:]
                potential_serials.append(formatted)
    
    return list(set(potential_serials))  # Remove duplicates

# Debug function to test serial validation
def debug_serial_validation(denomination: int, left_serial: str, right_serial: str):
    """Test why serial validation is failing"""
    print(f"\n🔍 DEBUG ₹{denomination} Serial Validation:")
    print(f"Left: '{left_serial}'")
    print(f"Right: '{right_serial}'")
    
    result = validate_serial_format(left_serial, right_serial, denomination)
    
    print(f"✅ Pass: {result['pass']}")
    print(f"📝 Format: {result['format']}")
    print(f"❌ Errors: {result.get('errors', [])}")
    print(f"💡 Explanation: {result.get('explanation', '')}")
    
    return result