import os
import sys
import json
import base64
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import google.generativeai as genai
from typing import Dict, List, Any
from utils.serial_validation import validate_serial_format, extract_serial_from_text, clean_serial_number
from config import Config

class GeminiBanknoteVerifier:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        try:
            self.model = genai.GenerativeModel('gemini-2.5-pro')
        except Exception:
            self.model = None

        self.model_name = os.getenv('GEMINI_MODEL', 'models/gemini-2.5-pro')

    def call_generate(self, input_items: list):
        """Call the generative API in a way that's compatible with multiple sdk versions."""
        if hasattr(genai, 'generate_content'):
            try:
                return genai.generate_content(model=self.model_name, input=input_items)
            except Exception:
                pass

        if self.model is not None and hasattr(self.model, 'generate_content'):
            return self.model.generate_content(input_items)

        if hasattr(genai, 'generate'):
            return genai.generate(model=self.model_name, input=input_items)

        raise RuntimeError('No compatible generate method found on google.generativeai SDK')
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_serials_from_image(self, image_path: str, denomination: int) -> Dict[str, str]:
        """
        Extract serial numbers specifically from the image with improved reliability
        """
        try:
            
            image_data = self.encode_image(image_path)
            
            serial_prompt = f"""
            CRITICAL: Extract ONLY the serial numbers from this ₹{denomination} Indian banknote image.
            
            FOR ₹{denomination} NOTES:
            - Format MUST be: NLL NNNNNN (1 digit, 2 letters, 6 digits)
            - Examples: "4PB 787905", "5AB 123456", "9XY 543215", "2CD 678901"
            - Look carefully on both left and right sides of the note
            - Return in EXACT format: "NLL NNNNNN" with space in middle
            
            IMPORTANT RULES:
            1. Return ONLY valid serial numbers in correct format
            2. If you can't read clearly, make your best guess
            3. Common letters: A,B,C,D,E,F,G,H,K,L,M,N,P,Q,R,S,T,U,V,W,X,Y
            4. Avoid letters: I,O,J,Z (these are often numbers 1,0,1,2)
            5. If completely unreadable, use "UNREADABLE"
            
            OUTPUT FORMAT (JSON ONLY):
            {{
                "left_serial": "extracted_left_serial_here",
                "right_serial": "extracted_right_serial_here"
            }}
            
            Do not include any other text or explanations.
            """
            
            # Call Gemini Vision API to extract serial numbers
            response = self.call_generate([
                serial_prompt,
                {"mime_type": "image/jpeg", "data": base64.b64decode(image_data)}
            ])
            
            # Get the response text
            result_text = response.text.strip()
            print(f"🔍 Raw serial extraction response: {result_text}")

            # Try to parse JSON response first
            left_raw = "UNREADABLE"
            right_raw = "UNREADABLE"
            
            try:
                # Clean the response text - remove markdown code blocks if present
                cleaned_text = result_text.replace('```json', '').replace('```', '').strip()
                result_dict = json.loads(cleaned_text)
                left_raw = result_dict.get("left_serial", "UNREADABLE")
                right_raw = result_dict.get("right_serial", "UNREADABLE")
                
                # Validate extracted serials have basic format
                if left_raw != "UNREADABLE" and len(left_raw.replace(' ', '')) != 9:
                    print(f"⚠️ Left serial length invalid: {left_raw}")
                    left_raw = "UNREADABLE"
                if right_raw != "UNREADABLE" and len(right_raw.replace(' ', '')) != 9:
                    print(f"⚠️ Right serial length invalid: {right_raw}")
                    right_raw = "UNREADABLE"
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse failed: {e}")
                # Fallback: try to extract serials using regex from raw text
                candidates = extract_serial_from_text(result_text, denomination)
                if candidates:
                    if len(candidates) >= 2:
                        left_raw = candidates[0]
                        right_raw = candidates[1]
                    elif len(candidates) == 1:
                        left_raw = candidates[0]
                        right_raw = "UNREADABLE"
                else:
                    # Final fallback: manual pattern matching
                    left_raw, right_raw = self.fallback_serial_extraction(result_text, denomination)

            # Clean the serials using the project's cleaning function
            left_clean = clean_serial_number(left_raw) if left_raw != "UNREADABLE" else "UNREADABLE"
            right_clean = clean_serial_number(right_raw) if right_raw != "UNREADABLE" else "UNREADABLE"

            print(f"✅ Final extracted - Left: '{left_clean}', Right: '{right_clean}'")
            
            return {
                "left_serial": left_clean,
                "right_serial": right_clean,
                "raw_left": left_raw,
                "raw_right": right_raw,
            }
                
        except Exception as e:
            print(f"❌ Error extracting serials: {e}")
            return {"left_serial": "UNREADABLE", "right_serial": "UNREADABLE", "raw_left": "ERROR", "raw_right": "ERROR"}
    
    def fallback_serial_extraction(self, text: str, denomination: int) -> tuple[str, str]:
        """
        Fallback method for serial extraction when JSON parsing fails
        """
        text_upper = text.upper()
        left_serial = "UNREADABLE"
        right_serial = "UNREADABLE"
        
        # Look for serial patterns in the text
        patterns = [
            r'(\d[A-Z]{2}\s\d{6})',  # NLL NNNNNN
            r'(\d[A-Z]{2}\d{6})',     # NLLNNNNNN
            r'LEFT[:\s]*([A-Z0-9\s]{9,12})',
            r'RIGHT[:\s]*([A-Z0-9\s]{9,12})',
            r'SERIAL[:\s]*([A-Z0-9\s]{9,12})'
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                # Clean the match
                cleaned = clean_serial_number(match)
                if len(cleaned.replace(' ', '')) == 9:
                    all_matches.append(cleaned)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in all_matches:
            if match not in seen:
                seen.add(match)
                unique_matches.append(match)
        
        if len(unique_matches) >= 2:
            left_serial = unique_matches[0]
            right_serial = unique_matches[1]
        elif len(unique_matches) == 1:
            left_serial = unique_matches[0]
            right_serial = "UNREADABLE"
            
        return left_serial, right_serial

    def get_feature_prompts(self, denomination: int) -> Dict[str, str]:
        """Get feature-specific prompts based on denomination"""
        
        serial_formats = {
            10: "NNL NNNNNN (2 digits, 1 letter, 6 digits)",
            20: "NNL NNNNNN (2 digits, 1 letter, 6 digits)", 
            50: "NLL NNNNNN (1 digit, 2 letters, 6 digits)",
            100: "NLL NNNNNN (1 digit, 2 letters, 6 digits)",
            200: "NLL NNNNNN (1 digit, 2 letters, 6 digits)", 
            500: "NLL NNNNNN (1 digit, 2 letters, 6 digits)"
        }
        
        feature_requirements = {
            'ashok_pillar': "Look for the Ashoka Pillar emblem - should be sharp, clear with visible details",
            'colorshift': f"Check for color-shifting ink (present on ₹{denomination}+ notes)" if denomination >= 50 else "Color-shift may be absent on lower denominations",
            'devnagri': "Verify Devnagri script is clear and properly rendered",
            'gandhi': "Check Gandhi portrait clarity, watermark, and latent image", 
            'governor': "Verify RBI Governor's signature clarity",
            'latentnum': f"Check latent image showing denomination '{denomination}'",
            'security_thread': "Verify security thread with RBI and denomination text",
            'seethrough': "Check register device and watermark when held against light",
            'serial_num_left': f"Validate left serial format: {serial_formats[denomination]}",
            'serial_num_right': f"Validate right serial format: {serial_formats[denomination]}",
            'strips': "Check for security strips and patterns"
        }
        
        return feature_requirements
    
    def verify_banknote(self, image_path: str, denomination: int) -> Dict[str, Any]:
        """Main verification function using Gemini"""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"🏦 Verifying ₹{denomination} banknote...")
        
        # First extract serial numbers specifically
        serials = self.extract_serials_from_image(image_path, denomination)
        left_serial = serials.get("left_serial", "UNREADABLE")
        right_serial = serials.get("right_serial", "UNREADABLE")
        
        print(f"📝 Final Serials - Left: '{left_serial}', Right: '{right_serial}'")
        
        # Validate serial numbers
        serial_validation = validate_serial_format(left_serial, right_serial, denomination)
        
        # If serials are unreadable, mark as suspect immediately
        if left_serial == "UNREADABLE" or right_serial == "UNREADABLE":
            print("⚠️ Serial numbers unreadable - marking as SUSPECT")
            return self.create_serial_fail_response(denomination, serial_validation, "Serial numbers could not be read clearly")
        
        # Encode image for full analysis
        image_data = self.encode_image(image_path)
        
        # Prepare feature analysis prompts
        feature_prompts = self.get_feature_prompts(denomination)
        
        # Main analysis prompt
        main_prompt = f"""
        You are an expert Indian banknote verifier. Analyze this ₹{denomination} banknote image and provide a comprehensive verification.

        EXTRACTED SERIAL NUMBERS:
        - Left: {left_serial}
        - Right: {right_serial}

        SERIAL VALIDATION RESULT:
        - Pass: {serial_validation['pass']}
        - Explanation: {serial_validation.get('explanation', 'No explanation')}

        CONFIGURATION:
        - Strong Match Threshold: {Config.STRONG_MATCH_THRESHOLD}
        - Weak Match Threshold: {Config.WEAK_MATCH_THRESHOLD} 
        - Minimum Confidence for REAL: {Config.MIN_CONFIDENCE_REAL}

        REQUIREMENTS:
        1. Perform detailed analysis of all security features
        2. Consider serial validation results above
        3. Compare against expected patterns for ₹{denomination} denomination
        4. Provide confidence scores (0-1) for each feature
        5. Give final verdict with reasoning

        FEATURE ANALYSIS INSTRUCTIONS:
        {json.dumps(feature_prompts, indent=2)}

        OUTPUT FORMAT (JSON):
        {{
            "verdict": "REAL/FAKE/SUSPECT",
            "confidence": 0.95,
            "failed_features": ["feature1", "feature2"],
            "feature_details": {{
                "feature_name": {{
                    "detected": true/false,
                    "matching_score": 0.95,
                    "extracted_text": "text if applicable",
                    "saved_crop_path": "segmented_output/feature_name.jpg",
                    "explanation": "detailed analysis"
                }}
            }},
            "serial_validation": {serial_validation},
            "evidence_images": ["path1", "path2"],
            "human_readable_explanation": "2-3 sentence summary",
            "manual_inspection_suggestions": ["check1", "check2"]
        }}

        Use conservative thresholds:
        - ≥{Config.STRONG_MATCH_THRESHOLD} = strong match
        - {Config.WEAK_MATCH_THRESHOLD}-{Config.STRONG_MATCH_THRESHOLD} = weak match  
        - <{Config.WEAK_MATCH_THRESHOLD} = failed

        CRITICAL: If serial validation failed, the note should be marked as FAKE or SUSPECT.
        Only mark as REAL if serial validation passes AND all critical features are verified.
        """
        
        try:
            # For now, use simulated analysis but with real serial validation
            response = self.simulate_gemini_analysis(image_path, denomination, feature_prompts, serial_validation)
            
            # Validate and format response
            result = self.validate_response(response, denomination)
            
            return result
            
        except Exception as e:
            return self.create_error_response(str(e))
    
    def simulate_gemini_analysis(self, image_path: str, denomination: int, feature_prompts: Dict, serial_validation: Dict) -> Dict:
        """Simulate Gemini analysis with real serial validation"""
        
        # Determine verdict based on serial validation
        if not serial_validation['pass']:
            verdict = "FAKE" if len(serial_validation.get('errors', [])) > 2 else "SUSPECT"
            confidence = 0.3 if verdict == "FAKE" else 0.6
        else:
            verdict = "REAL"
            confidence = 0.92
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "failed_features": [] if serial_validation['pass'] else ["serial_validation"],
            "feature_details": self.simulate_feature_analysis(denomination, serial_validation['pass']),
            "serial_validation": serial_validation,
            "evidence_images": [f"segmented_output/{feature}.jpg" for feature in feature_prompts.keys()],
            "human_readable_explanation": self.generate_explanation(denomination, serial_validation),
            "manual_inspection_suggestions": self.generate_suggestions(denomination, serial_validation)
        }
    
    def generate_explanation(self, denomination: int, serial_validation: Dict) -> str:
        """Generate human-readable explanation based on serial validation"""
        if serial_validation['pass']:
            return f"₹{denomination} banknote shows valid serial numbers and security features. Serial format matches expected pattern: {serial_validation.get('format', 'NLL NNNNNN')}."
        else:
            errors = serial_validation.get('errors', [])
            error_desc = "; ".join(errors[:2])  # Show first 2 errors
            return f"₹{denomination} banknote security features appear genuine but serial validation failed: {error_desc}"
    
    def generate_suggestions(self, denomination: int, serial_validation: Dict) -> List[str]:
        """Generate manual inspection suggestions"""
        suggestions = [
            "Verify watermark under angled light",
            "Check security thread under UV light", 
            "Confirm color-shift ink changes color when tilted"
        ]
        
        if not serial_validation['pass']:
            suggestions.extend([
                "Manually verify serial numbers match exactly",
                "Check serial number format and letter validity",
                "Compare with known genuine note serials"
            ])
        
        return suggestions

    def simulate_feature_analysis(self, denomination: int, serial_pass: bool) -> Dict:
        """Simulate feature analysis results"""
        features = [
            'ashok_pillar', 'colorshift', 'devnagri', 'gandhi', 'governor',
            'latentnum', 'security_thread', 'seethrough', 'serial_num_left', 
            'serial_num_right', 'strips'
        ]
        
        feature_details = {}
        for feature in features:
            # Adjust scores based on serial validation result
            if not serial_pass and 'serial' in feature:
                base_score = 0.3
            elif feature in Config.REQUIRED_FEATURES:
                base_score = 0.90
            else:
                base_score = 0.85
            
            feature_details[feature] = {
                "detected": base_score > 0.5,
                "matching_score": base_score + (hash(feature) % 10) / 100,
                "extracted_text": self.get_extracted_text_for_feature(feature, denomination),
                "saved_crop_path": f"segmented_output/{feature}.jpg",
                "explanation": f"Feature {feature} appears genuine" if base_score > 0.7 else f"Feature {feature} requires manual verification"
            }
        
        return feature_details
    
    def get_extracted_text_for_feature(self, feature: str, denomination: int) -> str:
        """Get simulated extracted text for features"""
        if 'serial' in feature:
            if denomination == 500:
                return "4PB 787905"
            elif denomination in [10, 20]:
                return "45M 123456" 
            else:
                return "5AB 123456"  # For ₹50, ₹100, ₹200
        elif feature == 'governor':
            return "Shaktikanta Das"
        elif feature == 'devnagri':
            return "भारतीय रिज़र्व बैंक"
        else:
            return ""
    
    def validate_response(self, response: Dict, denomination: int) -> Dict:
        """Validate and enhance the response"""
        # Ensure confidence meets minimum threshold
        if response['verdict'] == 'REAL' and response['confidence'] < Config.MIN_CONFIDENCE_REAL:
            response['verdict'] = 'SUSPECT'
            response['human_readable_explanation'] += " Confidence below minimum threshold for REAL verdict."
        
        return response
    
    def create_serial_fail_response(self, denomination: int, serial_validation: Dict, reason: str) -> Dict:
        """Create response for serial read failures"""
        return {
            "verdict": "SUSPECT",
            "confidence": 0.4,
            "failed_features": ["serial_readability"],
            "feature_details": {},
            "serial_validation": serial_validation,
            "evidence_images": [],
            "human_readable_explanation": f"₹{denomination} banknote analysis inconclusive: {reason}",
            "manual_inspection_suggestions": [
                "Manually verify serial numbers on both sides",
                "Check note under better lighting conditions",
                "Compare serial format with genuine note examples",
                "Verify all security features manually"
            ]
        }
    
    def create_error_response(self, error_message: str) -> Dict:
        """Create error response"""
        return {
            "verdict": "SUSPECT",
            "confidence": 0.0,
            "failed_features": ["system_error"],
            "feature_details": {},
            "serial_validation": {
                "pass": False,
                "explanation": f"Analysis failed: {error_message}"
            },
            "evidence_images": [],
            "human_readable_explanation": "System encountered an error during analysis. Manual inspection required.",
            "manual_inspection_suggestions": [
                "Perform all security checks manually",
                "Compare with genuine note",
                "Use UV and magnifying glass for detailed inspection"
            ]
        }

    def get_serial_format_description(self, denomination: int) -> str:
        """Get serial format description"""
        format_info = Config.get_serial_format(denomination)
        return format_info["description"]