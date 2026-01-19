import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import streamlit for secrets support
try:
    import streamlit as st
    def get_api_key():
        """Get API key from Streamlit secrets or environment"""
        try:
            return st.secrets.get("GEMINI_API_KEY", os.getenv('GEMINI_API_KEY'))
        except:
            return os.getenv('GEMINI_API_KEY')
except ImportError:
    def get_api_key():
        """Fallback to environment variables"""
        return os.getenv('GEMINI_API_KEY')

class Config:
    """Configuration class for Banknote Verifier"""
    
    # Gemini API
    GEMINI_API_KEY = get_api_key()
    
    # Application Settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2048'))
    
    # Security Thresholds
    STRONG_MATCH_THRESHOLD = float(os.getenv('STRONG_MATCH_THRESHOLD', '0.85'))
    WEAK_MATCH_THRESHOLD = float(os.getenv('WEAK_MATCH_THRESHOLD', '0.70'))
    MIN_CONFIDENCE_REAL = float(os.getenv('MIN_CONFIDENCE_REAL', '0.80'))
    
    # Feature Configuration
    REQUIRED_FEATURES = [
        'ashok_pillar', 'gandhi', 'security_thread', 
        'serial_num_left', 'serial_num_right'
    ]
    
    OPTIONAL_FEATURES = [
        'colorshift', 'devnagri', 'governor', 
        'latentnum', 'seethrough', 'strips'
    ]
    
    # Serial Number Validation
    ALLOWED_LETTERS = set('ABCDEFGHKLMNPQRSTUVW')
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        errors = []
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required. Please set it in .env file")
        
        if cls.STRONG_MATCH_THRESHOLD <= cls.WEAK_MATCH_THRESHOLD:
            errors.append("STRONG_MATCH_THRESHOLD must be greater than WEAK_MATCH_THRESHOLD")
        
        if cls.MIN_CONFIDENCE_REAL < 0 or cls.MIN_CONFIDENCE_REAL > 1:
            errors.append("MIN_CONFIDENCE_REAL must be between 0 and 1")
        
        return errors
    
    @classmethod
    def get_serial_format(cls, denomination: int) -> dict:
        """Get serial number format for denomination"""
        formats = {
            10: {"pattern": r'^(\d{2})([A-Z])\s(\d{6})$', "description": "NNL NNNNNN"},
            20: {"pattern": r'^(\d{2})([A-Z])\s(\d{6})$', "description": "NNL NNNNNN"},
            50: {"pattern": r'^(\d)([A-Z]{2})\s(\d{5})$', "description": "NLL NNNNN"},
            100: {"pattern": r'^(\d)([A-Z]{2})\s(\d{5})$', "description": "NLL NNNNN"},
            200: {"pattern": r'^(\d)([A-Z]{2})\s(\d{5})$', "description": "NLL NNNNN"},
            500: {"pattern": r'^(\d)([A-Z]{2})\s(\d{5})$', "description": "NLL NNNNN"}
        }
        return formats.get(denomination, {"pattern": "", "description": "Unknown"})

# Validate configuration on import
config_errors = Config.validate_config()
if config_errors:
    print("❌ Configuration errors:")
    for error in config_errors:
        print(f"   - {error}")
    print("Please check your .env file and fix the issues.")