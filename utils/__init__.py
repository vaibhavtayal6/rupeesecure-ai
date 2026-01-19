"""
Banknote Verification Utilities
"""

from .image_processing import (
    create_directory_structure,
    preprocess_image,
    save_feature_crop,
    resize_image,
    enhance_image_quality,
    extract_region
)

from .serial_validation import (
    validate_serial_format,
    check_serial_consistency,
    parse_serial_components,
    validate_serial_letters,
    get_denomination_format
)

__all__ = [
    # Image processing
    'create_directory_structure',
    'preprocess_image',
    'save_feature_crop',
    'resize_image',
    'enhance_image_quality',
    'extract_region',
    
    # Serial validation
    'validate_serial_format',
    'check_serial_consistency',
    'parse_serial_components',
    'validate_serial_letters',
    'get_denomination_format'
]

__version__ = "1.0.0"
__author__ = "Banknote Verification System"