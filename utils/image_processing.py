import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory_structure(base_dir: str = ".") -> None:
    """
    Create necessary directory structure for the banknote verification system
    
    Args:
        base_dir: Base directory where structure should be created
    """
    directories = [
        'segmented_output',
        'feature_crops/ashok_pillar',
        'feature_crops/colorshift', 
        'feature_crops/devnagri',
        'feature_crops/gandhi',
        'feature_crops/governor',
        'feature_crops/latentnum',
        'feature_crops/security_thread',
        'feature_crops/seethrough',
        'feature_crops/serial_num_left',
        'feature_crops/serial_num_right',
        'feature_crops/strips',
        'logs',
        'reports'
    ]
    
    created_dirs = []
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        try:
            os.makedirs(full_path, exist_ok=True)
            created_dirs.append(full_path)
        except Exception as e:
            logger.warning(f"Could not create directory {full_path}: {e}")
    
    logger.info(f"📁 Created/verified {len(created_dirs)} directories")
    return created_dirs

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (800, 400)) -> np.ndarray:
    """
    Preprocess image for banknote analysis
    
    Args:
        image_path: Path to input image
        target_size: Target size (width, height) for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        logger.info(f"📸 Loaded image: {image.shape} (HxWxC)")
        
        # Convert to RGB (OpenCV uses BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to standard banknote aspect ratio (2:1)
        original_height, original_width = image_rgb.shape[:2]
        target_width, target_height = target_size
        
        # Maintain aspect ratio while resizing
        aspect_ratio = original_width / original_height
        target_aspect = target_width / target_height
        
        if aspect_ratio > target_aspect:
            # Image is wider than target
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Image is taller than target
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        resized = cv2.resize(image_rgb, (new_width, new_height))
        
        # Create canvas of target size with white background
        canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
        
        # Center the resized image on canvas
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        logger.info(f"🔄 Resized image to: {canvas.shape} (HxWxC)")
        return canvas
        
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {e}")
        raise

def resize_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Resize image by scale factor
    
    Args:
        image: Input image
        scale_factor: Scaling factor (e.g., 0.5 for half size)
        
    Returns:
        Resized image
    """
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive")
    
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    
    return cv2.resize(image, (new_width, new_height))

def enhance_image_quality(image: np.ndarray, 
                         contrast: float = 1.2, 
                         brightness: int = 10,
                         sharpen: bool = True) -> np.ndarray:
    """
    Enhance image quality for better feature detection
    
    Args:
        image: Input image
        contrast: Contrast enhancement factor
        brightness: Brightness adjustment
        sharpen: Whether to apply sharpening filter
        
    Returns:
        Enhanced image
    """
    enhanced = image.copy()
    
    # Convert to float for processing
    enhanced = enhanced.astype(np.float32) / 255.0
    
    # Adjust contrast and brightness
    enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=contrast, beta=brightness)
    
    if sharpen:
        # Apply sharpening filter
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Apply Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Enhance edges
    enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    
    logger.info("✨ Image quality enhanced")
    return enhanced

def save_feature_crop(image: np.ndarray, 
                     feature_name: str, 
                     coordinates: Tuple[int, int, int, int],
                     output_dir: str = "segmented_output") -> str:
    """
    Save individual feature crop to disk
    
    Args:
        image: Source image
        feature_name: Name of the feature
        coordinates: (x, y, width, height) tuple
        output_dir: Output directory
        
    Returns:
        Path to saved crop
    """
    x, y, w, h = coordinates
    
    # Validate coordinates
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        logger.warning(f"⚠️ Coordinates out of bounds for {feature_name}: ({x}, {y}, {w}, {h})")
        # Adjust coordinates to be within bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
    
    # Extract region
    crop = image[y:y+h, x:x+w]
    
    if crop.size == 0:
        raise ValueError(f"Empty crop for {feature_name} with coordinates ({x}, {y}, {w}, {h})")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save crop
    output_path = os.path.join(output_dir, f"{feature_name}.jpg")
    
    # Convert back to BGR for OpenCV save if needed
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, crop_bgr)
    else:
        cv2.imwrite(output_path, crop)
    
    logger.info(f"💾 Saved {feature_name} crop: {output_path} ({crop.shape})")
    return output_path

def extract_region(image: np.ndarray, 
                  center_x: float, 
                  center_y: float, 
                  width_ratio: float, 
                  height_ratio: float) -> np.ndarray:
    """
    Extract region from image using relative coordinates
    
    Args:
        image: Source image
        center_x: X center coordinate (0-1)
        center_y: Y center coordinate (0-1)
        width_ratio: Width as ratio of image width (0-1)
        height_ratio: Height as ratio of image height (0-1)
        
    Returns:
        Extracted region
    """
    img_height, img_width = image.shape[:2]
    
    # Calculate absolute coordinates
    width_px = int(width_ratio * img_width)
    height_px = int(height_ratio * img_height)
    center_x_px = int(center_x * img_width)
    center_y_px = int(center_y * img_height)
    
    # Calculate bounding box
    x1 = max(0, center_x_px - width_px // 2)
    y1 = max(0, center_y_px - height_px // 2)
    x2 = min(img_width, center_x_px + width_px // 2)
    y2 = min(img_height, center_y_px + height_px // 2)
    
    return image[y1:y2, x1:x2]

def calculate_image_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate similarity between two images using structural similarity
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        Similarity score (0-1)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        logger.warning("scikit-image not available, using basic MSE")
        return calculate_basic_similarity(image1, image2)
    
    # Resize images to same size if needed
    if image1.shape != image2.shape:
        min_height = min(image1.shape[0], image2.shape[0])
        min_width = min(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (min_width, min_height))
        image2 = cv2.resize(image2, (min_width, min_height))
    
    # Convert to grayscale for SSIM
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = image1
        gray2 = image2
    
    # Calculate SSIM
    score, _ = ssim(gray1, gray2, full=True)
    return max(0.0, min(1.0, score))

def calculate_basic_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate basic similarity using Mean Squared Error
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        Similarity score (0-1)
    """
    # Resize images to same size
    if image1.shape != image2.shape:
        min_height = min(image1.shape[0], image2.shape[0])
        min_width = min(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (min_width, min_height))
        image2 = cv2.resize(image2, (min_width, min_height))
    
    # Calculate MSE
    mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
    
    # Convert MSE to similarity score (lower MSE = higher similarity)
    max_mse = 255.0 ** 2  # Maximum possible MSE for 8-bit images
    similarity = 1.0 - (mse / max_mse)
    
    return max(0.0, min(1.0, similarity))