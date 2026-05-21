import cv2
import numpy as np

def resize_image(image: np.ndarray, width: int = 1000) -> np.ndarray:
    """
    Resize image to a target width while maintaining aspect ratio.
    
    Args:
        image (np.ndarray): Input OpenCV image.
        width (int): Target width in pixels.
        
    Returns:
        np.ndarray: Resized image.
    """
    if image is None:
        raise ValueError("Input image is None")
        
    h, w = image.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    
    resized = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB or BGR image to grayscale.
    
    Args:
        image (np.ndarray): Input image.
        
    Returns:
        np.ndarray: Grayscale image.
    """
    if image is None:
        raise ValueError("Input image is None")
        
    if len(image.shape) == 2:
        # Already grayscale
        return image
        
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def reduce_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian Blur to reduce high frequency noise.
    
    Args:
        image (np.ndarray): Input image.
        kernel_size (int): Size of Gaussian kernel (must be odd).
        
    Returns:
        np.ndarray: Blurred image.
    """
    if image is None:
        raise ValueError("Input image is None")
        
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd
        
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_threshold(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """
    Apply thresholding to binarize the image.
    
    Args:
        image (np.ndarray): Input image (should be grayscale).
        method (str): 'otsu' or 'adaptive'.
        
    Returns:
        np.ndarray: Thresholded binarized image.
    """
    if image is None:
        raise ValueError("Input image is None")
        
    # Ensure image is grayscale
    gray = to_grayscale(image)
    
    if method == 'otsu':
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Fallback to standard simple thresholding if method unknown
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
    return thresh

def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    Sharpen blurry images to improve OCR accuracy.
    
    Args:
        image (np.ndarray): Input grayscale image.
        
    Returns:
        np.ndarray: Sharpened image.
    """
    if image is None:
        raise ValueError("Input image is None")
    
    # Unsharp mask for sharpening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.subtract(image, gaussian)
    sharpened = cv2.add(image, sharpened)
    return sharpened

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance contrast for better text visibility.
    
    Args:
        image (np.ndarray): Input grayscale image.
        
    Returns:
        np.ndarray: Contrast-enhanced image.
    """
    if image is None:
        raise ValueError("Input image is None")
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced

def preprocess_for_ocr(image_path_or_arr, resize_width: int = 512) -> np.ndarray:
    """
    ULTRA-FAST preprocessing for OCR - speed optimized.
    Minimal processing to maximize speed (3-5 seconds cached).
    
    Args:
        image_path_or_arr: Path to image file or numpy image array.
        resize_width (int): Target width (512px for maximum speed).
        
    Returns:
        np.ndarray: Preprocessed 3-channel image.
    """
    if isinstance(image_path_or_arr, str):
        image = cv2.imread(image_path_or_arr)
        if image is None:
            raise FileNotFoundError(f"Could not load image from path: {image_path_or_arr}")
    else:
        image = image_path_or_arr
        
    # Fastest possible: Direct resize
    h, w = image.shape[:2]
    new_h = int(resize_width * h / w)
    # Use fastest interpolation
    resized = cv2.resize(image, (resize_width, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to grayscale and back to 3-channel (required by PaddleOCR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    three_channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return three_channel
