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

def estimate_blur_score(image: np.ndarray) -> float:
    """Estimate image sharpness; lower scores usually mean blurrier text."""
    if image is None:
        raise ValueError("Input image is None")

    gray = to_grayscale(image)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def upscale_for_ocr(image: np.ndarray, target_width: int = 1200, max_width: int = 1800) -> np.ndarray:
    """Upscale smaller or blurry images before OCR to improve character detail."""
    if image is None:
        raise ValueError("Input image is None")

    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image dimensions")

    scale = target_width / float(w)
    if scale < 1.0:
        scale = 1.0

    new_width = min(int(w * scale), max_width)
    new_height = max(1, int(h * (new_width / float(w))))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

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
        
    blur_score = estimate_blur_score(image)
    h, w = image.shape[:2]

    # Low-resolution or blurry documents benefit from a larger OCR input size.
    if blur_score < 90.0 or min(h, w) < 900:
        resized = upscale_for_ocr(image, target_width=max(resize_width, 1200), max_width=1800)
    else:
        new_h = int(resize_width * h / w)
        resized = cv2.resize(image, (resize_width, new_h), interpolation=cv2.INTER_AREA if resize_width < w else cv2.INTER_CUBIC)

    # Convert to grayscale, then lightly enhance blurry text regions.
    gray = to_grayscale(resized)
    if blur_score < 90.0:
        gray = enhance_contrast(gray)
        gray = sharpen_image(gray)

    three_channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return three_channel
