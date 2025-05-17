import numpy as np
import cv2
from scipy.ndimage import uniform_filter

def normalize_image(image, window_size=5, mean0=0, var_scale=1.0, M_threshold=None):
    """
    Normalize an image using the G(i,j) equation with conditional processing
    """
    # Convert to float for calculations
    img_float = image.astype(np.float32)
    
    # Calculate local mean
    mean_local = uniform_filter(img_float, size=window_size)
    
    # Calculate local variance
    diff_squared = (img_float - mean_local)**2
    local_var = uniform_filter(diff_squared, size=window_size)
    
    # Create the result array
    normalized = np.zeros_like(img_float)
    
    # Calculate f(i,j) for the conditional part if threshold is provided
    if M_threshold is not None:
        # Example f(i,j): gradient magnitude
        f_ij = cv2.Sobel(img_float, cv2.CV_32F, 1, 1)
        
        # Apply different formulas based on condition
        # For pixels where f(i,j) > M
        mask = f_ij > M_threshold
        normalized[mask] = mean0 + np.sqrt(local_var[mask] / var_scale)
        
        # For remaining pixels (otherwise condition)
        normalized[~mask] = mean0 + np.sqrt(local_var[~mask] / var_scale)
    else:
        # If no threshold specified, apply the same formula to all pixels
        normalized = mean0 + np.sqrt(local_var / var_scale)
    
    # Scale back to original range
    min_val, max_val = np.min(normalized), np.max(normalized)
    if max_val > min_val:  # Avoid division by zero
        normalized = ((normalized - min_val) / (max_val - min_val)) * 255.0
    
    return normalized.astype(np.uint8)

# Example usage
if __name__ == "__main__":
    # Load image
    image = cv2.imread('D:/Work/Code/python/gabor_filter_fingerprint/finger_extract_gabor/DB2_B/109_5.tif', cv2.IMREAD_GRAYSCALE)
    
    # Apply normalization
    normalized_image = normalize_image(
        image, 
        window_size=5,
        mean0=128,
        var_scale=5.0,
        M_threshold=50  # Now actually used in the computation
    )
    
    cv2.imshow('Original', image)
    cv2.imshow('Normalized', normalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()