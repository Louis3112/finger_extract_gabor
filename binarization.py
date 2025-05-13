import cv2
import numpy as np
import matplotlib.pyplot as plt

def basic_binarization(image_path, threshold_value=127):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply basic thresholding
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(binary_image, cmap='gray'), plt.title('Binary Image')
    plt.tight_layout()
    plt.show()
    
    return binary_image


# Example usage
binary_img = adaptive_binarization('test2.jpeg', 127)