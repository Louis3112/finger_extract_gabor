import numpy as np
from tkinter import filedialog
import cv2

def local_variance(image, window_size):
    """
    Calculate local variance for each pixel in the image using the formula:
    
    var(m,n) = ∑(i=1 to w) ∑(j=1 to w) [g(i,j) - mean]² / (w × w)
    
    Args:
        image: Input grayscale image as a 2D numpy array
        window_size: Size of the square window (w)
    
    Returns:
        2D numpy array containing variance at each pixel
    """
    # Get image dimensions
    height, width = image.shape
    
    # Pad the image to handle boundary conditions
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    
    # Initialize output variance map
    variance_map = np.zeros((height, width))
    
    # For each pixel position in the original image
    for m in range(height):
        for n in range(width):
            # Extract the window centered at (m,n)
            window = padded_image[m:m+window_size, n:n+window_size]
            
            # Calculate mean of the window
            window_mean = np.mean(window)
            
            # Calculate the variance using the formula
            sum_squared_diff = 0
            for i in range(window_size):
                for j in range(window_size):
                    # g(i,j) - mean, squared
                    sum_squared_diff += (window[i, j] - window_mean)**2
            
            # Divide by window size to get variance
            variance_map[m, n] = sum_squared_diff / (window_size * window_size)
    
    return variance_map

# Example usage
def main():
    file_path = filedialog.askopenfilename()
    
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate local variance with window size 5
    variance_map = local_variance(image, window_size=5)
    
    # If you want to visualize (requires matplotlib):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(122)
    plt.imshow(variance_map, cmap='viridis')
    plt.title('Local Variance Map')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()