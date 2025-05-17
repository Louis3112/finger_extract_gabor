import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk

def segment_image(image, block_size=18, threshold=50):
    height = image.shape[0]
    width = image.shape[1]
    
    # Determine number of blocks
    num_block_rows = height // block_size
    num_block_cols = width // block_size
    
    segmented_image = np.zeros((height, width), dtype=np.uint8)
    
    for m in range(num_block_rows):
        for n in range(num_block_cols):
            # Extract the block
            block = image[m*block_size:(m+1)*block_size, n*block_size:(n+1)*block_size]
            
            # Calculate mean of the block
            block_sum = np.sum(block)
            mean = block_sum / (block_size * block_size)
            
            # Calculate variance of the block
            variance_sum = np.sum((block - mean) ** 2)
            variance = variance_sum / (block_size * block_size)
            
            # Check variance threshold
            if variance > threshold:
                # Set the segmented image part to the block's original values
                segmented_image[m*block_size:(m+1)*block_size, n*block_size:(n+1)*block_size] = block
    
    return segmented_image

# Initialize Tkinter
root = tk.Tk()
root.withdraw()  # Hide the root window

file_path = filedialog.askopenfilename()

if file_path:
    fingerprint_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    segmented = segment_image(fingerprint_image, block_size=16, threshold=150)

    cv2.imshow("Segmented Image", segmented)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()  # Close all windows