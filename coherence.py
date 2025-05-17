import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Create a sample noisy image (or load one)
np.random.seed(0)
image_size = 100
original_image = np.zeros((image_size, image_size))
original_image[20:80, 20:80] = 1 # A white square
noisy_image = original_image + 0.3 * np.random.randn(image_size, image_size)
noisy_image = np.clip(noisy_image, 0, 1) # Ensure values are in [0,1]

# Apply Gaussian filter
sigma_value = 2.0 # Try changing this value (e.g., 0.5, 1.0, 3.0)
smoothed_image_scipy = ndimage.gaussian_filter(noisy_image, sigma=sigma_value)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Clean Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(smoothed_image_scipy, cmap='gray')
plt.title(f'Gaussian Filtered (SciPy, sigma={sigma_value})')
plt.axis('off')

plt.tight_layout()
plt.show()