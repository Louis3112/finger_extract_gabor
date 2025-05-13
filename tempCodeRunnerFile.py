import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage import img_as_float
from scipy.ndimage import gaussian_filter
from skimage.io import imread
import matplotlib.pyplot as plt

def compute_structure_tensor(img, sigma=1.0):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    Axx = gaussian_filter(Ix**2, sigma)
    Axy = gaussian_filter(Ix*Iy, sigma)
    Ayy = gaussian_filter(Iy**2, sigma)

    return Axx, Axy, Ayy

def coherence_diffusion(img, iterations=15, sigma=1.5, gamma=0.2):
    # Parameter validation
    if iterations < 1:
        raise ValueError("Iterations must be positive")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    if gamma <= 0:
        raise ValueError("Gamma must be positive")

    # Image conversion
    img = img_as_float(img)
    if img.ndim == 3:
        img = rgb2gray(img)

    try:
        u = img.copy()
        for _ in range(iterations):
            Axx, Axy, Ayy = compute_structure_tensor(u, sigma)

            # Eigen decomposition
            trace = Axx + Ayy
            sqrt_term = np.sqrt((Axx - Ayy)**2 + 4*Axy**2)
            lambda1 = trace / 2 + sqrt_term / 2
            lambda2 = trace / 2 - sqrt_term / 2

            # Coherence (used to guide diffusion)
            coherence = (lambda1 - lambda2)**2 / (lambda1 + lambda2 + 1e-12)**2

            # Apply diffusion guided by coherence
            Ix = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)

            diffusion = gamma * (Ix * (1 - coherence) + Iy * (1 - coherence))
            u += diffusion
        return np.clip(u, 0, 1)
    except Exception as e:
        raise RuntimeError(f"Error during diffusion: {str(e)}")

# Usage
try:
    img = cv2.imread('test.jpg')
    filtered_img = coherence_diffusion(img)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Coherence Diffusion")
    plt.imshow(filtered_img, cmap='gray')
    plt.axis('off')
    plt.show()
except FileNotFoundError:
    print("Error: Image file not found")
except Exception as e:
    print(f"An error occurred: {str(e)}")