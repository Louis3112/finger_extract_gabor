import numpy as np
from scipy import fftpack
import cv2

def log_gabor(img, wavelength, orientation, bandwidth=0.5):
    rows, cols = img.shape
    radius = np.fft.fftshift(np.fft.fftfreq(cols)[np.newaxis, :]**2 + 
                            np.fft.fftfreq(rows)[:, np.newaxis]**2)
    radius = np.sqrt(radius)
    radius[0, 0] = 1  # Avoid division by zero
    
    # Radial component (log-Gabor)
    log_gabor = np.exp(-(np.log(radius * wavelength))**2 / 
                     (2 * np.log(bandwidth)**2))
    log_gabor[0, 0] = 0  # Set DC component to zero
    
    # Angular component
    theta = np.arctan2(-np.fft.fftfreq(rows)[:, np.newaxis],
                       np.fft.fftfreq(cols)[np.newaxis, :])
    spread = np.exp(-2 * (theta - orientation)**2 / (np.pi/4)**2)
    
    # Combine components
    filter = log_gabor * spread
    return filter
  
img = cv2.imread('test.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

normalized = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

_ , thresh = cv2.threshold(normalized, np.mean(normalized), 255, cv2.THRESH_BINARY)

imgResized = cv2.resize(thresh, (960, 540))

# Apply Gabor filter
gabor_filter = log_gabor(imgResized, 0.5, np.pi/4)

cv2.imshow('Gabor Filter', gabor_filter)