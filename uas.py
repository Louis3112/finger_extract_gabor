import customtkinter
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import ast
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import img_as_float
from scipy.ndimage import gaussian_filter
from scipy import fft

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

class identityFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_rowconfigure((0,1,2), weight=1)
        self.grid_columnconfigure((0,1), weight=1)

        self.title = customtkinter.CTkLabel(self, text="Fingerprint Enchancement with Multiple Filters", font=("Helvetica", 22, "bold"))
        self.title.grid(row = 0, column = 0, columnspan = 2, padx = 0, pady = 10, sticky = "nsew")

        self.louis = customtkinter.CTkLabel(self, text="Cornelius Louis Nathan", font=("Helvetica", 12, "bold"))
        self.louis.grid(row = 1, column = 0, padx = 0, pady = 0, sticky = "nsew")
        self.louisNIM = customtkinter.CTkLabel(self, text="TI 2023 C - 23051204085", font=("Helvetica", 12))
        self.louisNIM.grid(row = 2, column = 0, padx = 0, pady = 0, sticky = "nsew")

        self.noel = customtkinter.CTkLabel(self, text="Adriano Emmanuel", font=("Helvetica", 12, "bold"))
        self.noel.grid(row = 1, column = 1, padx = 0, pady = 0, sticky = "nsew")        
        self.noelNIM = customtkinter.CTkLabel(self, text="TI 2023 C - 23051204082", font=("Helvetica", 12))
        self.noelNIM.grid(row = 2, column = 1, padx = 0, pady = 0, sticky = "nsew")

class equationFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.grid_rowconfigure((0,1), weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.title = customtkinter.CTkLabel(self, text="The Required Steps", font=("Helvetica", 18, "bold"))
        self.title.grid(row = 0, column = 0, padx = 0, pady = 10, sticky="nsew")

        try:
            eq = Image.open("eq.png")
            eq = eq.resize((800, 300), Image.Resampling.LANCZOS)
            self.eq = customtkinter.CTkImage(eq, size=(800, 300))
            self.label = customtkinter.CTkLabel(master = self, image=self.eq, text="")
            self.label.grid(row = 1, column = 0, padx = 0, pady = (10,0), sticky="nsew")
        except FileNotFoundError:
            print("Error: 'eq.png' not found. Please ensure the file is in the correct directory.")
        
class imageFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.processed_image = None
        self.original_image = None

        self.max_width = 300
        self.max_height = 300
        
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_columnconfigure((0, 1), weight=1)
        
        # Original image frame
        self.orig_title = customtkinter.CTkLabel(self, text="Original Image")
        self.orig_title.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="s")
        
        # Create the image label for original image
        self.image_label = customtkinter.CTkLabel(self, text="No Image")
        self.image_label.grid(row=1, column=0, padx=10, pady=5, sticky="n")
        
        # Processed image frame title
        self.processed_title = customtkinter.CTkLabel(self, text="Processed Image")
        self.processed_title.grid(row=0, column=1, padx=10, pady=(10, 5), sticky="s")
        
        # Image frame after filtering
        self.processed_label = customtkinter.CTkLabel(self, text="No Image")
        self.processed_label.grid(row=1, column=1, padx=10, pady=5, sticky="n")
        
        # Buttons row
        self.button_frame = customtkinter.CTkFrame(self)
        self.button_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.button_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Upload Button
        self.upload_btn = customtkinter.CTkButton(self.button_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=5, pady=10, sticky="ew")
        
        # Convert Button
        self.convert_btn = customtkinter.CTkButton(self.button_frame, text="Convert", command=self.process_image)
        self.convert_btn.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = Image.open(file_path)
            
            # Store the original image for processing
            self.img = self.original_image.copy()
            
            # Resize with aspect ratio for display
            display_img = self.original_image.copy()
            display_img.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)

             # Convert to CTkImage (customtkinter's image format)
            self.ctk_image = customtkinter.CTkImage(display_img, size=(display_img.width, display_img.height))
            
            # Update the image label
            self.image_label.configure(image=self.ctk_image, text="")
            
            # Reset processed image
            self.processed_label.configure(image=None, text="No processed image")
            self.processed_image = None
    
    def process_image(self):
        if not hasattr(self, 'img') or self.img is None:
            print("Please upload an image first")
            return
            
        try:
            # Define default parameters for processing
            filter_params = {
                "kernel": 80,
                "sigma": 1,
                "theta": 1 * np.pi/2,
                "lambda": 1000,
                "gamma": 0.5,
                "phi": 0
            }
            
            normalized = self.normalize(np.array(self.img))
            
            segmented = self.segmentation(normalized)
            
            filtered = self.coherence_diffusion_filter(segmented, filter_params["sigma"])
            
            final_image = self.binarization(filtered)
            
            # Convert the result back to PIL Image for display
            self.processed_image = Image.fromarray(final_image)
            
            # Resize for display
            display_img = self.processed_image.copy()
            display_img.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
            
            # Create CTkImage for display
            self.processed_ctk_image = customtkinter.CTkImage(display_img, size=(display_img.width, display_img.height))
            
            # Update the processed image label
            self.processed_label.configure(image=self.processed_ctk_image, text="")
            
            print("Image processing complete!")

        except Exception as e:
            print(f"Error processing image: {e}")
    
    def normalize(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized_img.astype(np.uint8)
    
    # Create segmentation image from normalized image
    def segmentation(self, img):

        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    # Apply coherence diffusion filter after segmentation
    def coherence_diffusion_filter(self, img, sigma=3):

        filtered_img = gaussian_filter(img, sigma=sigma)
        
        return filtered_img.astype(np.uint8)
    
    # Apply Log Gabor Filter after Coherence Diffusion Filter
    def log_gabor_filter(self, img, wavelength=10, orientation=np.pi/4, bandwidth=0.5):
        img_float = img_as_float(img)
        
        img_fft = fft.fft2(img_float)
        img_fft_shifted = fft.fftshift(img_fft)
        
        rows, cols = img.shape
        y, x = np.mgrid[-rows//2:rows//2, -cols//2:cols//2]
        
        radius = np.sqrt(x**2 + y**2)
        radius[rows//2, cols//2] = 1  
        
        theta = np.arctan2(y, x)
        
        log_gabor = np.exp(-(np.log(radius / wavelength))**2 / (2 * np.log(bandwidth)**2))
        log_gabor[rows//2, cols//2] = 0
        
        
        angular = np.exp(-((theta - orientation) % np.pi)**2 / (2 * (np.pi/4)**2))
        
        filt = log_gabor * angular
        
        filtered_fft = img_fft_shifted * filt
        
        filtered_img = fft.ifft2(fft.ifftshift(filtered_fft))
        filtered_img = np.abs(filtered_img)
        
        filtered_img = 255 * (filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min())
        
        return filtered_img.astype(np.uint8)
    
    # Apply Binarization after Log Gabor Filter
    def binarization(self, image, threshold_value=127):
        binary_image = cv2.adaptiveThreshold(image.astype(np.uint8), 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        
        return binary_image

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Fingerprint Extraction with Gabor Filter")

        for i in range(3):
            self.grid_rowconfigure(i, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Row 1
        self.identityFr = identityFrame(self)
        self.identityFr.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")

        # Row 2
        self.eqFr = equationFrame(self)
        self.eqFr.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nsew")
        
        # Row 3 
        self.imageProcessingFrame = imageFrame(self)
        self.imageProcessingFrame.grid(row=2, column=0, padx=10, pady=(10, 10), sticky="nsew")
         
if __name__ == "__main__":
    app = App()
    app.mainloop()