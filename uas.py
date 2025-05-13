import customtkinter
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import ast
import cv2
from skimage.color import rgb2gray
from skimage import img_as_float
from scipy.ndimage import gaussian_filter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

class identityFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_rowconfigure((0,1,2), weight=1)
        self.grid_columnconfigure((0,1), weight=1)

        self.title = customtkinter.CTkLabel(self, text="Fingerprint Extraction with Gabor Filter", font=("Helvetica", 24, "bold"))
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

        self.title = customtkinter.CTkLabel(self, text="Gabor Equation", font=("Helvetica", 18, "bold"))
        self.title.grid(row = 0, column = 0, padx = 0, pady = 10, sticky="nsew")

        try:
            eq = Image.open("eq.png")
            eq = eq.resize((700, 160), Image.Resampling.LANCZOS)
            self.eq = customtkinter.CTkImage(eq, size=(700, 160))
        except FileNotFoundError:
            print("Error: 'eq.png' not found. Please ensure the file is in the correct directory.")

        self.label = customtkinter.CTkLabel(master = self, image = self.eq, text="")
        self.label.grid(row = 1, column = 0, padx = 0, pady = (10,0), sticky="nsew")

class parameterFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.grid_rowconfigure((0,1,2,3), weight = 1)
        for i in range(6):
            self.grid_columnconfigure(i, weight=1)

        self.title = customtkinter.CTkLabel(self, text= "Parameter Setting for Gabor Filter", font=("Helvetica", 18, "bold"))
        self.title.grid(row = 0, column = 0, columnspan = 6, padx = 0, pady = 10, sticky = "nsew")

        # Input Kernel
        self.kernel_label = customtkinter.CTkLabel(self, text="Kernel Size")
        self.kernel_label.grid(row = 1, column = 0, padx = (10,5), pady = (0, 10))
        self.kernel = customtkinter.CTkEntry(self,placeholder_text="ex : 50")
        self.kernel.grid(row = 2, column = 0, padx = (10,5), pady = 0)

        # Input Sigma
        self.sigma_label = customtkinter.CTkLabel(self, text="Sigma (σ)")
        self.sigma_label.grid(row = 1, column = 1, padx = 5, pady = (0, 10))
        self.sigma = customtkinter.CTkEntry(self,placeholder_text="ex : 3")
        self.sigma.grid(row = 2, column = 1, padx = 5, pady = 0)
 
        # Input Theta
        self.theta_label = customtkinter.CTkLabel(self, text="Theta (θ)")
        self.theta_label.grid(row = 1, column = 2, padx = 5, pady = (0, 10))
        self.theta = customtkinter.CTkEntry(self,placeholder_text="ex : 1*np.pi/4")
        self.theta.grid(row = 2, column = 2, padx = 5, pady = 0)

        # Input Lambda
        self.lambda_label = customtkinter.CTkLabel(self, text="Lambda (λ)")
        self.lambda_label.grid(row = 1, column = 3, padx = 5, pady = (0, 10))
        self.lambdaa = customtkinter.CTkEntry(self,placeholder_text="ex : 1*np.pi/4")
        self.lambdaa.grid(row = 2, column = 3, padx = 5, pady = 0)

        # Input Gamma
        self.gamma_label = customtkinter.CTkLabel(self, text="Gamma (γ)")
        self.gamma_label.grid(row = 1, column = 4, padx=5, pady = (0, 10))
        self.gamma = customtkinter.CTkEntry(self,placeholder_text="ex : 0.5")
        self.gamma.grid(row = 2, column = 4, padx = 5, pady = 0)      

        # Input Phi
        self.phi_label = customtkinter.CTkLabel(self, text="Phi (φ)")
        self.phi_label.grid(row = 1, column = 5, padx = (5,10), pady = (0, 10))
        self.phi = customtkinter.CTkEntry(self,placeholder_text="ex : 0")
        self.phi.grid(row = 2, column = 5, padx = (5,10), pady = 0)

        self.submit_btn = customtkinter.CTkButton(self, text="Submit", command=self.validate_inputs)
        self.submit_btn.grid(row = 3, column = 0, columnspan = 6, pady = 20, sticky = "nsew")
    def validate_inputs(self):
        try:
            kernel_input = int(self.kernel.get())
            sigma_input = int(self.sigma.get())
            theta_input = ast.literal_eval(self.theta.get())
            lambda_input = ast.literal_eval(self.lambdaa.get())
            gamma_input = float(self.gamma.get())
            phi_input = float(self.phi.get())

            print("All inputs valid:")
            print(f"Kernel: {kernel_input}, Sigma: {sigma_input}, Theta: {theta_input}")
            print(f"Lambda: {lambda_input}, Gamma: {gamma_input}, Phi: {phi_input}")
        except Exception as e:
            print("Invalid input:", e)
            
class imageFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        self.max_width = 300
        self.max_height = 300
        
        # Configure grid
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create the image label FIRST
        self.image_label = customtkinter.CTkLabel(self, text="No Image")
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Image frame after filtering
        self.image_frame = customtkinter.CTkFrame(self, text="Image after filtering")
        self.image_frame.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")
        
        # Upload Button
        self.upload_btn = customtkinter.CTkButton(self, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row = 3, column = 0,  padx = 5, pady = 10, sticky = "nsew")
        

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img = Image.open(file_path)
            
            # Resize with aspect ratio (optional bounding)
            self.img.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)

             # Convert to CTkImage (customtkinter's image format)
            self.ctk_image = customtkinter.CTkImage(self.img, size=(self.img.width, self.img.height))
            
            # Update the image label
            self.image_label.configure(image=self.ctk_image, text="")
            
    def normalize(self, img):
        img = img.convert('L')
        
        # Normalize
        normalized_img = cv2.normalize(np.array(img), None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized_img
        
    
    def segmentation(self, img):
        _ , thresh = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)
        
        return thresh
        
        
    def coherence_diffusion_filter(self, img):
        # Convert to numpy array
        img = np.array(self.img)
        
        # Apply the filter
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, cv2.getGaborKernel((31, 31), 0.5, 0, 1, 0, 0, ktype=cv2.CV_32F))
        
        return filtered_img
        
    
    def log_gabor_filter(self):
        
        
    def binarization(self):
            
class imageFilterFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        self.grid_rowconfigure((0,1,2,3), weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.title = customtkinter.CTkLabel(self, text="Image Filtering", font=("Helvetica", 18, "bold"))
        self.title.grid(row = 0, column = 0, padx = 0, pady = 10, sticky="nsew")
        
    
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Fingerprint Extraction with Gabor Filter")

        for i in range(5):
            self.grid_rowconfigure(i, weight = 1)
        self.grid_columnconfigure(0, weight = 1)

        # Row 1
        self.identityFr = identityFrame(self)
        self.identityFr.grid(row = 0, column = 0, padx = 10, pady = (10, 0), sticky = "nsew")

        # Row 2
        self.eqFr = equationFrame(self)
        self.eqFr.grid(row = 1, column = 0, padx = 10, pady = (10, 0), sticky = "nsew")
        
        # Row 3
        self.paramFr = parameterFrame(self)
        self.paramFr.grid(row = 2, column = 0, padx = 10, pady = (10, 0), sticky = "nsew")
        
        # Upload Image and Preview
        self.uploadImg = imageFrame(self)
        self.uploadImg.grid(row = 4, column = 0, padx = 10, pady = (15, 0), sticky = "nsew")     
        
        # Image after filtering
        self.filteredImg = imageFrame(self)
        self.filteredImg.grid(row = 5, column = 0, padx = 10, pady = (15, 0), sticky = "nsew")
        
        

app = App()
app.mainloop()