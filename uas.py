import customtkinter
from PIL import Image, ImageTk
import tkinter

class EquationFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        eq = Image.open("uas/eq.png")
        self.eq = customtkinter.CTkImage(eq, size=(800, 200))

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(master = self, image = self.eq, text="")
        self.label.grid(row = 0, column = 0, padx = 0, pady = 0, sticky="nsew")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        self.title("Fingerprint Extraction with Gabor Filter")
        self.grid_rowconfigure((0, 1, 2 ,3, 4), weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Row 1
        self.EqFrame = EquationFrame(self)
        self.EqFrame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")

app = App()
app.mainloop()