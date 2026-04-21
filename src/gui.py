import os
import threading
import tkinter as tk
from tkinter import filedialog
from typing import Optional, Tuple

import customtkinter as ctk
from PIL import Image

from src.engine import FaceEngine

class ModernGUI(ctk.CTk):
    """
    Modern Graphical User Interface using CustomTkinter.
    Features dark mode, thumbnail previews, multithreading for inference,
    and visual loading indicators.
    """

    def __init__(self):
        super().__init__()

        self.engine = FaceEngine()
        self.img1_path: Optional[str] = None
        self.img2_path: Optional[str] = None

        # --- Window Setup ---
        self.title("Face Similarity Pro")
        self.geometry("800x600")
        self.minsize(700, 550)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- Layout ---
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        self.header_label = ctk.CTkLabel(
            self, 
            text="Enterprise Face Similarity Analysis", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.header_label.grid(row=0, column=0, pady=(20, 10))

        # Main Content Frame (holds both image zones)
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=1)

        # --- Image Zone 1 ---
        self.zone1_frame = ctk.CTkFrame(self.content_frame)
        self.zone1_frame.grid(row=0, column=0, sticky="nsew", padx=10)
        self.zone1_frame.grid_rowconfigure(1, weight=1)
        self.zone1_frame.grid_columnconfigure(0, weight=1)
        
        self.zone1_label = ctk.CTkLabel(self.zone1_frame, text="Upload Image 1", font=ctk.CTkFont(size=16))
        self.zone1_label.grid(row=0, column=0, pady=10)
        
        self.img1_display = ctk.CTkLabel(self.zone1_frame, text="No Image Selected")
        self.img1_display.grid(row=1, column=0, pady=10)
        
        self.btn_upload1 = ctk.CTkButton(self.zone1_frame, text="Select File...", command=lambda: self.upload_image(1))
        self.btn_upload1.grid(row=2, column=0, pady=20)

        # --- Image Zone 2 ---
        self.zone2_frame = ctk.CTkFrame(self.content_frame)
        self.zone2_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        self.zone2_frame.grid_rowconfigure(1, weight=1)
        self.zone2_frame.grid_columnconfigure(0, weight=1)
        
        self.zone2_label = ctk.CTkLabel(self.zone2_frame, text="Upload Image 2", font=ctk.CTkFont(size=16))
        self.zone2_label.grid(row=0, column=0, pady=10)
        
        self.img2_display = ctk.CTkLabel(self.zone2_frame, text="No Image Selected")
        self.img2_display.grid(row=1, column=0, pady=10)
        
        self.btn_upload2 = ctk.CTkButton(self.zone2_frame, text="Select File...", command=lambda: self.upload_image(2))
        self.btn_upload2.grid(row=2, column=0, pady=20)

        # --- Actions & Status ---
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.btn_run = ctk.CTkButton(
            self.status_frame, 
            text="Run Comparison", 
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.start_comparison,
            height=40
        )
        self.btn_run.grid(row=0, column=0, pady=10)

        self.progressbar = ctk.CTkProgressBar(self.status_frame, mode="indeterminate")
        self.progressbar.grid(row=1, column=0, pady=(0, 10), sticky="ew")
        self.progressbar.set(0)
        self.progressbar.grid_remove() # Hide initially

        self.result_label = ctk.CTkLabel(
            self.status_frame, 
            text="", 
            font=ctk.CTkFont(size=18),
            wraplength=700
        )
        self.result_label.grid(row=2, column=0, pady=(0, 20))

        # Disable UI and load models
        self.set_ui_state("disabled")
        self.result_label.configure(text="Initializing ML Models... Please wait.", text_color="yellow")
        self.progressbar.grid()
        self.progressbar.start()
        
        # Start initialization thread
        threading.Thread(target=self._init_models_thread, daemon=True).start()

    def _init_models_thread(self):
        """Background thread to initialize heavy ML models on startup."""
        try:
            self.engine.initialize_models()
            self.after(0, self._on_models_ready)
        except Exception as e:
            self.after(0, self._on_init_error, str(e))

    def _on_models_ready(self):
        """Callback when models are loaded."""
        self.progressbar.stop()
        self.progressbar.grid_remove()
        self.result_label.configure(text="", text_color="white")
        self.set_ui_state("normal")

    def _on_init_error(self, error_msg: str):
        """Callback if model initialization fails."""
        self.progressbar.stop()
        self.result_label.configure(text=f"Initialization Error: {error_msg}", text_color="red")

    def set_ui_state(self, state: str):
        """Enable or disable interactive UI elements."""
        self.btn_upload1.configure(state=state)
        self.btn_upload2.configure(state=state)
        self.btn_run.configure(state=state)

    def upload_image(self, zone: int):
        """Opens native file picker and updates the UI with a thumbnail."""
        file_path = filedialog.askopenfilename(
            title=f"Select Image {zone}",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not file_path:
            return

        try:
            # Generate thumbnail using PIL
            img = Image.open(file_path)
            
            # Using CTkImage to hold the PIL image for high DPI support
            ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(250, 250))

            if zone == 1:
                self.img1_path = file_path
                self.img1_display.configure(image=ctk_image, text="")
                self.img1_display.image = ctk_image  # Keep reference
            else:
                self.img2_path = file_path
                self.img2_display.configure(image=ctk_image, text="")
                self.img2_display.image = ctk_image  # Keep reference
                
        except Exception as e:
            self.result_label.configure(text=f"Error loading image: {str(e)}", text_color="red")

    def start_comparison(self):
        """Validates inputs and starts the comparison thread."""
        if not self.img1_path or not self.img2_path:
            self.result_label.configure(text="Please upload both images before running comparison.", text_color="yellow")
            return

        # Prepare UI for processing
        self.set_ui_state("disabled")
        self.result_label.configure(text="Processing... Detecting and comparing faces...", text_color="cyan")
        self.progressbar.grid()
        self.progressbar.start()

        # Run inference in a daemon thread
        threading.Thread(
            target=self._compare_thread, 
            args=(self.img1_path, self.img2_path), 
            daemon=True
        ).start()

    def _compare_thread(self, path1: str, path2: str):
        """Background thread executing the deepface inference."""
        result = self.engine.compare_images(path1, path2)
        self.after(0, self._on_comparison_complete, result)

    def _on_comparison_complete(self, result: dict):
        """Callback invoked when ML inference finishes."""
        self.progressbar.stop()
        self.progressbar.grid_remove()
        self.set_ui_state("normal")

        if result.get("error"):
            self.result_label.configure(text=f"Error: {result['error']}", text_color="red")
            return

        score = result["score"]
        is_match = result["match"]

        if is_match:
            status_text = "are"
            color = "#00FF00"  # Bright green
        else:
            status_text = "are not"
            color = "#FF4444"  # Red
            
        # The strict output structure required
        output_msg = (
            f"Face similarity ratio: {score}%, "
            f"The two photos {status_text} the same person."
        )

        self.result_label.configure(text=output_msg, text_color=color)

def run_gui():
    app = ModernGUI()
    app.mainloop()

if __name__ == "__main__":
    run_gui()
