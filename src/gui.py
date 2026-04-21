import os
import threading
from tkinter import filedialog
from typing import Optional

import customtkinter as ctk
from PIL import Image

from src.engine import FaceEngine


class ModernGUI(ctk.CTk):
    """
    Modern Graphical User Interface using CustomTkinter.
    Provides separate Similarity and Extraction workflows while running
    heavy ML operations in background daemon threads.
    """

    def __init__(self):
        super().__init__()

        self.engine = FaceEngine()
        self.img1_path: Optional[str] = None
        self.img2_path: Optional[str] = None
        self.extraction_src_path: Optional[str] = None
        self.extraction_out_path: Optional[str] = None

        self.title("Face Similarity Pro")
        self.geometry("900x680")
        self.minsize(780, 620)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.header_label = ctk.CTkLabel(
            self,
            text="Enterprise Face Similarity Analysis",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.header_label.grid(row=0, column=0, pady=(20, 10))

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.similarity_tab = self.tabview.add("Similarity")
        self.extraction_tab = self.tabview.add("Extraction")

        self._build_similarity_tab()
        self._build_extraction_tab()

        self.set_ui_state("disabled")
        self.sim_result_label.configure(text="Initializing ML Models... Please wait.", text_color="yellow")
        self.ext_result_label.configure(text="Initializing ML Models... Please wait.", text_color="yellow")
        self.sim_progressbar.grid()
        self.ext_progressbar.grid()
        self.sim_progressbar.start()
        self.ext_progressbar.start()

        threading.Thread(target=self._init_models_thread, daemon=True).start()

    def _build_similarity_tab(self):
        self.similarity_tab.grid_columnconfigure(0, weight=1)
        self.similarity_tab.grid_columnconfigure(1, weight=1)
        self.similarity_tab.grid_rowconfigure(1, weight=1)

        self.zone1_frame = ctk.CTkFrame(self.similarity_tab)
        self.zone1_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8), pady=10)
        self.zone1_frame.grid_rowconfigure(1, weight=1)
        self.zone1_frame.grid_columnconfigure(0, weight=1)

        self.zone1_label = ctk.CTkLabel(self.zone1_frame, text="Upload Image 1", font=ctk.CTkFont(size=16))
        self.zone1_label.grid(row=0, column=0, pady=10)
        self.img1_display = ctk.CTkLabel(self.zone1_frame, text="No Image Selected")
        self.img1_display.grid(row=1, column=0, pady=10)
        self.btn_upload1 = ctk.CTkButton(self.zone1_frame, text="Select File...", command=lambda: self.upload_image(1))
        self.btn_upload1.grid(row=2, column=0, pady=20)

        self.zone2_frame = ctk.CTkFrame(self.similarity_tab)
        self.zone2_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(8, 0), pady=10)
        self.zone2_frame.grid_rowconfigure(1, weight=1)
        self.zone2_frame.grid_columnconfigure(0, weight=1)

        self.zone2_label = ctk.CTkLabel(self.zone2_frame, text="Upload Image 2", font=ctk.CTkFont(size=16))
        self.zone2_label.grid(row=0, column=0, pady=10)
        self.img2_display = ctk.CTkLabel(self.zone2_frame, text="No Image Selected")
        self.img2_display.grid(row=1, column=0, pady=10)
        self.btn_upload2 = ctk.CTkButton(self.zone2_frame, text="Select File...", command=lambda: self.upload_image(2))
        self.btn_upload2.grid(row=2, column=0, pady=20)

        self.sim_controls = ctk.CTkFrame(self.similarity_tab, fg_color="transparent")
        self.sim_controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 16))
        self.sim_controls.grid_columnconfigure(0, weight=1)

        self.btn_run = ctk.CTkButton(
            self.sim_controls,
            text="Run Similarity Comparison",
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.start_comparison,
            height=40,
        )
        self.btn_run.grid(row=0, column=0, pady=8)

        self.sim_progressbar = ctk.CTkProgressBar(self.sim_controls, mode="indeterminate")
        self.sim_progressbar.grid(row=1, column=0, pady=(0, 10), sticky="ew")
        self.sim_progressbar.set(0)
        self.sim_progressbar.grid_remove()

        self.sim_result_label = ctk.CTkLabel(
            self.sim_controls,
            text="",
            font=ctk.CTkFont(size=18),
            wraplength=760,
        )
        self.sim_result_label.grid(row=2, column=0)

    def _build_extraction_tab(self):
        self.extraction_tab.grid_columnconfigure(0, weight=1)

        self.ext_frame = ctk.CTkFrame(self.extraction_tab)
        self.ext_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=12)
        self.ext_frame.grid_columnconfigure(0, weight=1)
        self.ext_frame.grid_rowconfigure(1, weight=1)

        self.ext_label = ctk.CTkLabel(self.ext_frame, text="Select Source Image", font=ctk.CTkFont(size=16))
        self.ext_label.grid(row=0, column=0, pady=(12, 8))

        self.ext_display = ctk.CTkLabel(self.ext_frame, text="No Source Image Selected")
        self.ext_display.grid(row=1, column=0, pady=(2, 10))

        self.btn_upload_extract = ctk.CTkButton(
            self.ext_frame,
            text="Select Source File...",
            command=self.upload_extraction_image,
        )
        self.btn_upload_extract.grid(row=2, column=0, pady=8)

        self.ext_output_label = ctk.CTkLabel(self.ext_frame, text="Output: (not selected yet)", wraplength=760)
        self.ext_output_label.grid(row=3, column=0, pady=(4, 10))

        self.btn_run_extract = ctk.CTkButton(
            self.ext_frame,
            text="Run Face Extraction",
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.start_extraction,
            height=40,
        )
        self.btn_run_extract.grid(row=4, column=0, pady=8)

        self.ext_progressbar = ctk.CTkProgressBar(self.ext_frame, mode="indeterminate")
        self.ext_progressbar.grid(row=5, column=0, pady=(0, 10), sticky="ew")
        self.ext_progressbar.set(0)
        self.ext_progressbar.grid_remove()

        self.ext_result_label = ctk.CTkLabel(
            self.ext_frame,
            text="",
            font=ctk.CTkFont(size=18),
            wraplength=760,
        )
        self.ext_result_label.grid(row=6, column=0, pady=(0, 12))

    def _init_models_thread(self):
        try:
            self.engine.initialize_models()
            self.after(0, self._on_models_ready)
        except Exception as e:
            self.after(0, self._on_init_error, str(e))

    def _on_models_ready(self):
        self.sim_progressbar.stop()
        self.ext_progressbar.stop()
        self.sim_progressbar.grid_remove()
        self.ext_progressbar.grid_remove()
        self.sim_result_label.configure(text="", text_color="white")
        self.ext_result_label.configure(text="", text_color="white")
        self.set_ui_state("normal")

    def _on_init_error(self, error_msg: str):
        self.sim_progressbar.stop()
        self.ext_progressbar.stop()
        self.sim_progressbar.grid_remove()
        self.ext_progressbar.grid_remove()
        self.sim_result_label.configure(text=f"Initialization Error: {error_msg}", text_color="red")
        self.ext_result_label.configure(text=f"Initialization Error: {error_msg}", text_color="red")

    def set_ui_state(self, state: str):
        self.btn_upload1.configure(state=state)
        self.btn_upload2.configure(state=state)
        self.btn_run.configure(state=state)
        self.btn_upload_extract.configure(state=state)
        self.btn_run_extract.configure(state=state)

    def upload_image(self, zone: int):
        file_path = filedialog.askopenfilename(
            title=f"Select Image {zone}",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")],
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path)
            ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(250, 250))

            if zone == 1:
                self.img1_path = file_path
                self.img1_display.configure(image=ctk_image, text="")
                self.img1_display.image = ctk_image
            else:
                self.img2_path = file_path
                self.img2_display.configure(image=ctk_image, text="")
                self.img2_display.image = ctk_image
        except Exception as e:
            self.sim_result_label.configure(text=f"Error loading image: {e}", text_color="red")

    def _next_extracted_path(self, source_path: str) -> str:
        directory = os.path.dirname(source_path)
        ext = os.path.splitext(source_path)[1] or ".png"
        first = os.path.join(directory, f"extracted{ext}")
        if not os.path.exists(first):
            return first

        idx = 2
        while True:
            candidate = os.path.join(directory, f"extracted{idx}{ext}")
            if not os.path.exists(candidate):
                return candidate
            idx += 1

    def upload_extraction_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image for Extraction",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")],
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path)
            ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(300, 300))
            self.extraction_src_path = file_path
            self.extraction_out_path = self._next_extracted_path(file_path)
            self.ext_display.configure(image=ctk_image, text="")
            self.ext_display.image = ctk_image
            self.ext_output_label.configure(text=f"Output: {os.path.basename(self.extraction_out_path)}")
        except Exception as e:
            self.ext_result_label.configure(text=f"Error loading image: {e}", text_color="red")

    def start_comparison(self):
        if not self.img1_path or not self.img2_path:
            self.sim_result_label.configure(
                text="Please upload both images before running comparison.",
                text_color="yellow",
            )
            return

        self.set_ui_state("disabled")
        self.sim_result_label.configure(text="Processing... Detecting and comparing faces...", text_color="cyan")
        self.sim_progressbar.grid()
        self.sim_progressbar.start()

        threading.Thread(
            target=self._compare_thread,
            args=(self.img1_path, self.img2_path),
            daemon=True,
        ).start()

    def _compare_thread(self, path1: str, path2: str):
        try:
            result = self.engine.compare_images(path1, path2)
        except Exception as e:
            result = {"match": False, "score": 0.0, "error": str(e)}
        self.after(0, self._on_comparison_complete, result)

    def _on_comparison_complete(self, result: dict):
        self.sim_progressbar.stop()
        self.sim_progressbar.grid_remove()
        self.set_ui_state("normal")

        if result.get("error"):
            self.sim_result_label.configure(text=f"Error: {result['error']}", text_color="red")
            return

        score = result["score"]
        is_match = result["match"]
        if is_match:
            status_text = "are"
            color = "#00FF00"
        else:
            status_text = "are not"
            color = "#FF4444"

        output_msg = (
            f"Face similarity ratio: {score}%, "
            f"The two photos {status_text} the same person."
        )
        self.sim_result_label.configure(text=output_msg, text_color=color)

    def start_extraction(self):
        if not self.extraction_src_path:
            self.ext_result_label.configure(
                text="Please select a source image before running extraction.",
                text_color="yellow",
            )
            return
        self.extraction_out_path = self._next_extracted_path(self.extraction_src_path)
        self.ext_output_label.configure(text=f"Output: {os.path.basename(self.extraction_out_path)}")

        self.set_ui_state("disabled")
        self.ext_result_label.configure(text="Processing... Detecting face and extracting crop...", text_color="cyan")
        self.ext_progressbar.grid()
        self.ext_progressbar.start()

        threading.Thread(
            target=self._extract_thread,
            args=(self.extraction_src_path, self.extraction_out_path),
            daemon=True,
        ).start()

    def _extract_thread(self, src_path: str, out_path: str):
        try:
            confidence = self.engine.extract_face(src_path, out_path, padding=0.175)
            result = {"ok": True, "confidence": confidence, "output": out_path}
        except Exception as e:
            result = {"ok": False, "error": str(e)}
        self.after(0, self._on_extraction_complete, result)

    def _on_extraction_complete(self, result: dict):
        self.ext_progressbar.stop()
        self.ext_progressbar.grid_remove()
        self.set_ui_state("normal")

        if not result.get("ok"):
            self.ext_result_label.configure(text=f"Error: {result['error']}", text_color="red")
            return

        self.ext_result_label.configure(
            text=(
                f"Extraction complete: {os.path.basename(result['output'])} "
                f"(Confidence: {result['confidence']:.1%})"
            ),
            text_color="#00FF00",
        )


def run_gui():
    app = ModernGUI()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
