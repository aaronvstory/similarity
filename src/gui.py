"""CustomTkinter GUI frontend for the face similarity application."""

from __future__ import annotations

import os
import queue
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk
from PIL import Image, ImageOps

from src import APP_TITLE
from src.engine import FaceEngine


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


WINDOW_SIZE = "1280x880"
MIN_WINDOW_SIZE = (1120, 780)
PREVIEW_SIZE = (340, 240)

VALID_EXISTING_FILE_MODES = ("index", "skip", "overwrite")


class ModernGUI(ctk.CTk):
    """Modern dark-mode GUI for local portrait similarity and extraction."""

    def __init__(self, engine: FaceEngine) -> None:
        super().__init__()
        self.engine = engine

        self._engine_ready = False
        self._busy = False
        self._worker_queue: Optional[queue.Queue[tuple[str, Any]]] = None

        self._similarity_img1_path: Optional[Path] = None
        self._similarity_img2_path: Optional[Path] = None
        self._similarity_img1_preview: Optional[ctk.CTkImage] = None
        self._similarity_img2_preview: Optional[ctk.CTkImage] = None

        self._extraction_img_path: Optional[Path] = None
        self._extraction_preview: Optional[ctk.CTkImage] = None

        self._action_buttons: List[ctk.CTkBaseClass] = []

        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(*MIN_WINDOW_SIZE)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_shell()
        self._set_busy(True, "Loading ArcFace + RetinaFace models...", disable_actions=True)
        self._start_background_task(
            worker=self.engine.initialize_models,
            on_success=self._on_engine_ready,
            on_error=self._handle_error,
            busy_message="Loading ArcFace + RetinaFace models...",
            disable_actions=True,
        )
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_shell(self) -> None:
        header = ctk.CTkFrame(self, corner_radius=24)
        header.grid(row=0, column=0, padx=24, pady=(24, 14), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Face Similarity Pro",
            font=ctk.CTkFont(size=30, weight="bold"),
        )
        title.grid(row=0, column=0, padx=24, pady=(20, 2), sticky="w")

        subtitle = ctk.CTkLabel(
            header,
            text="Similarity and extraction are separated into dedicated tabs. Batch workflows stay one click away.",
            text_color="#A0AEC0",
            font=ctk.CTkFont(size=13),
        )
        subtitle.grid(row=1, column=0, padx=24, pady=(0, 18), sticky="w")

        status_frame = ctk.CTkFrame(self, corner_radius=18)
        status_frame.grid(row=2, column=0, padx=24, pady=(0, 20), sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Starting...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#7DD3FC",
        )
        self.status_label.grid(row=0, column=0, padx=20, pady=(14, 4), sticky="w")

        self.status_bar = ctk.CTkProgressBar(status_frame, mode="indeterminate", height=10)
        self.status_bar.grid(row=1, column=0, padx=20, pady=(0, 14), sticky="ew")
        self.status_bar.stop()

        self.tabs = ctk.CTkTabview(self, corner_radius=20)
        self.tabs.grid(row=1, column=0, padx=24, pady=(0, 18), sticky="nsew")
        self.tabs.add("Similarity")
        self.tabs.add("Extraction")

        self._build_similarity_tab(self.tabs.tab("Similarity"))
        self._build_extraction_tab(self.tabs.tab("Extraction"))
        self._configure_tree_style()

    def _build_similarity_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        hero = ctk.CTkFrame(tab, corner_radius=22)
        hero.grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 16), sticky="ew")
        hero.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hero,
            text="Similarity Check",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(18, 2), sticky="w")
        ctk.CTkLabel(
            hero,
            text="Pick two portraits or a root folder and let the app handle matching, scoring, and folder renaming.",
            text_color="#94A3B8",
            font=ctk.CTkFont(size=13),
        ).grid(row=1, column=0, padx=20, pady=(0, 12), sticky="w")

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 18), sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.similarity_card_one = self._build_image_card(
            body,
            title="Portrait A",
            description="First image used in the comparison.",
            browse_label="Choose Portrait A",
            clear_label="Clear",
            browse_command=lambda: self._select_similarity_image(1),
            clear_command=lambda: self._clear_similarity_image(1),
            column=0,
        )
        self.similarity_card_two = self._build_image_card(
            body,
            title="Portrait B",
            description="Second image used in the comparison.",
            browse_label="Choose Portrait B",
            clear_label="Clear",
            browse_command=lambda: self._select_similarity_image(2),
            clear_command=lambda: self._clear_similarity_image(2),
            column=1,
        )

        compare_bar = ctk.CTkFrame(tab, corner_radius=22)
        compare_bar.grid(row=2, column=0, columnspan=2, padx=18, pady=(0, 18), sticky="ew")
        compare_bar.grid_columnconfigure(0, weight=1)

        self.compare_button = ctk.CTkButton(
            compare_bar,
            text="Run Similarity",
            height=48,
            corner_radius=14,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._run_similarity_compare,
        )
        self.compare_button.grid(row=0, column=0, padx=20, pady=(18, 10), sticky="ew")
        self._action_buttons.append(self.compare_button)

        self.similarity_result_label = ctk.CTkLabel(
            compare_bar,
            text="Select two portraits to begin.",
            font=ctk.CTkFont(size=19, weight="bold"),
            wraplength=1100,
            justify="left",
        )
        self.similarity_result_label.grid(row=1, column=0, padx=20, pady=(0, 18), sticky="w")

        batch = ctk.CTkFrame(tab, corner_radius=22)
        batch.grid(row=3, column=0, columnspan=2, padx=18, pady=(0, 18), sticky="ew")
        batch.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            batch,
            text="Batch Folder Similarity",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=20, pady=(18, 6), sticky="w")
        ctk.CTkLabel(
            batch,
            text="Search folders for matching files, score them, and rename folders with the score token.",
            text_color="#94A3B8",
        ).grid(row=1, column=0, columnspan=3, padx=20, pady=(0, 12), sticky="w")

        self.similarity_root_entry = ctk.CTkEntry(batch, placeholder_text="Root folder")
        self.similarity_root_entry.grid(row=2, column=0, padx=(20, 10), pady=10, sticky="ew")
        ctk.CTkButton(batch, text="Browse", width=110, command=self._choose_similarity_root).grid(
            row=2, column=1, padx=(0, 10), pady=10, sticky="ew"
        )

        self.similarity_batch_button = ctk.CTkButton(
            batch,
            text="Run Batch Similarity",
            height=44,
            corner_radius=14,
            command=self._run_batch_similarity,
        )
        self.similarity_batch_button.grid(row=2, column=2, padx=(0, 20), pady=10, sticky="ew")
        self._action_buttons.append(self.similarity_batch_button)

        keyword_row = ctk.CTkFrame(batch, fg_color="transparent")
        keyword_row.grid(row=3, column=0, columnspan=3, padx=20, pady=(0, 16), sticky="ew")
        keyword_row.grid_columnconfigure((0, 1), weight=1)

        self.similarity_img1_keyword_entry = self._place_field(
            keyword_row,
            "Image 1 keyword/regex",
            lambda parent: ctk.CTkEntry(parent),
            0,
            default_value="extracted",
        )
        self.similarity_img2_keyword_entry = self._place_field(
            keyword_row,
            "Image 2 keyword/regex",
            lambda parent: ctk.CTkEntry(parent),
            1,
            default_value="selfie",
        )

        self.similarity_tree = self._build_tree(
            batch,
            columns=("folder", "score", "match", "status"),
            headings=("Folder", "Score", "Match", "Status"),
            span_columns=3,
        )
        batch.grid_rowconfigure(4, weight=1)

    def _build_extraction_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        hero = ctk.CTkFrame(tab, corner_radius=22)
        hero.grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 16), sticky="ew")
        hero.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hero,
            text="Extraction",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(18, 2), sticky="w")
        ctk.CTkLabel(
            hero,
            text="Extract one face or batch process a directory tree with configurable padding and file handling.",
            text_color="#94A3B8",
            font=ctk.CTkFont(size=13),
        ).grid(row=1, column=0, padx=20, pady=(0, 12), sticky="w")

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 18), sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.extraction_card = self._build_image_card(
            body,
            title="Source Portrait",
            description="Choose the source image to crop and save.",
            browse_label="Choose Source Image",
            clear_label="Clear",
            browse_command=self._select_extraction_image,
            clear_command=self._clear_extraction_image,
            column=0,
        )

        extraction_actions = ctk.CTkFrame(body, corner_radius=22)
        extraction_actions.grid(row=0, column=1, padx=(10, 18), pady=0, sticky="nsew")
        extraction_actions.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            extraction_actions,
            text="Single Extraction",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=20, pady=(18, 6), sticky="w")
        ctk.CTkLabel(
            extraction_actions,
            text="The output path respects the same index/skip/overwrite rules used by the CLI.",
            text_color="#94A3B8",
        ).grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 12), sticky="w")

        self.extraction_output_label = ctk.CTkLabel(
            extraction_actions,
            text="No source image selected.",
            text_color="#CBD5E1",
            wraplength=420,
            justify="left",
        )
        self.extraction_output_label.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="w")

        self.extract_button = ctk.CTkButton(
            extraction_actions,
            text="Run Extraction",
            height=46,
            corner_radius=14,
            command=self._run_single_extraction,
        )
        self.extract_button.grid(row=3, column=0, columnspan=2, padx=20, pady=(6, 18), sticky="ew")
        self._action_buttons.append(self.extract_button)

        batch = ctk.CTkFrame(tab, corner_radius=22)
        batch.grid(row=2, column=0, columnspan=2, padx=18, pady=(0, 18), sticky="ew")
        batch.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            batch,
            text="Batch Extraction",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, padx=20, pady=(18, 6), sticky="w")
        ctk.CTkLabel(
            batch,
            text="Pick a root folder, keyword, padding ratio, and existing-file mode.",
            text_color="#94A3B8",
        ).grid(row=1, column=0, columnspan=4, padx=20, pady=(0, 12), sticky="w")

        self.extraction_root_entry = ctk.CTkEntry(batch, placeholder_text="Root folder")
        self.extraction_root_entry.grid(row=2, column=0, padx=(20, 10), pady=10, sticky="ew")
        ctk.CTkButton(batch, text="Browse", width=110, command=self._choose_extraction_root).grid(
            row=2, column=1, padx=(0, 10), pady=10, sticky="ew"
        )

        self.extraction_batch_button = ctk.CTkButton(
            batch,
            text="Run Batch Extraction",
            height=44,
            corner_radius=14,
            command=self._run_batch_extraction,
        )
        self.extraction_batch_button.grid(row=2, column=2, padx=(0, 10), pady=10, sticky="ew")
        self._action_buttons.append(self.extraction_batch_button)

        controls = ctk.CTkFrame(batch, fg_color="transparent")
        controls.grid(row=3, column=0, columnspan=4, padx=20, pady=(0, 16), sticky="ew")
        controls.grid_columnconfigure((0, 1, 2), weight=1)

        self.extraction_keyword_entry = self._place_field(
            controls,
            "Extraction keyword/regex",
            lambda parent: ctk.CTkEntry(parent),
            0,
            default_value="front",
        )
        self.extraction_padding_entry = self._place_field(
            controls,
            "Padding ratio",
            lambda parent: ctk.CTkEntry(parent),
            1,
            default_value="0.175",
        )
        self.existing_mode_menu = self._place_field(
            controls,
            "Existing file mode",
            lambda parent: ctk.CTkOptionMenu(parent, values=list(VALID_EXISTING_FILE_MODES)),
            2,
            default_value="index",
        )

        self.extraction_tree = self._build_tree(
            batch,
            columns=("folder", "source", "confidence", "status"),
            headings=("Folder", "Source", "Confidence", "Status"),
            span_columns=4,
        )
        batch.grid_rowconfigure(4, weight=1)

    def _build_image_card(
        self,
        parent: ctk.CTkFrame,
        *,
        title: str,
        description: str,
        browse_label: str,
        clear_label: str,
        browse_command: Callable[[], None],
        clear_command: Callable[[], None],
        column: int,
    ) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, corner_radius=22)
        card.grid(row=0, column=column, padx=(0, 10) if column == 0 else (10, 0), sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=22, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(18, 4), sticky="w"
        )
        ctk.CTkLabel(card, text=description, text_color="#94A3B8", font=ctk.CTkFont(size=13)).grid(
            row=1, column=0, padx=20, sticky="w"
        )

        preview_container = ctk.CTkFrame(
            card,
            corner_radius=18,
            fg_color="#141A22",
            border_width=1,
            border_color="#273142",
        )
        preview_container.grid(row=2, column=0, padx=20, pady=16, sticky="nsew")
        preview_container.grid_columnconfigure(0, weight=1)
        preview_container.grid_rowconfigure(0, weight=1)

        preview_label = ctk.CTkLabel(
            preview_container,
            text="No image selected",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#64748B",
            width=PREVIEW_SIZE[0],
            height=PREVIEW_SIZE[1],
            compound="top",
        )
        preview_label.grid(row=0, column=0, padx=14, pady=14, sticky="nsew")

        path_label = ctk.CTkLabel(
            card,
            text="Waiting for selection...",
            text_color="#94A3B8",
            wraplength=420,
            justify="left",
        )
        path_label.grid(row=3, column=0, padx=20, pady=(0, 12), sticky="w")

        button_row = ctk.CTkFrame(card, fg_color="transparent")
        button_row.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        button_row.grid_columnconfigure((0, 1), weight=1)

        browse_button = ctk.CTkButton(button_row, text=browse_label, height=42, corner_radius=12, command=browse_command)
        browse_button.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        clear_button = ctk.CTkButton(button_row, text=clear_label, height=42, corner_radius=12, command=clear_command)
        clear_button.grid(row=0, column=1, padx=(8, 0), sticky="ew")

        if title == "Portrait A":
            self.similarity_img1_preview_label = preview_label
            self.similarity_img1_path_label = path_label
            self.similarity_img1_button = browse_button
            self.similarity_img1_clear_button = clear_button
            self._action_buttons.extend([browse_button, clear_button])
        elif title == "Portrait B":
            self.similarity_img2_preview_label = preview_label
            self.similarity_img2_path_label = path_label
            self.similarity_img2_button = browse_button
            self.similarity_img2_clear_button = clear_button
            self._action_buttons.extend([browse_button, clear_button])
        elif title == "Source Portrait":
            self.extraction_preview_label = preview_label
            self.extraction_path_label = path_label
            self.extraction_choose_button = browse_button
            self.extraction_clear_button = clear_button
            self._action_buttons.extend([browse_button, clear_button])

        return card

    def _place_field(
        self,
        parent: ctk.CTkFrame,
        label: str,
        widget_factory: Callable[[ctk.CTkFrame], Any],
        column: int,
        *,
        default_value: Optional[str] = None,
    ) -> Any:
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.grid(row=0, column=column, padx=10, sticky="ew")
        container.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(container, text=label, text_color="#94A3B8", anchor="w").grid(
            row=0, column=0, sticky="ew", pady=(0, 6)
        )
        widget = widget_factory(container)
        widget.grid(row=1, column=0, sticky="ew")
        if default_value is not None:
            if hasattr(widget, "insert"):
                widget.insert(0, default_value)
            elif hasattr(widget, "set"):
                widget.set(default_value)
        return widget

    def _build_tree(
        self,
        parent: ctk.CTkFrame,
        *,
        columns: tuple[str, ...],
        headings: tuple[str, ...],
        span_columns: int,
    ) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=8)
        for column, heading in zip(columns, headings, strict=True):
            tree.heading(column, text=heading)
            tree.column(column, width=180, anchor="w")

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.grid(row=4, column=0, columnspan=span_columns, padx=20, pady=(0, 20), sticky="nsew")
        scrollbar.grid(row=4, column=span_columns, padx=(0, 20), pady=(0, 20), sticky="ns")

        return tree

    def _configure_tree_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            return

        style.configure(
            "Treeview",
            background="#111827",
            fieldbackground="#111827",
            foreground="#E5E7EB",
            rowheight=30,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#1F2937",
            foreground="#F8FAFC",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Treeview", background=[("selected", "#2563EB")], foreground=[("selected", "#FFFFFF")])

    # ------------------------------------------------------------------
    # Background task plumbing
    # ------------------------------------------------------------------
    def _start_background_task(
        self,
        *,
        worker: Callable[[], Any],
        on_success: Callable[[Any], None],
        on_error: Callable[[Exception], None],
        busy_message: str,
        disable_actions: bool,
    ) -> None:
        self._set_busy(True, busy_message, disable_actions)
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def runner() -> None:
            try:
                result_queue.put(("success", worker()))
            except Exception as exc:  # pragma: no cover - exercised through UI flows
                result_queue.put(("error", exc))

        threading.Thread(target=runner, daemon=True).start()
        self._worker_queue = result_queue
        self.after(120, lambda: self._poll_worker_queue(result_queue, on_success, on_error))

    def _poll_worker_queue(
        self,
        result_queue: queue.Queue[tuple[str, Any]],
        on_success: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            self.after(120, lambda: self._poll_worker_queue(result_queue, on_success, on_error))
            return

        if status == "success":
            on_success(payload)
        else:
            on_error(payload)

    def _set_busy(self, active: bool, status_text: str, disable_actions: bool) -> None:
        self._busy = active
        self.status_label.configure(text=status_text)
        if active:
            self.status_bar.start()
        else:
            self.status_bar.stop()

        state = "disabled" if disable_actions else "normal"
        for button in self._action_buttons:
            button.configure(state=state)

        self._refresh_action_states()

    def _refresh_action_states(self) -> None:
        similarity_ready = self._engine_ready and self._similarity_img1_path and self._similarity_img2_path and not self._busy
        extraction_ready = self._engine_ready and self._extraction_img_path and not self._busy

        self.compare_button.configure(state="normal" if similarity_ready else "disabled")
        self.extract_button.configure(state="normal" if extraction_ready else "disabled")

        similarity_root = self.similarity_root_entry.get().strip()
        extraction_root = self.extraction_root_entry.get().strip()
        self.similarity_batch_button.configure(state="normal" if self._engine_ready and similarity_root and not self._busy else "disabled")
        self.extraction_batch_button.configure(state="normal" if self._engine_ready and extraction_root and not self._busy else "disabled")

    def _on_engine_ready(self, _: Any = None) -> None:
        self._engine_ready = True
        self._set_busy(False, "Models are ready. Choose a workflow.", disable_actions=False)

    def _handle_error(self, error: Exception) -> None:
        self._set_busy(False, "Operation failed.", disable_actions=False)
        self._show_warning(str(error))

    # ------------------------------------------------------------------
    # Image selection and preview
    # ------------------------------------------------------------------
    def _create_preview(self, image_path: Path) -> ctk.CTkImage:
        with Image.open(image_path) as source:
            prepared = ImageOps.exif_transpose(source).convert("RGB")
            preview = prepared.copy()
            preview.thumbnail(PREVIEW_SIZE)
            return ctk.CTkImage(light_image=preview, dark_image=preview, size=preview.size)

    def _select_similarity_image(self, slot: int) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            self._log("Similarity selection cancelled.")
            return

        try:
            resolved = Path(path).resolve()
            self.engine.validate_image_file(str(resolved))
            preview = self._create_preview(resolved)
        except Exception as exc:
            self._show_warning(str(exc))
            return

        if slot == 1:
            self._similarity_img1_path = resolved
            self._similarity_img1_preview = preview
            self.similarity_img1_preview_label.configure(image=preview, text="")
            self.similarity_img1_path_label.configure(text=str(resolved))
        else:
            self._similarity_img2_path = resolved
            self._similarity_img2_preview = preview
            self.similarity_img2_preview_label.configure(image=preview, text="")
            self.similarity_img2_path_label.configure(text=str(resolved))

        self._log(f"Loaded similarity image {slot}: {resolved}")
        self._refresh_action_states()

    def _clear_similarity_image(self, slot: int) -> None:
        if slot == 1:
            self._similarity_img1_path = None
            self._similarity_img1_preview = None
            self.similarity_img1_preview_label.configure(image=None, text="No image selected")
            self.similarity_img1_path_label.configure(text="Waiting for selection...")
        else:
            self._similarity_img2_path = None
            self._similarity_img2_preview = None
            self.similarity_img2_preview_label.configure(image=None, text="No image selected")
            self.similarity_img2_path_label.configure(text="Waiting for selection...")
        self._refresh_action_states()

    def _select_extraction_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a source image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            self._log("Extraction selection cancelled.")
            return

        try:
            resolved = Path(path).resolve()
            self.engine.validate_image_file(str(resolved))
            preview = self._create_preview(resolved)
        except Exception as exc:
            self._show_warning(str(exc))
            return

        self._extraction_img_path = resolved
        self._extraction_preview = preview
        self.extraction_preview_label.configure(image=preview, text="")
        self.extraction_path_label.configure(text=str(resolved))
        self.extraction_output_label.configure(text=f"Output will be written next to: {resolved.name}")
        self._log(f"Loaded extraction image: {resolved}")
        self._refresh_action_states()

    def _clear_extraction_image(self) -> None:
        self._extraction_img_path = None
        self._extraction_preview = None
        self.extraction_preview_label.configure(image=None, text="No image selected")
        self.extraction_path_label.configure(text="Waiting for selection...")
        self.extraction_output_label.configure(text="No source image selected.")
        self._refresh_action_states()

    # ------------------------------------------------------------------
    # Single similarity and extraction
    # ------------------------------------------------------------------
    def _run_similarity_compare(self) -> None:
        if not self._similarity_img1_path or not self._similarity_img2_path:
            self._show_warning("Choose both portraits before running similarity.")
            return

        self.similarity_result_label.configure(text="Processing the selected portraits...", text_color="#CBD5E1")
        self._start_background_task(
            worker=lambda: self.engine.compare_images(
                str(self._similarity_img1_path),
                str(self._similarity_img2_path),
            ),
            on_success=self._display_similarity_result,
            on_error=self._handle_error,
            busy_message="Comparing portraits...",
            disable_actions=True,
        )

    def _display_similarity_result(self, result: Dict[str, Any]) -> None:
        score = result.get("score", 0.0)
        error = result.get("error")
        is_match = bool(result.get("match"))

        if error:
            self.similarity_result_label.configure(text=str(error), text_color="#F59E0B")
            self._log(f"Similarity error: {error}")
        else:
            color = "#22C55E" if is_match else "#EF4444"
            verdict = "Match" if is_match else "No match"
            self.similarity_result_label.configure(
                text=f"{verdict} - {score:.2f}%",
                text_color=color,
            )
            self._log(f"Similarity complete: {score:.2f}% ({verdict})")

        self._set_busy(False, "Similarity complete.", disable_actions=False)
        self._refresh_action_states()

    def _run_single_extraction(self) -> None:
        if not self._extraction_img_path:
            self._show_warning("Choose a source image before running extraction.")
            return

        source = self._extraction_img_path
        out_path = self._get_available_path(str(source.parent), source.name)
        if not out_path:
            self._show_warning("Existing extracted file policy is set to skip and a target already exists.")
            return

        self.extraction_output_label.configure(text=f"Saving to: {out_path}")
        self._start_background_task(
            worker=lambda: self.engine.extract_face(
                str(source),
                out_path,
                padding=self._read_padding_ratio(),
            ),
            on_success=lambda confidence: self._display_extraction_result(source, out_path, confidence),
            on_error=self._handle_error,
            busy_message="Extracting face...",
            disable_actions=True,
        )

    def _display_extraction_result(self, source: Path, out_path: str, confidence: float) -> None:
        self.extraction_output_label.configure(
            text=f"Extracted {source.name} to {os.path.basename(out_path)} with confidence {confidence:.2%}",
        )
        self._log(f"Extraction complete: {out_path} ({confidence:.2%})")
        self._set_busy(False, "Extraction complete.", disable_actions=False)
        self._refresh_action_states()

    # ------------------------------------------------------------------
    # Batch workflows
    # ------------------------------------------------------------------
    def _choose_similarity_root(self) -> None:
        root = filedialog.askdirectory(title="Choose root directory for batch similarity")
        if root:
            self.similarity_root_entry.delete(0, "end")
            self.similarity_root_entry.insert(0, root)
            self._refresh_action_states()

    def _choose_extraction_root(self) -> None:
        root = filedialog.askdirectory(title="Choose root directory for batch extraction")
        if root:
            self.extraction_root_entry.delete(0, "end")
            self.extraction_root_entry.insert(0, root)
            self._refresh_action_states()

    def _run_batch_similarity(self) -> None:
        root = self.similarity_root_entry.get().strip()
        if not root:
            self._show_warning("Choose a root directory before running batch similarity.")
            return

        img1_keyword = self.similarity_img1_keyword_entry.get().strip() or "extracted"
        img2_keyword = self.similarity_img2_keyword_entry.get().strip() or "selfie"
        self._clear_tree(self.similarity_tree)
        self._log(f"Batch similarity started for {root}")

        def worker() -> List[Dict[str, Any]]:
            return self._batch_similarity(root, img1_keyword, img2_keyword)

        self._start_background_task(
            worker=worker,
            on_success=lambda results: self._display_batch_results(self.similarity_tree, results, mode="similarity"),
            on_error=self._handle_error,
            busy_message="Running batch similarity...",
            disable_actions=True,
        )

    def _run_batch_extraction(self) -> None:
        root = self.extraction_root_entry.get().strip()
        if not root:
            self._show_warning("Choose a root directory before running batch extraction.")
            return

        keyword = self.extraction_keyword_entry.get().strip() or "front"
        self._clear_tree(self.extraction_tree)
        self._log(f"Batch extraction started for {root}")

        def worker() -> List[Dict[str, Any]]:
            return self._batch_extraction(root, keyword)

        self._start_background_task(
            worker=worker,
            on_success=lambda results: self._display_batch_results(self.extraction_tree, results, mode="extraction"),
            on_error=self._handle_error,
            busy_message="Running batch extraction...",
            disable_actions=True,
        )

    def _batch_similarity(self, root_dir: str, img1_keyword: str, img2_keyword: str) -> List[Dict[str, Any]]:
        folders: List[tuple[str, str, str]] = []
        for dirpath, _, _ in os.walk(root_dir):
            img1 = self._find_image_with_keyword(dirpath, img1_keyword)
            img2 = self._find_image_with_keyword(dirpath, img2_keyword)
            if img1 and img2:
                folders.append((dirpath, img1, img2))

        folders.sort(key=lambda item: item[0].count(os.sep), reverse=True)
        results: List[Dict[str, Any]] = []
        for folder, img1, img2 in folders:
            comparison = self.engine.compare_images(img1, img2)
            if comparison.get("error"):
                results.append(
                    {
                        "folder": os.path.basename(folder),
                        "score": "-",
                        "match": "N/A",
                        "status": comparison["error"],
                    }
                )
                continue

            score = comparison["score"]
            new_path = self._get_new_folder_name(folder, score)
            if new_path != folder:
                os.rename(folder, new_path)
                folder_display = f"{os.path.basename(folder)} -> {os.path.basename(new_path)}"
                status = "Renamed"
            else:
                folder_display = os.path.basename(folder)
                status = "Skipped"

            results.append(
                {
                    "folder": folder_display,
                    "score": f"{score}%",
                    "match": "YES" if comparison["match"] else "NO",
                    "status": status,
                }
            )

        return results

    def _batch_extraction(self, root_dir: str, keyword: str) -> List[Dict[str, Any]]:
        jobs: List[tuple[str, str]] = []
        for dirpath, _, _ in os.walk(root_dir):
            match = self._find_image_with_keyword(dirpath, keyword)
            if match:
                out_path = self._get_available_path(dirpath, os.path.basename(match))
                if out_path:
                    jobs.append((match, out_path))

        results: List[Dict[str, Any]] = []
        for src, out in jobs:
            try:
                confidence = self.engine.extract_face(src, out, padding=self._read_padding_ratio())
                results.append(
                    {
                        "folder": os.path.basename(os.path.dirname(src)),
                        "source": os.path.basename(src),
                        "confidence": f"{confidence:.1%}",
                        "status": "SUCCESS",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "folder": os.path.basename(os.path.dirname(src)),
                        "source": os.path.basename(src),
                        "confidence": "0%",
                        "status": f"FAIL: {exc}",
                    }
                )

        return results

    def _display_batch_results(self, tree: ttk.Treeview, results: List[Dict[str, Any]], *, mode: str) -> None:
        self._clear_tree(tree)
        for row in results:
            values = tuple(row[column] for column in tree["columns"])
            tree.insert("", "end", values=values)

        self._log(f"Batch {mode} finished with {len(results)} result rows.")
        self._set_busy(False, f"Batch {mode} complete.", disable_actions=False)
        self._refresh_action_states()

    # ------------------------------------------------------------------
    # Shared batch helpers
    # ------------------------------------------------------------------
    def _find_image_with_keyword(self, folder_path: str, keyword: str) -> Optional[str]:
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        try:
            pattern = re.compile(keyword, re.IGNORECASE)
        except re.error:
            pattern = None

        keyword_lower = keyword.lower()
        candidates = [
            file
            for file in os.listdir(folder_path)
            if file.lower().endswith(valid_extensions)
        ]

        for file in candidates:
            file_lower = file.lower()
            if (pattern and pattern.search(file)) or (not pattern and keyword_lower in file_lower):
                return os.path.join(folder_path, file)

        best_path = None
        best_score = 0.0
        for file in candidates:
            stem = os.path.splitext(file)[0]
            score = self._fuzzy_match_score(keyword_lower, stem)
            if score > best_score:
                best_score = score
                best_path = os.path.join(folder_path, file)

        if best_path and best_score >= 0.60:
            return best_path
        return None

    def _fuzzy_match_score(self, keyword: str, filename_stem: str) -> float:
        keyword_clean = keyword.strip().lower()
        filename_clean = filename_stem.strip().lower()
        if not keyword_clean or not filename_clean:
            return 0.0

        sequence_score = SequenceMatcher(None, keyword_clean, filename_clean).ratio()
        filename_parts = [part for part in re.split(r"[_\-\s]+", filename_clean) if part]
        part_score = max((SequenceMatcher(None, keyword_clean, part).ratio() for part in filename_parts), default=0.0)

        keyword_tokens = set(re.findall(r"[a-z0-9]+", keyword_clean))
        filename_tokens = set(re.findall(r"[a-z0-9]+", filename_clean))
        token_overlap = (
            len(keyword_tokens & filename_tokens) / len(keyword_tokens)
            if keyword_tokens
            else 0.0
        )

        return max(sequence_score, part_score, token_overlap)

    def _get_available_path(self, dirpath: str, filename: str) -> Optional[str]:
        ext = os.path.splitext(filename)[1]
        base_name = "extracted"
        target_path = os.path.join(dirpath, f"{base_name}{ext}")
        source_path = os.path.join(dirpath, filename)

        mode = self._existing_file_mode()
        source_norm = os.path.normcase(os.path.normpath(source_path))
        target_norm = os.path.normcase(os.path.normpath(target_path))
        force_index = source_norm == target_norm

        if not os.path.exists(target_path):
            return target_path

        if mode == "skip" and not force_index:
            return None
        if mode == "overwrite" and not force_index:
            return target_path

        idx = 2
        while True:
            new_path = os.path.join(dirpath, f"{base_name}{idx}{ext}")
            if not os.path.exists(new_path):
                return new_path
            idx += 1

    def _existing_file_mode(self) -> str:
        selected = self.existing_mode_menu.get().strip().lower()
        if selected in VALID_EXISTING_FILE_MODES:
            return selected
        return "index"

    def _get_new_folder_name(self, current_path: str, score: float) -> str:
        parent = os.path.dirname(current_path)
        folder_name = os.path.basename(current_path)
        score_str = str(int(round(score)))

        patterns = [
            re.compile(r"(?P<prefix>.*?)(?:\s*\d{1,3}%?)(?P<suffix>\s*-\s*.*)?$"),
            re.compile(r"(?P<prefix>.*?)(?:\s*\d{1,3}%?)$"),
        ]
        for pattern in patterns:
            match = pattern.match(folder_name)
            if match and match.group("prefix").strip():
                prefix = match.group("prefix").rstrip()
                suffix = match.groupdict().get("suffix") or ""
                if suffix:
                    return os.path.join(parent, f"{prefix} {score_str}{suffix}")
                return os.path.join(parent, f"{prefix} {score_str}")

        if re.search(r"\d{1,3}%?(?=\s*-\s*)", folder_name):
            return re.sub(r"\d{1,3}%?(?=\s*-\s*)", score_str, current_path, count=1)
        if re.search(r"\d{1,3}%?$", folder_name):
            return re.sub(r"\d{1,3}%?$", score_str, current_path, count=1)
        return os.path.join(parent, f"{folder_name} {score_str}")

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _read_padding_ratio(self) -> float:
        try:
            value = float(self.extraction_padding_entry.get().strip())
        except ValueError:
            value = 0.175
        return max(0.0, min(1.0, value))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        self.status_label.configure(text=message)

    @staticmethod
    def _show_warning(message: str) -> None:
        messagebox.showwarning(APP_TITLE, message)

    def _on_close(self) -> None:
        self.engine.shutdown()
        self.destroy()


def run_gui() -> None:
    """Launch the GUI application."""
    app = ModernGUI(FaceEngine())
    app.mainloop()
