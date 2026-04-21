from __future__ import annotations

import importlib
import sys
import types
import unittest
from typing import ClassVar
from unittest.mock import patch


class _DeepFaceStub:
    @staticmethod
    def build_model(model_name: str):
        return model_name

    @staticmethod
    def extract_faces(**kwargs):
        return [{"face": "face", "facial_area": {"w": 1, "h": 1}}]

    @staticmethod
    def represent(**kwargs):
        return [{"embedding": [1.0, 0.0, 0.0]}]


class _WidgetStub:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.state = kwargs.get("state")
        self.text = kwargs.get("text", "")
        self.text_color = kwargs.get("text_color")
        self.image = kwargs.get("image")
        self.grid_hidden = False
        self.value = None

    def grid(self, *args, **kwargs):
        self.grid_hidden = False

    def grid_remove(self):
        self.grid_hidden = True

    def grid_rowconfigure(self, *args, **kwargs):
        return None

    def grid_columnconfigure(self, *args, **kwargs):
        return None

    def configure(self, **kwargs):
        self.kwargs.update(kwargs)
        if "state" in kwargs:
            self.state = kwargs["state"]
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "text_color" in kwargs:
            self.text_color = kwargs["text_color"]
        if "image" in kwargs:
            self.image = kwargs["image"]

    def set(self, value):
        self.value = value

    def start(self):
        return None

    def stop(self):
        return None


class _CTkStub(_WidgetStub):
    def title(self, *_args, **_kwargs):
        return None

    def geometry(self, *_args, **_kwargs):
        return None

    def minsize(self, *_args, **_kwargs):
        return None

    def after(self, _delay, callback, *args):
        callback(*args)

    def mainloop(self):
        return None


class _CTkImageStub:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs


class _CTkTabViewStub(_WidgetStub):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tabs = {}

    def add(self, name: str):
        tab = _WidgetStub()
        self.tabs[name] = tab
        return tab


class _CTkModuleStub(types.ModuleType):
    def __init__(self):
        super().__init__("customtkinter")
        self.CTk = _CTkStub
        self.CTkTabview = _CTkTabViewStub
        self.CTkFrame = _WidgetStub
        self.CTkLabel = _WidgetStub
        self.CTkButton = _WidgetStub
        self.CTkProgressBar = _WidgetStub
        self.CTkImage = _CTkImageStub
        self.CTkFont = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
        self.set_appearance_mode = lambda *args, **kwargs: None
        self.set_default_color_theme = lambda *args, **kwargs: None


class _EngineStub:
    def initialize_models(self):
        return None

    def compare_images(self, _path1: str, _path2: str):
        return {"match": True, "score": 92.5, "error": None}

    def extract_face(self, _src: str, _out: str, padding: float = 0.175):
        return 0.81


class _ThreadCaptureBase:
    instances: ClassVar[list["_ThreadCaptureBase"]] = []


class TestModernGUI(unittest.TestCase):
    def setUp(self) -> None:
        self.thread_instances: list[_ThreadCaptureBase] = []
        self._original_gui_module = sys.modules.pop("src.gui", None)
        self.addCleanup(self._restore_gui_module)

        deepface_module = types.ModuleType("deepface")
        deepface_module.DeepFace = _DeepFaceStub
        tkinter_module = types.ModuleType("tkinter")

        class _TclError(Exception):
            pass

        filedialog_module = types.ModuleType("tkinter.filedialog")
        filedialog_module.askopenfilename = lambda *args, **kwargs: ""
        tkinter_module.TclError = _TclError
        tkinter_module.filedialog = filedialog_module

        parent = self

        class _ThreadCapture(_ThreadCaptureBase):
            def __init__(self, target=None, args=(), daemon=None, **kwargs):
                self.target = target
                self.args = args
                self.daemon = daemon
                self.started = False
                parent.thread_instances.append(self)

            def start(self):
                self.started = True

        self.thread_capture_class = _ThreadCapture

        patcher = patch.dict(
            sys.modules,
            {
                "customtkinter": _CTkModuleStub(),
                "deepface": deepface_module,
                "tkinter": tkinter_module,
                "tkinter.filedialog": filedialog_module,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.gui_module = importlib.import_module("src.gui")
        self.gui_module = importlib.reload(self.gui_module)

        self.thread_patcher = patch.object(self.gui_module.threading, "Thread", self.thread_capture_class)
        self.thread_patcher.start()
        self.addCleanup(self.thread_patcher.stop)

        self.engine_patcher = patch.object(self.gui_module, "FaceEngine", _EngineStub)
        self.engine_patcher.start()
        self.addCleanup(self.engine_patcher.stop)

    def _restore_gui_module(self) -> None:
        sys.modules.pop("src.gui", None)
        if self._original_gui_module is not None:
            sys.modules["src.gui"] = self._original_gui_module

    def test_init_starts_model_warmup_on_daemon_thread(self) -> None:
        app = self.gui_module.ModernGUI()
        self.assertEqual(len(self.thread_instances), 1)
        thread = self.thread_instances[0]
        self.assertEqual(thread.target.__name__, "_init_models_thread")
        self.assertTrue(thread.daemon)
        self.assertTrue(thread.started)
        self.assertEqual(app.btn_run.state, "disabled")
        self.assertEqual(app.btn_run.kwargs.get("text"), "Run Similarity Comparison")

    def test_on_models_ready_re_enables_controls(self) -> None:
        app = self.gui_module.ModernGUI()
        app._on_models_ready()
        self.assertEqual(app.btn_upload1.state, "normal")
        self.assertEqual(app.btn_upload2.state, "normal")
        self.assertEqual(app.btn_run.state, "normal")
        self.assertEqual(app.btn_upload_extract.state, "normal")
        self.assertEqual(app.btn_run_extract.state, "normal")
        self.assertEqual(app.sim_result_label.text, "")
        self.assertEqual(app.ext_result_label.text, "")
        self.assertTrue(app.sim_progressbar.grid_hidden)
        self.assertTrue(app.ext_progressbar.grid_hidden)

    def test_start_comparison_spawns_daemon_worker_and_updates_status(self) -> None:
        app = self.gui_module.ModernGUI()
        app.img1_path = "img1.png"
        app.img2_path = "img2.png"
        self.thread_instances = []

        app.start_comparison()

        self.assertEqual(len(self.thread_instances), 1)
        thread = self.thread_instances[0]
        self.assertEqual(thread.target.__name__, "_compare_thread")
        self.assertEqual(thread.args, ("img1.png", "img2.png"))
        self.assertTrue(thread.daemon)
        self.assertEqual(app.btn_run.state, "disabled")
        self.assertIn("Processing...", app.sim_result_label.text)

    def test_on_comparison_complete_renders_expected_success_text(self) -> None:
        app = self.gui_module.ModernGUI()
        app._on_comparison_complete({"match": True, "score": 98.7, "error": None})
        self.assertIn("Face similarity ratio: 98.7%", app.sim_result_label.text)
        self.assertIn("are the same person", app.sim_result_label.text)
        self.assertEqual(app.sim_result_label.text_color, "#00FF00")

    def test_on_comparison_complete_renders_error(self) -> None:
        app = self.gui_module.ModernGUI()
        app._on_comparison_complete({"match": False, "score": 0, "error": "bad input"})
        self.assertEqual(app.sim_result_label.text, "Error: bad input")
        self.assertEqual(app.sim_result_label.text_color, "red")

    def test_start_extraction_spawns_daemon_worker_and_updates_status(self) -> None:
        app = self.gui_module.ModernGUI()
        app.extraction_src_path = "src.png"
        app.extraction_out_path = "extracted.png"
        self.thread_instances = []

        with patch("src.gui.os.path.exists", return_value=False):
            app.start_extraction()

        self.assertEqual(len(self.thread_instances), 1)
        thread = self.thread_instances[0]
        self.assertEqual(thread.target.__name__, "_extract_thread")
        self.assertEqual(thread.args, ("src.png", "extracted.png"))
        self.assertTrue(thread.daemon)
        self.assertEqual(app.btn_run_extract.state, "disabled")
        self.assertIn("Processing...", app.ext_result_label.text)

    def test_on_extraction_complete_renders_success_text(self) -> None:
        app = self.gui_module.ModernGUI()
        app._on_extraction_complete({"ok": True, "confidence": 0.91, "output": "C:/tmp/extracted.png"})
        self.assertIn("Extraction complete: extracted.png", app.ext_result_label.text)
        self.assertIn("Confidence: 91.0%", app.ext_result_label.text)
        self.assertEqual(app.ext_result_label.text_color, "#00FF00")

    def test_on_extraction_complete_renders_error(self) -> None:
        app = self.gui_module.ModernGUI()
        app._on_extraction_complete({"ok": False, "error": "face missing"})
        self.assertEqual(app.ext_result_label.text, "Error: face missing")
        self.assertEqual(app.ext_result_label.text_color, "red")


if __name__ == "__main__":
    unittest.main()
