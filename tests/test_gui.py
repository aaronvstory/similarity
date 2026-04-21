from __future__ import annotations

import importlib
import sys
import types
import unittest
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


class _CTkModuleStub(types.ModuleType):
    def __init__(self):
        super().__init__("customtkinter")
        self.CTk = _CTkStub
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


class _ThreadCapture:
    instances: list["_ThreadCapture"] = []

    def __init__(self, target=None, args=(), daemon=None, **kwargs):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        _ThreadCapture.instances.append(self)

    def start(self):
        self.started = True


class TestModernGUI(unittest.TestCase):
    def setUp(self) -> None:
        _ThreadCapture.instances = []
        self._original_gui_module = sys.modules.pop("src.gui", None)
        self.addCleanup(self._restore_gui_module)

        deepface_module = types.ModuleType("deepface")
        deepface_module.DeepFace = _DeepFaceStub

        patcher = patch.dict(
            sys.modules,
            {
                "customtkinter": _CTkModuleStub(),
                "deepface": deepface_module,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.gui_module = importlib.import_module("src.gui")
        self.gui_module = importlib.reload(self.gui_module)

        self.thread_patcher = patch.object(self.gui_module.threading, "Thread", _ThreadCapture)
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
        self.assertEqual(len(_ThreadCapture.instances), 1)
        thread = _ThreadCapture.instances[0]
        self.assertEqual(thread.target.__name__, "_init_models_thread")
        self.assertTrue(thread.daemon)
        self.assertTrue(thread.started)
        self.assertEqual(app.btn_run.state, "disabled")
        self.assertEqual(app.btn_run.kwargs.get("text"), "Run Comparison")

    def test_on_models_ready_re_enables_controls(self) -> None:
        app = self.gui_module.ModernGUI()
        app._on_models_ready()
        self.assertEqual(app.btn_upload1.state, "normal")
        self.assertEqual(app.btn_upload2.state, "normal")
        self.assertEqual(app.btn_run.state, "normal")
        self.assertEqual(app.result_label.text, "")
        self.assertTrue(app.progressbar.grid_hidden)

    def test_start_comparison_spawns_daemon_worker_and_updates_status(self) -> None:
        app = self.gui_module.ModernGUI()
        app.img1_path = "img1.png"
        app.img2_path = "img2.png"
        _ThreadCapture.instances = []

        app.start_comparison()

        self.assertEqual(len(_ThreadCapture.instances), 1)
        thread = _ThreadCapture.instances[0]
        self.assertEqual(thread.target.__name__, "_compare_thread")
        self.assertEqual(thread.args, ("img1.png", "img2.png"))
        self.assertTrue(thread.daemon)
        self.assertEqual(app.btn_run.state, "disabled")
        self.assertIn("Processing...", app.result_label.text)

    def test_on_comparison_complete_renders_expected_success_text(self) -> None:
        app = self.gui_module.ModernGUI()
        app._on_comparison_complete({"match": True, "score": 98.7, "error": None})
        self.assertIn("Face similarity ratio: 98.7%", app.result_label.text)
        self.assertIn("are the same person", app.result_label.text)
        self.assertEqual(app.result_label.text_color, "#00FF00")

    def test_on_comparison_complete_renders_error(self) -> None:
        app = self.gui_module.ModernGUI()
        app._on_comparison_complete({"match": False, "score": 0, "error": "bad input"})
        self.assertEqual(app.result_label.text, "Error: bad input")
        self.assertEqual(app.result_label.text_color, "red")


if __name__ == "__main__":
    unittest.main()
