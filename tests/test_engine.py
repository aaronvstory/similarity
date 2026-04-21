from __future__ import annotations

import importlib
import sys
import types
import unittest
from typing import Any, ClassVar
from unittest.mock import patch


class _DeepFaceRecorder:
    extract_calls: ClassVar[list[dict[str, Any]]] = []
    represent_calls: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def reset(cls) -> None:
        cls.extract_calls = []
        cls.represent_calls = []

    @staticmethod
    def build_model(model_name: str):
        return model_name

    @classmethod
    def extract_faces(cls, **kwargs):
        cls.extract_calls.append(kwargs)
        if kwargs["img_path"] == "image-a.jpg":
            return [
                {"face": "small-a", "facial_area": {"w": 10, "h": 10}},
                {"face": "large-a", "facial_area": {"w": 30, "h": 25}},
            ]
        if kwargs["img_path"] == "image-b.jpg":
            return [
                {"face": "small-b", "facial_area": {"w": 12, "h": 12}},
                {"face": "large-b", "facial_area": {"w": 40, "h": 20}},
            ]
        raise AssertionError(f"Unexpected image path: {kwargs['img_path']}")

    @classmethod
    def represent(cls, **kwargs):
        cls.represent_calls.append(kwargs)
        embeddings = {
            "large-a": [1.0, 0.0, 0.0],
            "large-b": [0.8, 0.6, 0.0],
        }
        return [{"embedding": embeddings[kwargs["img_path"]]}]

def _build_deepface_module() -> types.ModuleType:
    deepface_module = types.ModuleType("deepface")
    deepface_module.DeepFace = _DeepFaceRecorder
    return deepface_module


class TestFaceEngine(unittest.TestCase):
    def setUp(self) -> None:
        self._original_engine_module = sys.modules.pop("src.engine", None)
        self.addCleanup(self._restore_engine_module)

        deepface_patcher = patch.dict(sys.modules, {"deepface": _build_deepface_module()})
        deepface_patcher.start()
        self.addCleanup(deepface_patcher.stop)

        self.engine_module = importlib.import_module("src.engine")
        self.engine_module.FaceEngine._instance = None
        _DeepFaceRecorder.reset()

    def _restore_engine_module(self) -> None:
        sys.modules.pop("src.engine", None)
        if self._original_engine_module is not None:
            sys.modules["src.engine"] = self._original_engine_module

    def test_compare_images_uses_largest_face_and_embeddings(self) -> None:
        engine = self.engine_module.FaceEngine()

        with patch.object(engine, "validate_image_file"):
            result = engine.compare_images("image-a.jpg", "image-b.jpg")

        self.assertIsNone(result["error"])
        self.assertTrue(result["match"])
        self.assertAlmostEqual(result["score"], 94.12, places=2)
        self.assertEqual(len(_DeepFaceRecorder.extract_calls), 2)
        self.assertEqual(len(_DeepFaceRecorder.represent_calls), 2)
        self.assertEqual(_DeepFaceRecorder.represent_calls[0]["img_path"], "large-a")
        self.assertEqual(_DeepFaceRecorder.represent_calls[1]["img_path"], "large-b")
        self.assertTrue(
            all(call["detector_backend"] == "skip" for call in _DeepFaceRecorder.represent_calls)
        )

    def test_identical_embeddings_map_to_full_score(self) -> None:
        engine = self.engine_module.FaceEngine()
        with patch.object(engine, "validate_image_file"):
            result = engine.compare_images("image-a.jpg", "image-a.jpg")

        self.assertIsNone(result["error"])
        self.assertTrue(result["match"])
        self.assertEqual(result["score"], 100.0)

    def test_shutdown_falls_back_when_cancel_futures_is_unsupported(self) -> None:
        engine = self.engine_module.FaceEngine()

        class ExecutorStub:
            def __init__(self) -> None:
                self.calls = []

            def shutdown(self, wait=False, cancel_futures=False):
                self.calls.append((wait, cancel_futures))
                if cancel_futures:
                    raise TypeError("cancel_futures unsupported")

        executor = ExecutorStub()
        engine._executor = executor

        engine.shutdown()

        self.assertEqual(executor.calls, [(False, True), (False, False)])
        self.assertIsNone(engine._executor)


if __name__ == "__main__":
    unittest.main()
