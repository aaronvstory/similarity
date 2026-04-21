from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch


class _DeepFaceRecorder:
    extract_calls = []
    represent_calls = []

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


deepface_module = types.ModuleType("deepface")
deepface_module.DeepFace = _DeepFaceRecorder
sys.modules["deepface"] = deepface_module


class TestFaceEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine_module = importlib.import_module("src.engine")
        self.engine_module = importlib.reload(self.engine_module)
        self.engine_module.FaceEngine._instance = None
        _DeepFaceRecorder.reset()

    def test_compare_images_uses_largest_face_and_embeddings(self) -> None:
        engine = self.engine_module.FaceEngine()

        with patch.object(engine, "validate_image_file"):
            result = engine.compare_images("image-a.jpg", "image-b.jpg")

        self.assertIsNone(result["error"])
        self.assertTrue(result["match"])
        self.assertLess(result["score"], 100.0)
        self.assertEqual(len(_DeepFaceRecorder.extract_calls), 2)
        self.assertEqual(len(_DeepFaceRecorder.represent_calls), 2)
        self.assertEqual(_DeepFaceRecorder.represent_calls[0]["img_path"], "large-a")
        self.assertEqual(_DeepFaceRecorder.represent_calls[1]["img_path"], "large-b")
        self.assertEqual(_DeepFaceRecorder.represent_calls[0]["detector_backend"], "skip")

    def test_identical_embeddings_map_to_full_score(self) -> None:
        engine = self.engine_module.FaceEngine()
        embedding = self.engine_module.np.asarray([1.0, 0.0, 0.0], dtype=float)

        distance = engine._cosine_distance(embedding, embedding)

        self.assertEqual(distance, 0.0)


if __name__ == "__main__":
    unittest.main()
