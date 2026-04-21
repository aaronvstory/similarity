from __future__ import annotations

import io
import sys
import types
import unittest
from unittest.mock import patch

import main


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


class TestMainRouting(unittest.TestCase):
    def test_similarity_mode_routes_to_batch_similarity(self) -> None:
        deepface_module = types.ModuleType("deepface")
        deepface_module.DeepFace = _DeepFaceStub

        with patch.dict(sys.modules, {"deepface": deepface_module}):
            fake_cli = patch("src.cli.ProCLI").start()
            self.addCleanup(patch.stopall)

            fake_instance = fake_cli.return_value

            with patch.object(sys, "argv", ["main.py", "--mode", "similarity", "--root", "C:/temp/root", "--yes"]):
                main.main()

        fake_instance.apply_runtime_config.assert_called_once()
        fake_instance.run_batch_similarity.assert_called_once_with(root_dir="C:/temp/root", confirm=False)

    def test_compare_mode_requires_both_paths(self) -> None:
        stdout = io.StringIO()
        deepface_module = types.ModuleType("deepface")
        deepface_module.DeepFace = _DeepFaceStub

        with patch.dict(sys.modules, {"deepface": deepface_module}):
            fake_cli = patch("src.cli.ProCLI").start()
            self.addCleanup(patch.stopall)

            with (
                patch.object(sys, "argv", ["main.py", "--mode", "compare", "--img1", "one.jpg"]),
                patch("sys.stdout", stdout),
            ):
                with self.assertRaises(SystemExit) as cm:
                    main.main()

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("requires both --img1 and --img2", stdout.getvalue())
        fake_cli.return_value.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
