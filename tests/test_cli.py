from __future__ import annotations

import builtins
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
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


deepface_module = types.ModuleType("deepface")
deepface_module.DeepFace = _DeepFaceStub
sys.modules["deepface"] = deepface_module

cli_module = importlib.import_module("src.cli")
cli_module = importlib.reload(cli_module)
ProCLI = cli_module.ProCLI


class TestProCLI(unittest.TestCase):
    def setUp(self) -> None:
        with patch("src.cli.console.print"):
            self.cli = ProCLI()

    def test_import_does_not_require_tkinter(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tkinter":
                raise AssertionError("tkinter should not be imported at module import time")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=guarded_import):
            reloaded = importlib.reload(cli_module)

        self.assertTrue(hasattr(reloaded, "ProCLI"))

    def test_load_config_normalizes_invalid_existing_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "img1_keyword": "alpha",
                        "existing_file_mode": "bogus",
                    }
                ),
                encoding="utf-8",
            )

            self.cli.config_path = str(config_path)
            self.cli.config = {
                "img1_keyword": "extracted",
                "img2_keyword": "selfie",
                "extraction_keyword": "front",
                "padding_ratio": 0.175,
                "existing_file_mode": "index",
            }

            with patch("src.cli.console.print") as mock_print:
                self.cli.load_config()

        self.assertEqual(self.cli.config["img1_keyword"], "alpha")
        self.assertEqual(self.cli.config["existing_file_mode"], "index")
        self.assertTrue(any("existing_file_mode" in str(call.args[0]) for call in mock_print.call_args_list))

    def test_apply_runtime_config_rejects_invalid_padding(self) -> None:
        with self.assertRaisesRegex(ValueError, "padding_ratio"):
            self.cli.apply_runtime_config(padding_ratio=2.0)

    def test_get_available_path_forces_index_when_source_is_already_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "extracted.png"
            source.write_bytes(b"fake")

            self.cli.config["existing_file_mode"] = "skip"
            next_path = self.cli._get_available_path(tmpdir, source.name)

        self.assertEqual(Path(next_path).name, "extracted2.png")

    def test_batch_similarity_processes_deepest_folders_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent = root / "parent"
            child = parent / "child"
            child.mkdir(parents=True)

            rename_order: list[str] = []

            def walk_stub(_root):
                yield (str(root), ["parent"], [])
                yield (str(parent), ["child"], [])
                yield (str(child), [], [])

            def find_stub(dirpath: str, keyword: str) -> str | None:
                if dirpath in {str(parent), str(child)}:
                    return os.path.join(dirpath, f"{keyword}.jpg")
                return None

            with (
                patch.object(self.cli, "_ensure_models_initialized"),
                patch.object(self.cli, "_log_to_manifest"),
                patch.object(self.cli, "_find_image_with_keyword", side_effect=find_stub),
                patch.object(self.cli.engine, "compare_images", return_value={"match": True, "score": 91.0, "error": None}),
                patch("src.cli.os.walk", side_effect=walk_stub),
                patch("src.cli.os.rename", side_effect=lambda src, dst: rename_order.append(src)),
                patch("src.cli.console.print"),
            ):
                self.cli.run_batch_similarity(root_dir=str(root), confirm=False)

        self.assertEqual(rename_order, [str(child), str(parent)])


if __name__ == "__main__":
    unittest.main()
