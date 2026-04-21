from __future__ import annotations

import importlib
import json
import shutil
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


def _load_cli_module_with_stub():
    deepface_module = types.ModuleType("deepface")
    deepface_module.DeepFace = _DeepFaceStub

    with patch.dict(sys.modules, {"deepface": deepface_module}, clear=False):
        sys.modules.pop("src.engine", None)
        sys.modules.pop("src.cli", None)
        cli_module = importlib.import_module("src.cli")
        cli_module = importlib.reload(cli_module)
    return cli_module


class TestCLIIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.cli_module = _load_cli_module_with_stub()
        with patch.object(self.cli_module.console, "print"):
            self.cli = self.cli_module.ProCLI()

    def _create_seed_fixture(self, root: Path) -> Path:
        seed_root = root / "seed-fixture"
        source_folder = seed_root / "DASHER 100 - Jane Doe (123)_front"
        source_folder.mkdir(parents=True)
        (source_folder / "fronnt_scan.png").write_bytes(b"front-image")
        return seed_root

    def test_batch_extract_then_similarity_renames_score_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_root = self._create_seed_fixture(root)
            run_root = root / "run-root"
            shutil.copytree(seed_root, run_root)

            def fake_extract_face(input_path: str, output_path: str, padding: float = 0.175) -> float:
                Path(output_path).write_bytes(b"extracted-image")
                return 0.91

            with (
                patch.object(self.cli, "_ensure_models_initialized"),
                patch.object(self.cli_module.console, "print"),
                patch.object(self.cli.engine, "extract_face", side_effect=fake_extract_face),
            ):
                self.cli.run_batch_extraction(
                    root_dir=str(run_root),
                    confirm=False,
                )

            extracted_output = run_root / "DASHER 100 - Jane Doe (123)_front" / "extracted.png"
            self.assertTrue(extracted_output.exists())

            # Use fuzzy keywords intentionally to validate approximate matching.
            self.cli.config["img1_keyword"] = "extractd"
            self.cli.config["img2_keyword"] = "front"
            with (
                patch.object(self.cli, "_ensure_models_initialized"),
                patch.object(self.cli_module.console, "print"),
                patch.object(
                    self.cli.engine,
                    "compare_images",
                    return_value={"match": True, "score": 98.7, "error": None},
                ),
            ):
                self.cli.run_batch_similarity(
                    root_dir=str(run_root),
                    confirm=False,
                )

            renamed = run_root / "DASHER 98 - Jane Doe (123)_front"
            original = run_root / "DASHER 100 - Jane Doe (123)_front"
            self.assertTrue(renamed.exists())
            self.assertFalse(original.exists())

            manifest = run_root / "manifest.json"
            self.assertTrue(manifest.exists())
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(manifest_data.get("operations", [])), 2)
            op_types = [op.get("type") for op in manifest_data["operations"]]
            self.assertIn("batch_extraction", op_types)
            self.assertIn("batch_similarity", op_types)


if __name__ == "__main__":
    unittest.main()
