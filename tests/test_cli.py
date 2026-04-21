import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import ProCLI


class TestProCLI(unittest.TestCase):
    def setUp(self) -> None:
        with patch("src.cli.console.print"):
            self.cli = ProCLI()

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
                patch("src.cli.Confirm.ask", return_value=True),
                patch("src.cli.console.print"),
            ):
                self.cli.run_batch_similarity(root_dir=str(root), confirm=False)

        self.assertEqual(rename_order, [str(child), str(parent)])


if __name__ == "__main__":
    unittest.main()
