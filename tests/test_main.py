import sys
import unittest
from unittest.mock import patch

import main


class TestMainRouting(unittest.TestCase):
    def test_similarity_mode_routes_to_batch_similarity(self) -> None:
        fake_cli = patch("src.cli.ProCLI").start()
        self.addCleanup(patch.stopall)

        fake_instance = fake_cli.return_value

        with patch.object(sys, "argv", ["main.py", "--mode", "similarity", "--root", "C:/temp/root", "--yes"]):
            main.main()

        fake_instance.apply_runtime_config.assert_called_once()
        fake_instance.run_batch_similarity.assert_called_once_with(root_dir="C:/temp/root", confirm=False)


if __name__ == "__main__":
    unittest.main()
