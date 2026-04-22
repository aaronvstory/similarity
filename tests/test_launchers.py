from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_fake_python(path: Path, *, version_ok: bool, tkinter_ok: bool) -> None:
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
            mkdir -p "$3/bin"
            printf '%s\\n' '#!/usr/bin/env bash' 'exec python "$@"' > "$3/bin/python"
            chmod +x "$3/bin/python"
            exit 0
        fi
        if [ "$1" = "-" ]; then
            payload="$(cat)"
            if printf '%s' "$payload" | grep -q "sys.version_info"; then
                {"exit 0" if version_ok else "exit 1"}
            fi
            if printf '%s' "$payload" | grep -q "import tkinter"; then
                {"exit 0" if tkinter_ok else "exit 1"}
            fi
            exit 0
        fi
        exit 0
        """
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestCommandLaunchers(unittest.TestCase):
    def _run_launcher(
        self,
        launcher_name: str,
        *,
        candidate_list: str,
        fake_bins: dict[str, tuple[bool, bool]],
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            launcher_src = REPO_ROOT / launcher_name
            launcher_dst = tmp_path / launcher_name
            shutil.copy2(launcher_src, launcher_dst)
            launcher_dst.chmod(launcher_dst.stat().st_mode | stat.S_IXUSR)

            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            for name, (version_ok, tkinter_ok) in fake_bins.items():
                _write_fake_python(
                    bin_dir / name,
                    version_ok=version_ok,
                    tkinter_ok=tkinter_ok,
                )

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
            env["SIMILARITY_LAUNCHER_DRY_RUN"] = "1"
            env["SIMILARITY_PYTHON_CANDIDATES"] = candidate_list

            return subprocess.run(
                ["/bin/bash", str(launcher_dst)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                env=env,
                timeout=20,
            )

    def test_run_gui_command_skips_incompatible_default_python(self) -> None:
        result = self._run_launcher(
            "run_gui.command",
            candidate_list="python3 python3.11",
            fake_bins={
                "python3": (False, False),  # Simulates Python 3.14 without _tkinter.
                "python3.11": (True, True),
            },
        )
        output = f"{result.stdout}\n{result.stderr}"
        self.assertEqual(result.returncode, 0, msg=output)
        self.assertIn("[INFO] Using Python: ", output)
        self.assertIn("python3.11", output)
        self.assertIn("Dry run mode enabled", output)

    def test_run_gui_command_reports_no_tkinter_python(self) -> None:
        result = self._run_launcher(
            "run_gui.command",
            candidate_list="python3",
            fake_bins={
                "python3": (False, False),
            },
        )
        output = f"{result.stdout}\n{result.stderr}"
        self.assertNotEqual(result.returncode, 0, msg=output)
        self.assertIn("compatible Python with tkinter support", output)

    def test_run_cli_command_prefers_supported_version(self) -> None:
        result = self._run_launcher(
            "run_cli.command",
            candidate_list="python3 python3.11",
            fake_bins={
                "python3": (False, False),  # Version check fails.
                "python3.11": (True, True),
            },
        )
        output = f"{result.stdout}\n{result.stderr}"
        self.assertEqual(result.returncode, 0, msg=output)
        self.assertIn("[INFO] Using Python: ", output)
        self.assertIn("python3.11", output)
        self.assertIn("Dry run mode enabled", output)


if __name__ == "__main__":
    unittest.main()
