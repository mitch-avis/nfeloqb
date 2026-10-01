import subprocess
import sys
from pathlib import Path


def test_package_root_import_exposes_run_without_optimizer_dependencies() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import nfeloqb; print(callable(nfeloqb.run))",
        ],
        capture_output=True,
        cwd=repo_root,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"
