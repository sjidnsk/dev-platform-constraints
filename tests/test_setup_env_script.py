import shutil
import subprocess
import unittest
from pathlib import Path


class SetupEnvScriptTests(unittest.TestCase):
    def test_environment_yml_declares_conda_runtime_dependencies(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        environment_yml = repo_root / "environment.yml"

        content = environment_yml.read_text(encoding="utf-8")

        self.assertIn("name: lunar-explorer", content)
        self.assertIn("python=3.12", content)
        self.assertIn("numpy>=1.26,<2.3", content)
        self.assertIn("libblas=*=*openblas", content)
        self.assertIn("pip", content)

    def test_project_metadata_requires_python_312(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        pyproject = repo_root / "pyproject.toml"

        content = pyproject.read_text(encoding="utf-8")

        self.assertIn('requires-python = ">=3.12,<3.13"', content)

    def test_powershell_dry_run_prints_conda_setup_steps(self) -> None:
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            self.skipTest("PowerShell is required for setup_env.ps1")

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "setup_env.ps1"

        result = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-DryRun",
                "-RunValidation",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("DRY RUN", output)
        self.assertIn("D:\\conda_envs\\lunar-explorer", output)
        self.assertIn("conda env create/update", output)
        self.assertIn("environment.yml", output)
        self.assertIn("conda install -p", output)
        self.assertIn("-c conda-forge", output)
        self.assertIn("python=3.12", output)
        self.assertIn("assert sys.version_info[:2] == (3, 12)", output)
        self.assertIn("conda run -p", output)
        self.assertIn("PYTHONPATH", output)
        self.assertNotIn("pip install -e", output)
        self.assertIn("python -m unittest discover -s tests", output)
        self.assertIn("scripts\\run_minimal_closure.py", output)
        self.assertNotIn("python -m venv", output)

    def test_ubuntu_bash_dry_run_prints_conda_setup_steps(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is required for setup_env.sh")

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "setup_env.sh"

        syntax = subprocess.run([bash, "-n", str(script)], cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(syntax.returncode, 0, syntax.stdout + syntax.stderr)

        result = subprocess.run(
            [
                bash,
                str(script),
                "--dry-run",
                "--run-validation",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("DRY RUN", output)
        self.assertIn("lunar-explorer", output)
        self.assertIn("conda env create/update", output)
        self.assertIn("environment.yml", output)
        self.assertIn("conda install -p", output)
        self.assertIn("-c conda-forge", output)
        self.assertIn("python=3.12", output)
        self.assertIn("assert sys.version_info[:2] == (3, 12)", output)
        self.assertIn("conda run -p", output)
        self.assertIn("PYTHONPATH", output)
        self.assertNotIn("pip install -e", output)
        self.assertIn("python -m unittest discover -s tests", output)
        self.assertIn("scripts/run_minimal_closure.py", output)
        self.assertNotIn("python -m venv", output)


if __name__ == "__main__":
    unittest.main()
