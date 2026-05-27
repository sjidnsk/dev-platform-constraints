import unittest
from pathlib import Path


class RepositoryConventionTests(unittest.TestCase):
    def test_agents_records_chinese_readme_and_comment_policy(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        agents = (repo_root / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn("README 使用中文编写", agents)
        self.assertIn("代码注释和 docstring 使用中文编写", agents)
        self.assertIn("禁止批量删除文件或目录", agents)

    def test_readme_uses_chinese_sections(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        readme = (repo_root / "README.md").read_text(encoding="utf-8")

        self.assertIn("## 范围", readme)
        self.assertIn("## 开发状态", readme)
        self.assertIn("## 运行", readme)
        self.assertIn("## 假设", readme)
        self.assertIn("PROJECT_BOUNDARY.md", readme)

    def test_project_boundary_is_formal_repository_document(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        boundary_path = repo_root / "PROJECT_BOUNDARY.md"
        boundary = boundary_path.read_text(encoding="utf-8")

        self.assertTrue(boundary_path.exists())
        for heading in ("## 定位", "## 负责范围", "## 不负责范围", "## 对外接口边界", "## 当前阶段原则"):
            with self.subTest(heading=heading):
                self.assertIn(heading, boundary)
        self.assertIn("model-explorer", boundary)

    def test_generate_example_data_is_documented_as_generated_artifact(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        script = (repo_root / "scripts" / "generate_example_data.py").read_text(encoding="utf-8")

        self.assertIn("sample_grid.npz", script)
        self.assertIn("scripts/generate_example_data.py 是生成型脚本", readme)
        self.assertIn("不是核心运行依赖", readme)
        self.assertIn("不提交版本库", readme)

    def test_gitignore_covers_python_caches_tool_caches_and_generated_data(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")

        for pattern in (
            "__pycache__/",
            "*.py[cod]",
            "*.egg-info/",
            ".pytest_cache/",
            ".ruff_cache/",
            ".mypy_cache/",
            "data/*.npz",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, gitignore)


if __name__ == "__main__":
    unittest.main()
