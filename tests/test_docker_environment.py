"""验证镜像验收失败不发布、成功后发布及新增模型覆盖,不运行真实 Docker."""

from datetime import date
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DockerEnvironmentTest(unittest.TestCase):
    """用命令替身检查正式构建入口的交付顺序."""

    def setUp(self):
        """准备隔离的构建上下文和 Docker 替身,不读取平台本地配置."""
        base = Path("/tmp/hailongcodex") / date.today().isoformat() / "docker_environment_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(dir=base))
        self.platform = self.work / "docker/mtk_g720"
        self.platform.mkdir(parents=True)
        for name in ("build_image.sh", "Dockerfile", "requirements.txt", "yolov5-requirements.txt",
                     "setup_container.sh", "verify_environment.py"):
            shutil.copy2(ROOT / "docker/mtk_g720" / name, self.platform / name)
        (self.work / "mtk_models/deploy").mkdir(parents=True)
        shutil.copy2(ROOT / "mtk_models/deploy/compile.sh", self.work / "mtk_models/deploy/compile.sh")
        (self.work / "tools").mkdir()
        (self.work / "sdk/offline_tool").mkdir(parents=True)
        (self.work / "gai").mkdir()
        (self.work / "tools/storage.sh").write_text(
            f"export MTK_HOST_SDK_ROOT={self.work}/sdk\n"
            f"export MTK_GAI_SDK_ROOT={self.work}/gai\n"
            "export MTK_G720_IMAGE=test-only:current\n")
        (self.work / "bin").mkdir()
        docker = self.work / "bin/docker"
        docker.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                          "with open(os.environ['DOCKER_CALLS'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                          "if sys.argv[1]=='run': sys.exit(int(os.environ['VERIFY_EXIT']))\n")
        docker.chmod(0o755)
        self.log = self.work / "calls.jsonl"

    def run_build(self, verification_exit):
        """运行真实构建脚本,只替换 Docker 命令并收集调用顺序."""
        environment = dict(os.environ, PATH=str(self.work / "bin") + os.pathsep + os.environ["PATH"],
                           DOCKER_CALLS=str(self.log), VERIFY_EXIT=str(verification_exit))
        result = subprocess.run(["bash", str(self.platform / "build_image.sh")],
                                env=environment, capture_output=True, text=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_failed_verification_does_not_publish(self):
        """验收失败必须保留正式标签,不能发布或删除旧环境."""
        result, calls = self.run_build(7)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual([call[0] for call in calls], ["buildx", "run"])

    def test_success_publishes_after_verification(self):
        """仅验收成功后发布正式镜像并移除候选标签."""
        result, calls = self.run_build(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[0] for call in calls], ["buildx", "run", "tag", "image"])
        self.assertEqual(calls[2], ["tag", "test-only:current-candidate", "test-only:current"])
        self.assertEqual(calls[3], ["image", "rm", "test-only:current-candidate"])

    def test_new_model_without_check_stops_build(self):
        """总入口新增模型但未补验收时,在构建前直接停止."""
        entry = self.work / "mtk_models/deploy/compile.sh"
        entry.write_text(entry.read_text().replace("MODELS=(", "MODELS=(\n    test/new_model"))
        result, calls = self.run_build(0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("new_model", result.stderr)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
