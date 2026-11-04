"""RDK 公共文件传输及四个总入口的发布,不处理模型推理或评分."""

from datetime import date
from pathlib import Path, PurePosixPath
import posixpath
import shlex

import paramiko

from tools.rdk_config import load_config


ROOT = Path(__file__).resolve().parents[1]


def connection(config):
    """读取本地连接配置,使用已固定的板端主机密钥建立连接."""
    board = config["board"]
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / "configs/platforms/s100.local.known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(hostname=board["host"], username=board["user"], password=board.get("password"), timeout=15)
    return client


def upload(sftp, destination, files):
    """按清单上传普通文件,目标必须位于指定目录之内."""
    for index, (source, name) in enumerate(files, 1):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not Path(source).is_file():
            raise ValueError(f"上传文件或相对路径无效: {name}.")
        target = posixpath.join(destination, name)
        parent = "/"
        for part in PurePosixPath(target).parent.parts[1:]:
            parent = posixpath.join(parent, part)
            try:
                sftp.stat(parent)
            except FileNotFoundError:
                sftp.mkdir(parent)
        sftp.put(str(source), target)
        if index % 25 == 0 or index == len(files) or len(files) <= 20:
            print(f"[上传] {index}/{len(files)} {name}", flush=True)


def publish_entries(config):
    """将板端总入口和本地回传设置放到统一部署根目录的 deploy 下."""
    results = config["results"]
    if not results["host"] or not Path(results["root"]).is_absolute() or results["root"] == "/":
        raise ValueError("请在 S100 本地配置填写 results.host 和 results.root.")
    temporary = Path("/tmp/hailongcodex") / date.today().isoformat() / "s100/deploy"
    temporary.mkdir(parents=True, exist_ok=True)
    settings = temporary / "platform.local.sh"
    values = {"RDK_RESULTS_HOST": results["host"], "RDK_RESULTS_ROOT": results["root"],
              "RDK_RESULTS_KEY": results["key"]}
    settings.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()), encoding="utf-8")
    settings.chmod(0o600)
    files = [(ROOT / "rdk_models/deploy" / name, name) for name in
             ("compile.sh", "evaluate.sh", "compile_board.sh", "evaluate_board.sh", "common.sh")]
    files.append((settings, "platform.local.sh"))
    with connection(config) as client, client.open_sftp() as sftp:
        upload(sftp, str(PurePosixPath(config["board"]["models_root"]) / "deploy"), files)
        sftp.chmod(str(PurePosixPath(config["board"]["models_root"]) / "deploy/platform.local.sh"), 0o600)


if __name__ == "__main__":
    publish_entries(load_config())
