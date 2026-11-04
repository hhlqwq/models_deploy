#!/usr/bin/env bash
# 从零恢复 Linux x86_64 SDK（Orbbec + Intel async_infer_lib）
#
# 逻辑：
#   1. 检查 dep/ 是否完整 → 完整则直接退出
#   2. 检查 orbbec_sdk 是否已在 dep/ → 在则跳过下载解压
#   3. 检查 async_infer_lib 源码是否已在 cache/ → 在则跳过 clone
#   4. 检查 async_infer SDK 是否已构建 → 在则跳过构建
#   5. 按需执行缺失步骤
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="${ASYNC_INFER_REPOSITORY:-ssh://git@git.realthon.com:2222/aitc/customer/jack/poc/async_infer_lib.git}"
branch="${ASYNC_INFER_BRANCH:-intel}"

dep_target=linux-x86_64
arch=x86_64
tag=v2.8.7

dest="$root/dep/$dep_target"
work="${SDK_RESTORE_WORKDIR:-$root/cache/$dep_target}"
archive="$work/orbbec.tar.gz"

# ── 工具检查 ──────────────────────────────────────────────────────────────
install_missing_tools() {
  local missing_packages=()
  local dependencies=(
    "curl:curl" "tar:tar" "git:git" "rsync:rsync"
    "ssh:openssh-client" "cmake:cmake"
  )
  for item in "${dependencies[@]}"; do
    local tool="${item%%:*}" package="${item##*:}"
    if ! command -v "$tool" >/dev/null 2>&1; then
      missing_packages+=("$package")
    fi
  done
  if [[ ${#missing_packages[@]} -eq 0 ]]; then
    echo "[DEP] 基础依赖检查通过"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[ERROR] 缺少 ${missing_packages[*]}，且无 apt-get" >&2; exit 1
  fi
  local apt_cmd=(apt-get)
  [[ "$(id -u)" -ne 0 ]] && apt_cmd=(sudo apt-get)
  "${apt_cmd[@]}" update
  DEBIAN_FRONTEND=noninteractive "${apt_cmd[@]}" install -y --no-install-recommends "${missing_packages[@]}"
  echo "[DEP] 所有基础依赖已安装"
}

# ── 完整性校验 ────────────────────────────────────────────────────────────
is_complete_sdk() {
  local sdk="$1"
  [[ -f "$sdk/orbbec_sdk/lib/OrbbecSDKConfig.cmake" ]] || return 1
  compgen -G "$sdk/orbbec_sdk/lib/libOrbbecSDK.so*" >/dev/null || return 1
  [[ -d "$sdk/orbbec_sdk/lib/extensions" ]] || return 1
  [[ -f "$sdk/async_infer_sdk/sdk/include/async_infer.h" ]] || return 1
  grep -q 'TID_MOBILENETV2' "$sdk/async_infer_sdk/sdk/include/async_infer.h" || return 1
  [[ -f "$sdk/async_infer_sdk/sdk/lib/libasync_infer.so" ]] || return 1
  [[ -f "$sdk/async_infer_sdk/example/files/config/mobilenetv2_config.json" ]] || return 1
  [[ -f "$sdk/async_infer_sdk/example/files/models/mobilenetv2/mobilenetv2.xml" ]] || return 1
  [[ -f "$sdk/async_infer_sdk/example/files/models/mobilenetv2/mobilenetv2.bin" ]] || return 1
}

# ── 快速退出：dep/ 已完整 ────────────────────────────────────────────────
if is_complete_sdk "$dest"; then
  echo "[OK] $dep_target SDK 完整，跳过恢复：$dest"
  exit 0
fi

models_src="$root/models"
if [[ ! -f "$models_src/mobilenetv2/mobilenetv2.xml" ||
      ! -f "$models_src/mobilenetv2/mobilenetv2.bin" ]]; then
  echo "[ERROR] 缺少 MobileNetV2 XML/BIN：$models_src/mobilenetv2/" >&2
  exit 1
fi

install_missing_tools
mkdir -p "$work" "$dest"

# -----------------------------------------------------------------------------
# 基础工具检查与自动安装
# -----------------------------------------------------------------------------
install_missing_tools() {
  local missing_packages=()

  # 格式：命令名:apt包名
  local dependencies=(
    "curl:curl"
    "tar:tar"
    "git:git"
    "rsync:rsync"
    "ssh:openssh-client"
    "cmake:cmake"
  )

  for item in "${dependencies[@]}"; do
    local tool="${item%%:*}"
    local package="${item##*:}"

    if command -v "$tool" >/dev/null 2>&1; then
      echo "[DEP] $tool 已安装"
    else
      echo "[DEP] 缺少 $tool，将自动安装：$package"
      missing_packages+=("$package")
    fi
  done

  if [[ ${#missing_packages[@]} -eq 0 ]]; then
    echo "[DEP] 基础依赖检查通过"
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[ERROR] 检测到缺失依赖，但当前系统没有 apt-get，无法自动安装。" >&2
    echo "        缺失软件包：${missing_packages[*]}" >&2
    exit 1
  fi

  local apt_cmd=(apt-get)

  # Docker 中通常是 root；如果不是 root，则尝试 sudo
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      apt_cmd=(sudo apt-get)
    else
      echo "[ERROR] 当前用户不是 root，且系统没有 sudo，无法自动安装依赖。" >&2
      exit 1
    fi
  fi

  echo "[DEP] 更新 APT 软件包索引..."
  "${apt_cmd[@]}" update

  echo "[DEP] 安装缺失依赖：${missing_packages[*]}"
  DEBIAN_FRONTEND=noninteractive \
    "${apt_cmd[@]}" install -y --no-install-recommends "${missing_packages[@]}"

  # 安装完成后二次校验
  for item in "${dependencies[@]}"; do
    local tool="${item%%:*}"

    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "[ERROR] 自动安装后仍未找到命令：$tool" >&2
      exit 1
    fi
  done

  echo "[DEP] 所有基础依赖已准备完成"
}

install_missing_tools
mkdir -p "$work" "$dest"

# ── Step 1: Orbbec SDK ───────────────────────────────────────────────────
if [[ -d "$dest/orbbec_sdk/lib" ]] && compgen -G "$dest/orbbec_sdk/lib/libOrbbecSDK.so*" >/dev/null; then
  echo "[1/4] Orbbec SDK 已存在于 dep/，跳过"
else
  if [[ -f "$archive" ]]; then
    echo "[1/4] 使用预下载的 Orbbec 包：$archive"
  else
    api="https://api.github.com/repos/orbbec/OrbbecSDK_v2/releases/tags/$tag"
    echo "[1/4] 查询 Orbbec $tag linux_$arch Release 资产"
    url="${ORBBEC_RELEASE_URL:-}"
    if [[ -z "$url" ]]; then
      url="$(curl -fsSL "$api" | grep -o 'https:[^" ]*OrbbecSDK[^" ]*linux_'"$arch"'[^" ]*\.tar\.gz' | head -1 || true)"
    fi
    if [[ -z "$url" ]]; then
      echo "[ERROR] 未找到 Orbbec $tag linux_$arch 资产。" >&2
      echo "        请在可联网机器下载后放到：$archive" >&2
      exit 1
    fi
    echo "[1/4] 下载 Orbbec $tag ($arch) 到 $archive"
    if ! curl -fL --retry 3 --progress-bar "$url" -o "$archive"; then
      rm -f "$archive"
      echo "[ERROR] Orbbec 下载失败。请在其它环境下载后放到：$archive" >&2
      exit 1
    fi
  fi

  echo "[1/4] 解压并组织 Orbbec SDK"
  rm -rf "$work/orbbec"
  mkdir -p "$work/orbbec"
  tar -xzf "$archive" -C "$work/orbbec"
  orbbec="$(find "$work/orbbec" -type d -name include -print -quit | xargs -r dirname)"
  [[ -n "$orbbec" && -f "$orbbec/lib/OrbbecSDKConfig.cmake" ]] || {
    echo "[ERROR] Orbbec 压缩包结构无效：$archive" >&2; exit 1
  }
  rm -rf "$dest/orbbec_sdk"
  rsync -a "$orbbec/" "$dest/orbbec_sdk/"
fi

# ── Step 2: AsyncInfer 源码 ──────────────────────────────────────────────
src="$work/async_infer_lib"

if [[ -d "$src/.git" ]]; then
  echo "[2/4] AsyncInfer 源码已存在于 cache/，跳过 clone"
  if [[ -n "$(git -c safe.directory="$src" -C "$src" status --porcelain)" ]]; then
    echo "[ERROR] SDK 缓存仓库存在未提交改动，不自动构建或切换：$src" >&2
    exit 1
  fi
  # 尝试 fetch 更新（无网络/SSH 不可用时静默跳过，禁止交互式提示）
  GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=no" \
    git -c safe.directory="$src" -C "$src" fetch --quiet --prune origin "$branch" </dev/null 2>/dev/null || true
  # 优先用 remote branch，回退到本地 branch，再回退到 HEAD commit
  revision_ref=""
  if git -c safe.directory="$src" -C "$src" rev-parse --verify --quiet "origin/$branch^{commit}" >/dev/null 2>&1; then
    revision_ref="origin/$branch"
  elif git -c safe.directory="$src" -C "$src" rev-parse --verify --quiet "$branch^{commit}" >/dev/null 2>&1; then
    revision_ref="$branch"
  else
    # detached HEAD — 直接用当前 commit
    revision_ref="$(git -c safe.directory="$src" -C "$src" rev-parse HEAD)"
  fi
  if [[ "$(git -c safe.directory="$src" -C "$src" rev-parse HEAD)" !=
        "$(git -c safe.directory="$src" -C "$src" rev-parse "$revision_ref")" ]]; then
    echo "[ERROR] SDK 缓存仓库与 $revision_ref 不一致，请先更新：$src" >&2
    exit 1
  fi
else
  echo "[2/4] 克隆 AsyncInfer 源码：$repo ($branch)"
  if ! git clone --progress --branch "$branch" "$repo" "$src"; then
    echo "[ERROR] AsyncInfer 克隆失败。请确认 Git 网络和 SSH 凭据。" >&2
    echo "        离线接续：在可联网机器执行 git clone --branch $branch $repo，" >&2
    echo "        再将完整目录放到：$src" >&2
    exit 1
  fi
  revision_ref="origin/$branch"
fi
revision="$(git -c safe.directory="$src" -C "$src" rev-parse HEAD)"
echo "[2/4] AsyncInfer 固定提交：$revision"

# ── Step 3: 构建 AsyncInfer SDK ──────────────────────────────────────────
sdk="$src/output/sdk"
echo "[3/4] 构建 AsyncInfer SDK Release"
_ov="$(find /opt/intel -maxdepth 2 -name setupvars.sh -path '*/l_openvino*' 2>/dev/null | head -1)"
if [[ -n "$_ov" && -f "$_ov" ]]; then
  set +u; source "$_ov" 2>/dev/null; set -u
fi
BUILD_TYPE=Release AISDK_GIT_HASH="$revision" AISDK_GIT_DESC="$revision" bash "$src/build.sh" sdk
[[ -f "$sdk/include/async_infer.h" ]] || {
  echo "[ERROR] AsyncInfer SDK 构建失败，产物不存在：$sdk" >&2; exit 1
}

# ── Step 4: 同步到 dep/，保留已有文件 ───────────────────────────────────
echo "[4/4] 同步 AsyncInfer SDK 到 $dest"
mkdir -p "$dest/async_infer_sdk/sdk" "$dest/async_infer_sdk/example/files"
rsync -a "$sdk/" "$dest/async_infer_sdk/sdk/"
rsync -a "$src/example/files/" "$dest/async_infer_sdk/example/files/"

# ── Step 5: 模型 ────────────────────────────────────────────────────────
models_src="$root/models"
models_dst="$dest/async_infer_sdk/example/files/models"
if [[ -d "$models_src/yolo" ]]; then
  mkdir -p "$models_dst/yolo"
  rsync -a "$models_src/yolo/" "$models_dst/yolo/"
  echo "[5/5] 已拷贝 YOLO 模型到 $models_dst/yolo/"
fi
mkdir -p "$models_dst/mobilenetv2"
rsync -a "$models_src/mobilenetv2/mobilenetv2.xml" \
         "$models_src/mobilenetv2/mobilenetv2.bin" "$models_dst/mobilenetv2/"
echo "[5/5] 已拷贝 MobileNetV2 模型到 $models_dst/mobilenetv2/"

# ── 最终校验 ────────────────────────────────────────────────────────────
is_complete_sdk "$dest" || {
  echo "[ERROR] SDK 恢复后完整性校验失败：$dest" >&2; exit 1
}

echo "[OK] 已恢复并校验 $dep_target SDK：$dest"
