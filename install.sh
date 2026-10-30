#!/usr/bin/env bash
# xMIGCS 一键安装（脚本所在目录即为仓库根；可从任意 cwd 执行）
#
# - Ubuntu 22.04：Python 3.10 虚拟环境（兼容 ROS humble 的 rclpy/pinocchio）+ sptlib cp310
# - Ubuntu 24.04：Python 3.12 虚拟环境 + sptlib cp312
#
# 环境变量（可选）:
#   SKIP_PINOCCHIO=1        不安装 ros pinocchio
#   SKIP_UV_INSTALL=1       不尝试 curl 安装 uv
#   SKIP_XLOG=1             跳过 xlog 预编译包下载与 shell 配置
#   SKIP_BODYCTRL_MSGS=1    跳过 bodyctrl_msgs 的 .deb 安装（需 sudo / dpkg）
#   若当前运行用户名（sudo 时取 SUDO_USER）转小写为 ubuntu，也会自动跳过 xlog 与 bodyctrl_msgs
#
# xlog 与 shell 片段路径：使用「当前登录用户」家目录 /home/<用户名>（sudo 时依 SUDO_USER，避免写到 /root）。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

XMIGCS_RUN_USER="${SUDO_USER:-${USER:-}}"
if [[ -z "$XMIGCS_RUN_USER" ]]; then
  XMIGCS_RUN_USER="$(id -un 2>/dev/null || true)"
fi
XMIGCS_RUN_USER_LOWER="${XMIGCS_RUN_USER,,}"
SKIP_VENDOR_PACKAGES_FOR_UBUNTU_USER=0
if [[ "$XMIGCS_RUN_USER_LOWER" == "ubuntu" ]]; then
  SKIP_VENDOR_PACKAGES_FOR_UBUNTU_USER=1
fi
# Ubuntu 22.04 — sptlib cp310
SPTLIB_WHEEL_URL_22="${SPTLIB_WHEEL_URL:-http://10.10.250.160:5000/download/evt1%E8%84%9A%E8%B8%9D/sptlib_python-0.1.0-cp310-cp310-linux_x86_64.whl}"
SPTLIB_WHEEL_FILE_22="sptlib_python-0.1.0-cp310-cp310-linux_x86_64.whl"
# Ubuntu 24.04 — sptlib cp312
SPTLIB_WHEEL_URL_24="http://10.10.250.160:5000/download/evt2%E8%84%9A%E8%B8%9D/sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl"
SPTLIB_WHEEL_FILE_24="sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl"

log() { printf '%s\n' "$*"; }
die() { printf '错误: %s\n' "$*" >&2; exit 1; }

# 资源安装用的「当前登录用户」家目录：/home/<用户名>。用 sudo 跑脚本时不用 /root。
xmigcs_install_user_home() {
  if [[ -n "${SUDO_USER:-}" ]]; then
    local uh
    uh="$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6)"
    if [[ -n "${uh:-}" && -d "$uh" ]]; then
      printf '%s' "$uh"
      return 0
    fi
  fi
  printf '%s' "${HOME}"
}

ensure_path_uv() {
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
}

UB_VER=""
if [[ -f /etc/os-release ]]; then
  # shellcheck source=/dev/null
  . /etc/os-release
  if [[ "${ID:-}" == "ubuntu" ]]; then
    UB_VER="${VERSION_ID:-}"
  fi
fi
if [[ "$UB_VER" != "22.04" && "$UB_VER" != "24.04" ]]; then
  die "本脚本仅支持 Ubuntu 22.04 或 24.04，当前 VERSION_ID=${UB_VER:-未知或非 Ubuntu}"
fi

log "检测到 Ubuntu ${UB_VER}，使用对应安装流程。"

if ! command -v curl >/dev/null 2>&1; then
  die "需要 curl，请先安装（例如 apt install curl）"
fi

ensure_path_uv

if ! command -v uv >/dev/null 2>&1; then
  if [[ "${SKIP_UV_INSTALL:-0}" == "1" ]]; then
    die "未找到 uv，且已设置 SKIP_UV_INSTALL=1，请自行安装 uv 并加入 PATH"
  fi
  log "安装 uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ensure_path_uv
fi
command -v uv >/dev/null 2>&1 || die "uv 不可用，请检查 PATH（常见路径: ~/.local/bin）"

if [[ "$UB_VER" == "22.04" ]]; then
  log "创建 Python 3.10 虚拟环境并同步依赖..."
  uv venv --python 3.10
  uv sync --no-install-project
else
  log "创建 Python 3.12 虚拟环境并同步依赖..."
  uv venv --python 3.12
  uv sync --no-install-project
fi

# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"
export XMIGCS_DEV=1
log "清理源码目录中的旧 Cython 扩展，避免 editable 安装时优先导入 .so..."
find "$ROOT/src/xmigcs" -type f \( -name "*.cpython-*.so" -o -name "*.abi3.so" \) -delete
uv pip uninstall xmigcs -y >/dev/null 2>&1 || true
uv pip install -e .

if ! command -v wget >/dev/null 2>&1; then
  die "需要 wget 以下载 sptlib，请先安装（例如 apt install wget）"
fi

if [[ "$UB_VER" == "22.04" ]]; then
  log "安装本地 sptlib（cp310）..."
  # rm -f "$ROOT/$SPTLIB_WHEEL_FILE_22"
  # wget -O "$ROOT/$SPTLIB_WHEEL_FILE_22" "$SPTLIB_WHEEL_URL_22" \
  #   || die "sptlib 下载失败，请检查网络/VPN 或设置 SPTLIB_WHEEL_URL"
  uv pip install "$ROOT/vendor_packages/$SPTLIB_WHEEL_FILE_22"
else
  log "安装本地 sptlib（cp312）..."
  # rm -f "$ROOT/$SPTLIB_WHEEL_FILE_24"
  # wget -O "$ROOT/$SPTLIB_WHEEL_FILE_24" "$SPTLIB_WHEEL_URL_24" \
  #   || die "sptlib 下载失败，请检查网络/VPN"
  uv pip install "$ROOT/vendor_packages/$SPTLIB_WHEEL_FILE_24"
fi

if [[ "${SKIP_PINOCCHIO:-0}" != "1" ]]; then
  if [[ -n "${ROS_DISTRO:-}" ]]; then
    log "安装 ros-${ROS_DISTRO}-pinocchio（需要 sudo）..."
    sudo apt-get update -qq
    sudo apt-get install -y "ros-${ROS_DISTRO}-pinocchio"
  else
    log "未设置 ROS_DISTRO，跳过 apt 安装 pinocchio（仿真/开发可设 SKIP_PINOCCHIO=1 消除本提示；真机请先 source ROS setup）"
  fi
else
  log "已设置 SKIP_PINOCCHIO=1，跳过 ros pinocchio"
fi

log "校验 xmigcs 可导入..."
"$ROOT/.venv/bin/python" -c "import xmigcs"

if [[ "${SKIP_XLOG:-0}" != "1" && "$SKIP_VENDOR_PACKAGES_FOR_UBUNTU_USER" != "1" ]]; then
  log "安装 xlog 预编译包..."
  XMIGCS_USER_HOME="$(xmigcs_install_user_home)"
  XLOG_PREFIX="${XMIGCS_USER_HOME}/xlogs_depends/xlog_v1.0.1"
  mkdir -p "$XLOG_PREFIX"
  if [[ "$UB_VER" == "22.04" ]]; then
    XLOG_TAR_NAME="xlog_v1.0.1_standalone_humble_linux_x86_64.tar.gz"
    XLOG_PY_SITE="python3.10"
  else
    XLOG_TAR_NAME="xlog_v1.0.1_standalone_jazzy_linux_x86_64.tar.gz"
    XLOG_PY_SITE="python3.12"
  fi
  XLOG_TAR_URL="http://10.10.250.160:5000/download/xlog/${XLOG_TAR_NAME}"
  XLOG_LOCAL_TAR_PATH="${ROOT}/vendor_packages/${XLOG_TAR_NAME}"

  # wget -O "$XLOG_TAR_PATH" "$XLOG_TAR_URL" || die "xlog 下载失败，请检查网络"
  log "解压本地 xlog 文件..."
  tar -C "$XLOG_PREFIX" -xf "$XLOG_LOCAL_TAR_PATH" || die "解压本地文件 ${XLOG_LOCAL_TAR_PATH} 失败，请检查文件是否存在此目录"
  # rm -f "$XLOG_TAR_PATH"

  append_xlog_to_rc() {
    local rc=$1
    [[ -f "$rc" ]] || touch "$rc"
    if grep -q "xMIGCS install.sh: xlog" "$rc" 2>/dev/null; then
      log "xlog 环境片段已存在于 $(basename "$rc")，跳过追加"
      return 0
    fi
    cat <<EOF >>"$rc"

# xMIGCS install.sh: xlog
export XLOG_PREFIX="${XLOG_PREFIX}"
export CMAKE_PREFIX_PATH="\$XLOG_PREFIX:\${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="\$XLOG_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="\$XLOG_PREFIX/lib/${XLOG_PY_SITE}/site-packages:\${PYTHONPATH:-}"
# end xMIGCS xlog
EOF
  }

  append_xlog_to_rc "${XMIGCS_USER_HOME}/.bashrc"
  append_xlog_to_rc "${XMIGCS_USER_HOME}/.zshrc"

  export XLOG_PREFIX
  export CMAKE_PREFIX_PATH="${XLOG_PREFIX}:${CMAKE_PREFIX_PATH:-}"
  export LD_LIBRARY_PATH="${XLOG_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${XLOG_PREFIX}/lib/${XLOG_PY_SITE}/site-packages:${PYTHONPATH:-}"

  log "xlog 已解压到 ${XLOG_PREFIX}，环境变量已写入 ${XMIGCS_USER_HOME}/.bashrc 与 ${XMIGCS_USER_HOME}/.zshrc（未重复追加时）。"
else
  if [[ "$SKIP_VENDOR_PACKAGES_FOR_UBUNTU_USER" == "1" ]]; then
    log "当前运行用户名为 ${XMIGCS_RUN_USER}，跳过 xlog"
  else
    log "已设置 SKIP_XLOG=1，跳过 xlog"
  fi
fi

if [[ "${SKIP_BODYCTRL_MSGS:-0}" != "1" && "$SKIP_VENDOR_PACKAGES_FOR_UBUNTU_USER" != "1" ]]; then
  log "安装 bodyctrl_msgs（ROS 接口 .deb，需 sudo）..."
  INSTALL_CACHE="${ROOT}/.xmigcs_install_cache"
  mkdir -p "$INSTALL_CACHE"
  if [[ "$UB_VER" == "22.04" ]]; then
    BODYCTRL_DEB_NAME="ros-humble-bodyctrl-msgs_0.0.0-0jammy_amd64.deb"
  else
    BODYCTRL_DEB_NAME="ros-jazzy-bodyctrl-msgs_0.0.0-0noble_amd64.deb"
  fi
  BODYCTRL_DEB_URL="http://10.10.250.160:5000/download/ros_interface/${BODYCTRL_DEB_NAME}"
  BODYCTRL_DEB_PATH="${ROOT}/vendor_packages/${BODYCTRL_DEB_NAME}"

  # wget -O "$BODYCTRL_DEB_PATH" "$BODYCTRL_DEB_URL" || die "bodyctrl_msgs .deb 下载失败，请检查网络"
  log "利用本地 bodyctrl_msgs .deb 文件进行安装..."
  if ! sudo dpkg -i "$BODYCTRL_DEB_PATH"; then
    log "dpkg 未完成（常见为依赖未满足），执行 apt-get -f 修复..."
    sudo apt-get install -f -y || die "bodyctrl_msgs 依赖修复失败"
  fi
  log "bodyctrl_msgs 已安装。"
else
  if [[ "$SKIP_VENDOR_PACKAGES_FOR_UBUNTU_USER" == "1" ]]; then
    log "当前运行用户名为 ${XMIGCS_RUN_USER}，跳过 bodyctrl_msgs"
  else
    log "已设置 SKIP_BODYCTRL_MSGS=1，跳过 bodyctrl_msgs"
  fi
fi

log ""
log "安装完成 (Ubuntu ${UB_VER})"
log "= = = INSTALL XMIGCS SUCCESS = = ="
log "激活虚拟环境:"
log "  source ${ROOT}/.venv/bin/activate"
log "仿真或真机运行:"
log "  修改 src/xmigcs/config/dex_config.yaml 中的 sim 后运行: xmigcs"
if [[ "${SKIP_XLOG:-0}" != "1" && "$SKIP_VENDOR_PACKAGES_FOR_UBUNTU_USER" != "1" ]]; then
  _uh="$(xmigcs_install_user_home)"
  log "xlog: 新开终端自动生效；当前 shell 已临时 export，或执行: source ${_uh}/.bashrc  或  source ${_uh}/.zshrc"
fi

log "操作手册（飞书）: https://zitd5je6f7j.feishu.cn/wiki/Xawxwy8Aaipn4RkoEEJcTGtAnXe"
