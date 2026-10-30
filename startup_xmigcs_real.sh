#!/usr/bin/env bash
# =============================================================================
# 一键启动 xMIGCS 真机运行环境（tmux 三窗格版）
#
#   Pane 1 (左上) : body_control   对应原「终端1」 sudo su + ros2 launch body_control
#   Pane 2 (右上) : joystick      对应原「终端2」 sudo su + ros2 launch joystick
#   Pane 3 (下)   : xmigcs        对应原「终端3」 source setup.bash + xmigcs 主控制节点
#
# 用法:
#   bash startup_xmigcs_real.sh          # 默认密码 123
#   SUDO_PASS=你的密码 bash startup_xmigcs_real.sh   # 指定 sudo 密码
#
# 快捷键: Ctrl-b d 分离; 在窗格内 Ctrl+C 停止对应节点
# =============================================================================

set -euo pipefail

SESSION_NAME="xmigcs_real"
WINDOW_NAME="xMIGCS"
TARGET_WINDOW="${SESSION_NAME}:${WINDOW_NAME}"

# 真机 sudo 密码（真机默认 123，可用环境变量 SUDO_PASS 覆盖）
SUDO_PASS="${SUDO_PASS:-123}"

# 真机路径
ROS_SETUP="/home/ubuntu/xos/setup.bash"   # xos 环境（含 ROS humble + bodyctrl 等）
XMIGCS_DIR="/home/ubuntu/xmigcs-hust"          # xmigcs 安装目录（含 .venv）

if ! command -v tmux >/dev/null 2>&1; then
    echo "错误: 系统未安装 tmux，请先执行: sudo apt install tmux"
    exit 1
fi

# 重建会话（避免重复启动残留旧进程）
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

echo "正在启动 xMIGCS 真机运行环境 (tmux 会话: ${SESSION_NAME}) ..."
echo "  - Pane 1: body_control"
echo "  - Pane 2: joystick"
echo "  - Pane 3: xmigcs"

# ----------------------------------------------------------------------------
# Pane 1: body_control（原终端1: sudo su -> source -> ros2 launch body_control）
# ----------------------------------------------------------------------------
tmux new-session -d -s "${SESSION_NAME}" -n "${WINDOW_NAME}" bash -lc "
  export HOME=/home/ubuntu   # 保证 sudo 环境下 HOME 路径正确
  echo '${SUDO_PASS}' | sudo -S -E bash -c '
    source ${ROS_SETUP}
    ros2 launch body_control body_control.launch.py
  '
  exec bash
"

# ----------------------------------------------------------------------------
# Pane 2: joystick（原终端2: sudo su -> source -> ros2 launch joystick）
# ----------------------------------------------------------------------------
tmux split-window -h -t "${TARGET_WINDOW}" bash -lc "
  export HOME=/home/ubuntu
  echo '${SUDO_PASS}' | sudo -S -E bash -c '
    source ${ROS_SETUP}
    ros2 launch joystick joystick.launch.py
  '
  exec bash
"

# ----------------------------------------------------------------------------
# Pane 3: xmigcs 主控制节点（原终端3: source -> cd xmigcs -> .venv -> xmigcs）
# ----------------------------------------------------------------------------
tmux split-window -v -t "${TARGET_WINDOW}" bash -lc "
  source ${ROS_SETUP}
  cd ${XMIGCS_DIR}
  source .venv/bin/activate
  xmigcs
  exec bash
"

# 平铺布局并附加
tmux select-layout -t "${TARGET_WINDOW}" tiled
tmux attach-session -t "${SESSION_NAME}"
