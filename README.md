# xMIGCS

xMIGCS (X-humanoid Motion Intelligence Group Control System) 是一个人形机器人运动控制系统，基于有限状态机（FSM）和 RL 策略模块实现对机器人的灵活控制。

## 功能特性

- **状态机管理**: 基于 FSM 模块实现机器人行为的状态流转控制
- **RL 策略推理**: 支持 ONNX 模型实时推理，策略模块化可扩展
- **多种控制输入**: 支持云卓手柄、XBOX 手柄、键盘控制
- **安全保护**: 电机故障自动切入阻尼状态、摔倒自动停止
- **配置驱动**: 使用 YAML 配置参数，支持仿真/真机快速切换
- **模块化设计**: 各策略独立封装，便于扩展与维护
- **多通信模式**: 支持 ROS2、LPC、SHM RPC、TCP RPC

## 当前状态机



| 状态 | 功能 | 说明 |
|------|------|------|
| STOP | 停止保持 | 保持当前关节位置不动 |
| STANDZERO | 站立零位 | 线性插值到站立姿态（需机器人站立才能切入） |
| DAMPING | 阻尼保护 | 低增益锁定关节（电机故障时自动进入） |
| SWR | 全身行走 | AMP 6模式行走，手柄控速 |
| AMPMLP | AMP MLP行走 | 带历史帧的MLP行走策略，手柄控速 |
| SUPERMIMIC | 参考动作跟踪 | 播放打包的 walk1_subject1 动作 |
| TELEOP_YIBO | GMR/NPZ 跟踪 | 源部署框架的770维策略，独立接收进程 |
| WHOLE_BODY_CONTROL | 独立算法实验状态 | TELEOP_YIBO的完整副本，独立代码、配置和资源 |


## 安装与运行

### 环境要求

- Ubuntu 22.04 或 24.04
- curl、wget
- Git (用于版本控制)

> `install.sh` 会自动检测 Ubuntu 版本：
> - **Ubuntu 22.04**：创建 Python 3.10 虚拟环境（兼容 ROS humble 的 rclpy/pinocchio）
> - **Ubuntu 24.04**：创建 Python 3.12 虚拟环境

### 一键安装（仿真与真机步骤相同）

请使用 `bash ./install.sh` 而非 `./install.sh`。

- 可选环境变量

| 变量 | 说明 |
|------|------|
| `SKIP_PINOCCHIO=1` | 跳过 ros pinocchio 安装 |
| `SKIP_UV_INSTALL=1` | 跳过 uv 自动安装（需自行确保 uv 可用） |
| `SKIP_XLOG=1` | 跳过 xlog 预编译包下载 |
| `SKIP_BODYCTRL_MSGS=1` | 跳过 bodyctrl_msgs .deb 安装 |

### 手动安装（如一键安装失败）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境（22.04 用 3.10，24.04 用 3.12）
uv venv --python 3.10   # Ubuntu 22.04
# uv venv --python 3.12  # Ubuntu 24.04

uv sync --no-install-project
source .venv/bin/activate
export XMIGCS_DEV=1
uv pip install -e .

# 安装 sptlib（22.04 用 cp310，24.04 用 cp312）
# Ubuntu 22.04:
# wget http://10.10.250.160:5000/download/evt1%E8%84%9A%E8%B8%9D/sptlib_python-0.1.0-cp310-cp310-linux_x86_64.whl
uv pip install vendor_packages/sptlib_python-0.1.0-cp310-cp310-linux_x86_64.whl
# Ubuntu 24.04（真机）:
# wget http://10.10.250.160:5000/download/evt2%E8%84%9A%E8%B8%9D/sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl
uv pip install vendor_packages/sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl
```

- **注意：** 安装 xlog

`install.sh` 会安装 xlog 预编译包。如果一键安装失败，请查看 [XLOG_USAGE.md](XLOG_USAGE.md) 来完成安装。

### 代码打包(ubuntu 24环境下执行)

```bash
# 生产环境，编译wheel包(仅需要构建wheel包才运行下面的代码)
cd xmigcs
source .venv/bin/activate
unset XMIGCS_DEV
rm -r build
rm -r dist
uv build --wheel

# 安装xMIGCS wheel包
#仿真
uv pip install dist/xmigcs-*.whl
#真机
pip show xmigcs
#如果上面有输出，就手动卸载
pip uninstall xmigcs --break-system-packages
# 将运控包拷贝到Ubuntu下,运行下面的指令安装包
pip install xmigcs-xx.whl(替换为拷贝的运控包) --break-system-packages
# 确认安装成功，查看版本
pip show xmigcs
```


## 控制器使用说明



### 真机运行

**注意:** 
    - 先检查自启动是否关闭，未关闭头顶不是白灯
    ```bash
    sudo systemctl status proc_manager.service # 查看状态
     # Loaded: loaded (/etc/systemd/system/proc_manager.service; disabled; preset: enabled)
     # Active: inactive (dead)
    
    sudo systemctl disable proc_manager.service # 关闭自启动
    ```

- 一键启动
 - 注意启动前更改`startup_xmigcs_real.sh`脚本中的机器人密码和代码路径
```bash
    bash startup_xmigcs_real.sh
```


- 手动启动终端
```bash
# 真机上启动body_control（终端1）
sudo su
source /home/ubuntu/xos/setup.bash
ros2 launch body_control body_control.launch.py
# 如果需要单独启动手柄（终端2，可选）
sudo su
source /home/ubuntu/xos/setup.bash
ros2 launch joystick joystick.launch.py
# 启动主控制节点（终端3）
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
source /home/ubuntu/xos/setup.bash
# 真机上启动xMIGCS, 使用开发者方式安装的启动方式
cd xmigcs
source .venv/bin/activate
# 使用 配置文件中的通信方式
xmigcs
```


### 仿真运行

1. 修改配置文件 `src/xmigcs/config/dex_config.yaml`：

```yaml
sim: true    # true=仿真（跳过串并联踝解算），false=真机（执行串并联踝解算）
```

2. 启动仿真：

```bash
# 终端1：RL节点
cd xmigcs
# source /opt/ros/humble/setup.bash
source .venv/bin/activate
xmigcs
# 终端2：手柄节点（键盘不用）
cd xmigcs
source /opt/ros/humble/setup.bash
source .venv/bin/activate
ros2 launch joystick joystick.launch.py
# 终端3：运行mujoco-sim
```

```bash
xmigcs   # 仿真与真机统一入口
```

> **注意**: `xmigcs` 会读取 `dex_config.yaml` 中的 `sim`。仿真设为 `true`，真机设为 `false`；模式在控制线程启动前确定。

### 代码打包（真机 OTA）

```bash
source .venv/bin/activate
unset XMIGCS_DEV
rm -rf build dist
uv build --wheel
# 将 dist/xmigcs-*.whl 拷贝到真机后安装
```

## 控制器使用说明

### 云卓 T12 手柄

使用前确保所有按键和拨杆已回中。

#### 状态切换

| 操作 | 命令 | 说明 |
|------|------|------|
| d 键按下 | gotoZERO | STOP → 站立零位 |
| b 键按下（e/f/g/h 中位） | gotoSWR | 切换到 SWR 行走 |
| a 键按下（e/f/g/h 中位） | gotoAMPMLP | 切换到 AMP MLP 行走 |
| a+b 同时按下（e/f/g/h 中位） | gotoSUPERMIMIC | 从 STANDZERO 播放参考动作 |
| c  | gotoSTOP | 任意状态 → STOP |

#### 运动控制（行走态）

| 摇杆 | 功能 |
|------|------|
| 左摇杆 Y1 | 前后移动 |
| 左摇杆 X1 | 左右移动 |
| 右摇杆 X2 | 机身旋转 |
| e 上拨 | 禁用手柄控制 |

### 键盘控制

在 `dex_config.yaml` 中设置 `control_tool: keyboard` 后可用。

| 按键 | 功能 |
|------|------|
| w / s | 前进 / 后退（增量调节） |
| a / d | 左移 / 右移 |
| q / e | 左转 / 右转 |
| r | 重置速度为 0 |
| z | gotoZERO（站立零位） |
| m | gotoSWR（SWR 行走） |
| n | gotoAMPMLP |
| v | gotoSUPERMIMIC |
| i | gotoTELEOP_YIBO（源部署策略，从AMPMLP或SWR进入） |
| u | gotoWHOLE_BODY_CONTROL（独立算法实验状态，从AMPMLP或SWR进入） |
| c | gotoSTOP（紧急停止） |
| ← / → | 调节高度 |
| x / Ctrl+C | 退出 |

### XBOX 手柄

```bash
# 启动 XBOX 手柄数据节点
ros2 run joy joy_node --ros-args --remap joy:=xbox_data
```

| 按钮 | 功能 |
|------|------|
| X | gotoZERO |
| B | gotoSWR |
| A | gotoAMPMLP |
| A+B | gotoSUPERMIMIC |
| 左摇杆 Y1 | 前后移动 |
| 左摇杆 X1 | 左右移动 |
| 右摇杆 X2 | 机身旋转 |

## 安全保护配置

在 `dex_config.yaml` 中配置：

```yaml
safety:
  fall_protection: true         # 摔倒保护开关（true: 开启，false: 关闭）
  fall_gravity_threshold: -0.3  # 摔倒判定阈值（投影重力 z 分量 > 此值视为摔倒）
```

- 开启后，系统通过 IMU 投影重力实时检测机器人姿态
- 倾斜超过阈值（默认约 73°）自动切入 STOP 状态
- 电机报错时自动切入 DAMPING 状态（始终生效，不受此开关控制）

## 项目结构

```
.
├── vendor_packages/       # 离线供应商安装包（xlog、sptlib、bodyctrl_msgs）
├── src/                    # 源代码
│   └── xmigcs/             # 核心源码
│       ├── FSM/            # 状态机模块
│       │   ├── fsm_base.py         # FSM 抽象基类 (FSMState, RobotFSM)
│       │   └── robot_fsm.py        # FSM 实现 (动态加载策略、安全守护)
│       ├── common/         # 通用模块
│       │   ├── rotation.py         # 四元数/旋转数学工具
│       │   ├── joint_mapping.py    # 关节映射 (lab↔mj, 从配置加载)
│       │   ├── robot_data.py       # 机器人状态数据总线
│       │   ├── robot_interface.py  # 硬件通信编排层
│       │   ├── communication_adapter.py  # 通信适配器抽象缝
│       │   ├── motor_command_utils.py    # 电机指令转换与插值工具
│       │   ├── dynamic_model.py    # Pinocchio 动力学模型
│       │   ├── joystick.py         # 控制器输入 (ControlFlag)
│       │   └── body_id_map.py      # 电机 ID 映射
│       ├── config/         # 配置文件
│       │   ├── dex_config.yaml     # 主配置 (状态机、关节列表、通信模式等)
│       │   └── *.urdf              # 机器人 URDF 模型
│       ├── policy/         # 策略模块
│       │   ├── onnx_policy_base.py # ONNX 策略基类
│       │   ├── ampmlp/            # AMP MLP 行走策略
│       │   ├── swr/               # 全身行走策略
│       │   ├── stop/              # 停止状态
│       │   ├── standzero/         # 站立零位
│       │   └── damping/           # 阻尼保护
│       ├── utils/          # 工具模块
│       ├── rl_control_node.py      # 主控制节点（仿真与真机统一入口）
├── logs/                   # 日志与分析文件夹
├── scripts/                # 辅助脚本
├── test/                   # 测试文件 (性能对比、关节测试等)
├── deploy.bash             # 部署脚本
├── install.sh              # 一键安装脚本（支持离线本地包安装）
├── install_wheel_real.sh   # 真机 wheel 包快捷安装脚本
├── startup_xmigcs_real.sh  # 唯一真机入口；手柄选择策略
├── pyproject.toml          # 项目配置文件
├── README.md               # 项目说明文档
└── requirements.txt        # 依赖包列表
```

## SuperMimic EVT2 参考动作跟踪

`policy/supermimic/` 已恢复为 `/home/mig/ZhWT/xmigcs-hust` 的原版实现与配置，
当前使用 29 关节 `policy.onnx`、配套 JSON 和 `fight1_subject3.npz` 参考动作。
该检查点使用 `reference_relative` 目标编码。每 20 ms 输入最近 10 帧
99 维机器人观测和未来 10 帧共 730 维参考目标，输出 29 维原始动作。
策略类会按关节名称映射到机器人 31 电机顺序；头部两个电机增益为零。

先进入 `STANDZERO`，再用云卓手柄 **A+B**、Xbox **A+B** 或键盘 **v**
进入 `SUPERMIMIC`。进入后先用 2 秒插值到参考动作首帧，再播放参考动作；
停止行为保持原版，也可以随时按原有停止键。控制周期、插值时间、
动作文件和 IMU 帧约定可在
`src/xmigcs/policy/supermimic/config/supermimic.yaml` 修改。

部署端从参考 NPZ 生成未来目标，不读取真机的全局位置。实际关节角、
关节速度和 IMU 姿态/角速度仍来自机器人。默认假设 `RobotData` 提供的
IMU 数据已在 pelvis 坐标系；如果是原始 `imu_waist_link` 数据，将配置中的
`imu_frame` 改为 `imu_waist_link`。更换 ONNX 时需一起更换同名 `.onnx.json`，
并确认它记录的 `input_future_goal.encoding` 与参考目标构造一致。

代码已用打包的 ONNX 和参考动作完成离线推理及特征数值比对；真机传感器
坐标系、执行器增益和 2 秒过渡仍需上机核对。

`TELEOP_YIBO` 从 `/home/mig/universal_control/xmigcs-deploy` 迁入，独立于 `v` 的
`SUPERMIMIC`。默认模型为 `GMT_0_0523.onnx`，输入770维、输出29维，PD和动作缩放
来自ONNX。配置、模型和轨迹均在 `policy/teleop_yibo/`，支持 `gmr` / `npz`。
从 `AMPMLP` 或 `SWR` 按 `i` 进入，`c` 停止；`v` 的现有策略保留。
GMR使用源版接收子进程、共享内存和12 ms插值延迟，参考关节按模型顺序传入。
外层100 Hz、策略50 Hz，I同步发送命令；其他状态保留原周期和后台发送方式。
沿用源行为：缓存参考停更后继续跟踪最后参考，不自动超时转STOP。
配置、协议和运行步骤见 [TELEOP_YIBO 使用说明](docs/teleop_yibo.md)。

`u` 对应独立的 `WHOLE_BODY_CONTROL`，完整复制当前 `TELEOP_YIBO` 的计算逻辑和
资源，保留GMR/NPZ、PD/动作缩放、调度及同步发送特性。新算法可以在
`policy/whole_body_control/` 内开发；原 `i` 和 `v` 保留。
使用说明见 [WHOLE_BODY_CONTROL](docs/whole_body_control.md)。

## 如何添加新的控制策略(推荐使用skill：add-policy)

### 方式一：继承 ONNXPolicyBase（推荐）

适用于基于 ONNX 模型的 RL 策略。基类封装了配置加载、ONNX 推理会话、metadata 解析、lab↔mj 关节映射、观测/动作缓冲区管理。

1. 在 `policy/` 下创建新策略文件夹，例如 `dance/`
2. 创建以下文件：
   - `fsm_dance.py` — 实现 `FSMStateDANCE(ONNXPolicyBase)`
   - `config/dance.yaml` — 策略配置（参考已有策略的 YAML）
   - `model/` — 放置 ONNX 模型文件

```python
from xmigcs.policy.onnx_policy_base import ONNXPolicyBase
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
from xmigcs.common.joint_mapping import MJ_JOINTS_31
import numpy as np
import os


class FSMStateDANCE(ONNXPolicyBase):
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "dance.yaml")
        cfg = self._load_config(config_path)       # 加载 YAML 配置
        self._init_buffers()                         # 初始化缓冲区
        self._init_onnx_session(                     # 加载 ONNX 模型
            os.path.join(current_dir, "model"),
            cfg["model_path"],
        )
        self._parse_onnx_metadata()                  # 解析 ONNX metadata
        self._build_joint_mapping(MJ_JOINTS_31)      # 构建 lab↔mj 映射

    def compute_observation(self, flag: ControlFlag) -> None:
        """构建观测向量（每个策略不同）"""
        ...

    def _apply_action_to_robot(self) -> None:
        """将推理输出写入 robot_data"""
        ...

    def _apply_pd_gains(self) -> None:
        """设置 PD 增益"""
        ...

    def check_transition(self, action: str = "", target_state=None, **kwargs) -> bool:
        """守卫条件：返回 True 允许切换，False 禁止"""
        return True
```

3. 在 `dex_config.yaml` 中注册新状态：

```yaml
states:
  STOP: 0
  STANDZERO: 1
  DAMPING: 2
  SWR: 3
  AMPMLP: 4
  DANCE: 5    # 新增状态

transitions:
  gotoSTOP:
    - {from: STOP, to: STOP}   # 全局事件，只需一条即可自动补全

  gotoDANCE:
    - {from: STANDZERO, to: DANCE}
    - {from: DANCE, to: DANCE}
```

4. 在 `common/joystick.py` 中添加按键映射

### 方式二：直接继承 FSMState

适用于非 ONNX 策略（如纯逻辑控制、插值运动等）：

```python
from xmigcs.FSM.fsm_base import FSMState
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData


class FSMStateMYSTATE(FSMState):
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

    def on_enter(self):
        pass

    def run(self, flag: ControlFlag):
        pass

    def on_exit(self):
        pass

    def check_transition(self, action: str = "", target_state=None, **kwargs) -> bool:
        return True
```

### 共享工具模块

| 模块 | 用途 |
|------|------|
| `xmigcs.common.rotation` | 四元数/旋转数学（euler→quat, quat_mul, matrix_from_quat 等） |
| `xmigcs.common.joint_mapping` | 关节映射（MJ_JOINTS_31/29, JointMapping, scatter/gather） |
| `xmigcs.common.motor_command_utils` | 电机指令转换、状态间插值 |
| `xmigcs.common.communication_adapter` | 通信适配器抽象缝（扩展新通信方式） |

### 关节名配置

关节名列表统一在 `src/xmigcs/config/dex_config.yaml` 的 `mj_joints` 字段中维护。修改关节顺序时只需改 YAML，所有策略自动生效。

## 架构决策记录


## 开发与贡献

欢迎对项目进行贡献，开发前请确保：

1. 遵循项目代码规范
2. 添加适当的测试用例
3. 提交前运行所有测试确保无误

## 许可证

本项目仅供内部使用。
