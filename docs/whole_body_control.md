# WHOLE_BODY_CONTROL：SuperMimic 算法实验状态

键盘 `u`；独立于 `i / TELEOP_YIBO` 和 `v / SUPERMIMIC`。
从 SWR 或 AMPMLP 按 `u` 进入，按 `m` / `n` 返回，按 `c` 停止。
原有全局 DAMPING 转换继续有效。

## 模型与控制

配置入口：`src/xmigcs/policy/whole_body_control/config/whole_body_control.yaml`。
默认使用本目录独立复制的 `model/policydexevt.onnx`，旧模型和动作文件均保留。
旧 `policy47500_50hz.onnx` 为 770 维单输入，不能在新的双输入状态中运行；
原有算法仍在 i 状态中。

- 输入：`observation_history` 或 `obs_history` `[1,H,99]`，`future_goal` `[1,73*K]`。
- 输出：`action [1,29]`。H/K 从模型读取，当前默认模型均为 10。
- 每帧观测：关节位置减默认角 29、关节速度 29、pelvis 局部角速度 3、
  局部重力方向 3、上一原始 action 29、当前参考相对机器人姿态 6。
- 历史从旧到新；首次重复当前观测填满。只在策略推理时更新历史。
- 未来目标为 t+1 到 t+K；每帧含相对位移 3、相对航向姿态 6、
  该未来帧局部根部线速度 3/角速度 3、关节位置 29、关节速度乘 0.1 后 29。
- 两处旋转 6D 均为 `R00,R01,R10,R11,R20,R21`。
- 默认角、action scale、PD 和关节顺序从 ONNX `supermimic_interface` 的
  `robot_config` 等字段读取，要求完整模型快照，不读取旁置 JSON。
- 目标为 `q_default + action_scale * action`，沿用现有映射和同步发送。
  可选 `tau_limit_enabled` 保留，YAML 限值为电机顺序；省略限值时读取模型。
- 外层 100 Hz，`physical_dt: 0.01` × `decimation: 2` = 20 ms，策略 50 Hz。
  这里的 `physical_dt` 不修改外部模拟器的物理积分步长。
- `imu_frame: pelvis` 沿用 RobotData 的坐标约定；原始传感器坐标数据可配置
  `imu_waist_link`，通过 URDF 安装旋转转换姿态和角速度。

进入后前两个有效策略周期用于航向对齐，不推理。NPZ 沿用旧行为，消耗前两帧；
GMR 缓存独立按参考时间推进。不会像独立 sim2sim 脚本一样直接重置机器人位置/速度。
头部两个关节继续由框架管理；非推理周期沿用原状态行为，仅设置 PD，不重写目标。

## NPZ

`ref_source: npz` 是默认配置，`motion_file` 保留原动作。相对路径基于本状态 `model/`。

支持旧 IsaacLab 字段 `joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w`，
以及 compact `root_pos/rest`、`qpos/qvel`、prepared `root_p/root_quat/q/root_v/root_w/qd`。
参考必须为 50 Hz，四元数为 wxyz，根部线/角速度为世界坐标系；`qpos/qvel` 文件也按
sim2sim 的这个参考约定解释，不能直接假定原始 MuJoCo qvel 的局部角速度符合该约定。
多片段文件用 `motion_clip` 选择。旧文件根部由 `npz_anchor_body_index` 指定。

有 `joint_names` 时按文件元数据映射。无名字时用 YAML `npz_joint_names`：默认沿用旧
770 维策略的左右交错顺序；若数据导出器顺序不同，应修改该配置。映射后才进入新模型。

直接取当前帧及未来 K 帧，无 GMR 缓存延迟。`hold_final_reference: true` 时
`motion_length` 可作为截断边界；false 时保持旧行为，以文件结尾为边界。
末尾重复最后姿态和关节位置，只将超出有效范围的补帧速度清零，实际末帧速度保留。

## GMR：逐步补齐的延迟窗口

配置 `ref_source: gmr`。复用既有接收子进程、共享内存、232 浮点协议和 12 ms 插值；
不修改原有参考数组。新增 `RobotData.ref_sample_time`（插值样本代表的单调时刻）及
`ref_received_time`（最近接收时刻），缓存不以读取次数判断新帧。

关节顺序由 `gmr_joint_names` 指定，默认同旧 GMR 约定。使用 pelvis 的世界位置、wxyz
姿态、世界线/角速度和关节位置/速度。状态在外层每周期采集参考，按 20 ms 时间网格
插值重采样，再保存当前帧和 K 帧未来参考。输入接收链路保留最新帧，因此丢失的上游
原始帧不能被保证完整恢复；窗口是已接收样本的重采样结果。

以 K=10 为例：

| 已收到 | 当前参考 | 未来窗口 |
|---|---|---|
| F0 | F0 | F0 重复 10 次 |
| F1 | F0 | F1 重复 10 次 |
| F2 | F0 | F1、F2，剩余重复 F2 |
| F10 | F0 | F1…F10 |
| F11 | F1 | F2…F11 |

首次参考到达即可进入航向对齐和推理流程，不额外等待窗口填满。缺失尾部复制最新
采样参考的完整信息（包括速度），属于启动近似，不使用 NPZ 末尾的静止补帧规则。
延迟从 0 增加到 `K*20 ms`，完整窗口后约 200 ms，另计接收和插值延迟。
机器人观测始终实时；姿态误差和未来目标使用同一窗口的当前参考。

无第一帧时保持已有目标。断流后不推进窗口，仍用实时机器人状态跟踪冻结的窗口。
样本时间间隔超过 `gmr_gap_reset_s`（默认 0.1 s）时，恢复的第一帧直接入队并重新设置
采样时钟，不跨断流时段生成大量虚构帧；原窗口逐步被新参考替换，因此恢复过渡阶段
窗口并非跨断流时段的连续真实时间轨迹。重复或倒序时间戳忽略。
退出清空历史和缓存；重入时可用接口已经持有的最后参考重新填充起始窗口。

## 验证

```bash
PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -m unittest discover -s test -p test_whole_body_control.py -v
PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -m unittest discover -s test -p test_teleop_yibo.py -v
```

U 测试使用真实 ONNX，覆盖观测/历史、参数和电机映射、NPZ 末尾、GMR 填充/重采样/
断流/重入及公共时间字段。外部 `sim2sim.py` 存在时直接对比其 NumPy 观测和目标编码，
以及同输入的模型输出；可通过 `XMIGCS_SIM2SIM` 指定路径。此对比不运行 MuJoCo 物理。
I 测试继续覆盖原算法对照、状态切换和发送交接；ROS 接收进程测试需显式启用。
