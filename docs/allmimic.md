# ALLMIMIC 状态（o 键）

`ALLMIMIC` 复用 whole_body_control 的外层调度、NPZ/GMR 参考读取与 GMR 延迟窗口。从 SWR/AMPMLP 按 `o` 进入，`m`/`n` 返回，`c` 停止。策略周期为 `physical_dt * decimation = 0.02 s`（50 Hz）。

默认策略、导出信息与状态配置：

- `src/xmigcs/policy/allmimic/model/actor_0164000.onnx`
- `src/xmigcs/policy/allmimic/model/actor_0164000.json`
- `src/xmigcs/policy/allmimic/motion_50hz/Lafan_urdf/walk1_subject1.npz`
- `src/xmigcs/policy/allmimic/config/allmimic.yaml`

模型没有嵌入控制元数据。PD、默认角、action scale 和残差掩码从 ONNX 同目录的同名 JSON 读取；更换模型时应同时更换对应 JSON。本状态要求导出配置中的 `goal_pos` 和 `goal_rot` 均为 `current_ref`，heading 为 `xaxis`。旧的 robot-relative 模型不能直接用于此状态。

当前输入为 `obs_history [1,930]`（10×93）和 `goal [1,730]`（10×73），输出 `action [1,29]`。模型内含归一化，外部输入不预归一化。

93 维单帧：关节角减默认角 29，关节角速度 29，pelvis 局部角速度 3，局部重力方向 3，上次原始 action 29。10 帧由旧到新展开，首次用当前帧填充。机器人姿态和角速度来自 IMU；`imu_frame` 根据实际传感器坐标系选择。

每个目标帧：

1. 未来参考根部位置减当前参考根部位置，转到当前参考的航向坐标系：3 维，包括高度差。
2. 未来参考根部姿态相对当前参考航向的旋转 6D：6 维，按第一列三项、第二列三项展开。
3. 未来参考关节角 29、参考关节速度 29（不乘 0.1）。
4. 未来参考根部自身坐标系下的线速度 3、角速度 3。

目标使用 t+1 到 t+10。整个目标由当前和未来参考生成，不读取机器人 pelvis 的世界位置，也不使用此前的参考位置近似。它与 U 状态的 goal 在字段顺序、旋转 6D 展开方式及关节速度缩放上不同。

目标关节位置 `q0 + action_scale * policy_action_gain * action + reference_mask * (q_ref_next - q0)`，其中 `q_ref_next` 为 t+1 的参考关节角。当前模型两条腿的 reference_mask 为 0（以默认角为基准），腰和手臂为 1（以 t+1 参考关节角为基准）。控制参数取自 JSON。

当前默认 `ref_source: npz`，使用配置指定的 Lafan_urdf `walk1_subject1.npz`；也可在 YAML 中设为 `gmr`。沿用状态原来的进入流程，前两个有效参考周期不推理；NPZ 在这两拍继续前进。GMR 未来窗口按 U 的策略逐步填充，满窗后参考滞后约 200 ms，另加传输和插值延迟。NPZ 超出动作末尾的未来帧重复末帧全量信息，包含末帧速度。

```bash
PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -B -m unittest discover -s test -p test_allmimic.py -v
```

测试覆盖新模型真实推理、目标位姿编码、残差目标、NPZ/GMR 和按键切换，不启动电机接口。
