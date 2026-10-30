# TELEOP_YIBO：源部署框架迁移

本状态来自 `/home/mig/universal_control/xmigcs-deploy/src/xmigcs/policy/teleop_yibo`。
策略文件和 `common/gmr_shared_receiver.py` 按源版本迁入。`SUPERMIMIC` 的目录、
模型、配置和策略计算保留，`v` 与 `i` 对应两个独立状态。

## 使用

编辑 `src/xmigcs/policy/teleop_yibo/config/teleop_yibo.yaml`：

- `onnx_path: GMT_0_0523.onnx`：默认模型，输入 `[1,770]`，输出 `[1,29]`。
- `ref_source: gmr`：默认实时参考；设置 `npz` 使用回放。
- `motion_file: ../motion_50hz/walk_forward_hips_amplified_002__A001.npz`：
  和源配置相同的轨迹，已改为项目内相对路径。
- `physical_dt: 0.02`、`decimation: 1`：策略周期20 ms。
- `tau_limit_enabled: false`：保持源默认设置。

`onnx_path` 和 `motion_file` 的相对路径都基于该状态的 `model/` 目录。
源目录中的模型及NPZ均已迁入，包含 `model/` 的子目录及 `motion_50hz/`。

先用现有流程进入 `AMPMLP` 或 `SWR`，再按 `i` 进入 `TELEOP_YIBO`。
可按 `n` 返回 `AMPMLP`、`m` 返回 `SWR`，`c` 进入STOP。
与源配置一致，不提供STANDZERO或SUPERMIMIC直接进入I的转换。
`v` 的原有入口仍然有效。

## 策略计算

770维按观测项分别堆叠五帧，历史由旧到新，首个有效观测重复填充：

```text
[参考关节位置/速度58×5 | 相对姿态6×5 | 实测角速度3×5
 | 实测关节位置减默认角度29×5 | 实测关节速度29×5 | 上次动作29×5]
```

前两个有效策略tick只更新航向对齐，随后开始推理。
关节顺序、默认角度、Kp、Kd和action scale读取ONNX的CSV元数据。
目标为 `default_joint_pos + action_scale * action`，映射到前29个身体关节。
状态本身不写两个头部关节。它不使用SuperMimic的未来目标或JSON侧文件。

## GMR数据链路

```text
ROS2接收子进程 → 共享内存最新帧 → RobotInterface.get_realtime_gmr_data()
             → 最新两帧插值 → RobotData.ref_* → TELEOP_YIBO
```

`dex_config.yaml` 的 `robot_interface` 中沿用源配置：

```yaml
gmr_interp_delay_sec: 0.012
gmr_shared_receiver:
  enabled: true
  gmr_topic: /gmr_info
  gmr_6kp_topic: /gmr_6kp
  cpu_ids: [3]
  fifo_prio: 40
  startup_timeout_s: 5.0
```

接收子进程使用BEST_EFFORT、KEEP_LAST、depth=1。
保留源版本的CPU/实时调度配置和启动失败后的本进程ROS订阅路径。
备用路径订阅 `/gmr_info`，队列容量32，控制循环取最新帧。
`/gmr_6kp` 是源接收器保留的槽位，TELEOP_YIBO不消费它。

`/gmr_info` 为 `Float32MultiArray`，长度232：

| 范围 | 内容 |
|---|---|
| 0:72 | 24个body的世界位置 |
| 72:168 | 24个body的世界四元数，wxyz |
| 168:197 | 29个关节位置，模型顺序，rad |
| 197:226 | 29个关节速度，同顺序，rad/s |
| 226:229 | 根部世界线速度 |
| 229:232 | 根部世界角速度 |

参考关节数组必须已经按模型 `joint_names` 排列，源接收器不做名称重排。
插值保留源线性位置/速度插值及四元数线性混合后归一化。

## 调度和行为边界

当前框架已有外层循环。I无 `dt_`，使用默认10 ms调用周期；状态内部根据
`RobotData.time_now_` 每20 ms执行一次参考推进及推理。I的等待时序采用源版
`loop_start + period`，发送在控制线程同步执行，之后更新机器人反馈。
后台发送线程跳过I的命令帧；退出I后由其他状态的新帧恢复后台发送，发布锁保证
发送权限交接。其他状态继续按自己的 `dt_` 及现有发送线程运行。

本次保持源策略的实际行为：

- 尚未收到参考时不推理、不改目标，但仍设置PD。
- 非推理tick只设置PD，不额外恢复或重写目标。
- 缓存参考停更后继续推理跟踪最后参考，没有自动断流STOP。
- NPZ按策略计数推进；前两个航向对齐tick也消耗帧。
- NPZ到末尾后钳制到最后一帧；`hold_final_reference` 还可按 `motion_length`
  提前钳制。不会自动请求STOP。
- 重入清空历史和计数，再进行航向对齐。

旧SuperMimic遥操作的未来帧缓存、自动超时退出及专用FSM退出请求均已移除。
上述源行为没有在此次迁移中另行改进。

## 验证

在项目依赖环境（包含ROS2 Python路径）运行，不启动电机接口：

```bash
PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -m unittest discover -s test -p test_teleop_yibo.py -v
```

如源部署目录仍可用，测试会额外比较源版和迁移版的逐帧观测、动作和关节命令；
可通过 `XMIGCS_SOURCE_DEPLOY` 指定源目录。
独立域的ROS接收子进程测试只发布参考消息：

```bash
ROS_DOMAIN_ID=231 ROS_LOCALHOST_ONLY=1 XMIGCS_ROS_TEST=1 \
  PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -m unittest discover -s test -p test_teleop_yibo.py -v
```

旧AMP测试会替换ONNX/ROS模块，需与本测试分别启动进程。
