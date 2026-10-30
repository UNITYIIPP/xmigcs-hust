# WHOLE_BODY_CONTROL：独立算法实验状态

从当前 `policy/teleop_yibo/` 完整复制，状态名为 `WHOLE_BODY_CONTROL`，键盘为 `u`。
原 `i / TELEOP_YIBO` 和 `v / SUPERMIMIC` 保留。

## 文件与开发入口

```text
src/xmigcs/policy/whole_body_control/
├── fsm_whole_body_control.py       # FSMStateWHOLE_BODY_CONTROL，新算法入口
├── config/whole_body_control.yaml # 独立参数与资源选择
├── model/                        # 全部模型/动作及其原有子目录
└── motion_50hz/                   # 全部50 Hz参考动作
```

策略代码直接继承 `FSMState`，不继承或导入TELEOP_YIBO。
所有模型、动作文件均为独立文件副本，不使用指向I目录的软链接或硬链接。
两个状态的ONNX会话、观测历史、动作和计数器也独立。

复制时沿用当前用户配置：

- `onnx_path: policy47500_50hz.onnx`
- `ref_source: npz`
- `motion_file: ../motion_50hz/walk_forward_hips_amplified_002__A001.npz`
- `physical_dt: 0.01`、`decimation: 2`，策略周期20 ms。

相对路径基于本状态的 `model/` 目录。后续只需修改本目录的代码、YAML及资源。

## 切换和运行

从SWR或AMPMLP按 `u` 进入；按 `m` / `n` 返回SWR / AMPMLP，按 `c` 停止。
全局DAMPING转换同样可用。与I相同，不提供STANDZERO直接进入的转换。

初始版本保留I的全部行为：770维观测、五帧分项历史、29关节输出、前两个有效
策略tick的航向对齐、ONNX读取PD/默认角度/action scale、可选力矩限制、
NPZ末帧钳制及重入复位。头部两个关节由现有框架配置管理。
外层100 Hz，推理50 Hz；U使用与I相同的同步命令发送和外层等待逻辑。
其他状态继续使用原有调度和后台发送。

将 `ref_source` 改成 `gmr` 可使用实时参考。I和U复用框架的GMR接收子进程、
共享内存、12 ms插值及 `RobotData.ref_*`；每次仅当前状态计算控制输出。
GMR协议、缓存停更后继续跟踪旧参考等行为与I相同，见
[GMR协议和源行为说明](teleop_yibo.md)。公共GMR接口改动会同时影响I和U。

## 验证

测试不启动电机接口，覆盖与I相同的观测、动作、GMR/NPZ行为，以及独立资源和缓冲区：

```bash
PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -m unittest discover -s test -p test_whole_body_control.py -v
PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -m unittest discover -s test -p test_teleop_yibo.py -v
```

第二组还验证I/U/V键盘事件、状态注册与转换，以及I/U同步发送到其他状态后台发送
的交接。初始算法基准使用相同输入与源部署策略逐帧对照；后续修改算法时，应相应更新基准。
