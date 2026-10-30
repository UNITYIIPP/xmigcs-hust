#!/usr/bin/env python3
"""
测试脚本：监测 ankle_serial_to_parallel 和 ankle_parallel_to_serial 的运算速度和结果
"""
import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xmigcs.common.func_sp_trans_evt_analytic import AnalyticFuncSPTrans as FuncSPTrans

def test_ankle_parallel_to_serial():
    """
    测试 ankle_parallel_to_serial 函数的性能和结果
    """
    print("=" * 60)
    print("测试 ankle_parallel_to_serial 函数")
    print("=" * 60)
    
    # 初始化转换器
    fun_s2p = FuncSPTrans()
    
    # 准备测试数据 (并联关节数据)
    q_a_p = np.array([0.1, 0.2, -0.1, -0.2])  # 左右脚踝的pitch和roll
    qdot_a_p = np.array([0.01, 0.02, -0.01, -0.02])
    tor_a_p = np.array([1.0, 2.0, -1.0, -2.0])
    
    # 多次运行测试性能
    num_runs = 1000
    times = []
    
    print(f"输入数据:")
    print(f"  q_a_p: {q_a_p}")
    print(f"  qdot_a_p: {qdot_a_p}")
    print(f"  tor_a_p: {tor_a_p}")
    print()
    
    # 预热运行
    for _ in range(10):
        fun_s2p.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        fun_s2p.calcFK()
        fun_s2p.calcIK()
        
        q_a_s = np.zeros(4)
        qdot_a_s = np.zeros(4)
        tor_a_s = np.zeros(4)
        fun_s2p.get_s_state(q_a_s, qdot_a_s, tor_a_s)
    
    # 性能测试
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        fun_s2p.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        fun_s2p.calcFK()
        fun_s2p.calcIK()
        
        q_a_s = np.zeros(4)
        qdot_a_s = np.zeros(4)
        tor_a_s = np.zeros(4)
        fun_s2p.get_s_state(q_a_s, qdot_a_s, tor_a_s)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # 输出结果
    print(f"输出数据:")
    print(f"  q_a_s: {q_a_s}")
    print(f"  qdot_a_s: {qdot_a_s}")
    print(f"  tor_a_s: {tor_a_s}")
    print()
    
    # 性能统计
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    print(f"性能统计 (运行 {num_runs} 次):")
    print(f"  平均时间: {avg_time*1000:.4f} ms")
    print(f"  最短时间: {min_time*1000:.4f} ms")
    print(f"  最长时间: {max_time*1000:.4f} ms")
    print(f"  时间标准差: {std_time*1000:.4f} ms")
    print(f"  平均频率: {1.0/avg_time:.2f} Hz")
    print()

def test_ankle_serial_to_parallel():
    """
    测试 ankle_serial_to_parallel 函数的性能和结果
    """
    print("=" * 60)
    print("测试 ankle_serial_to_parallel 函数")
    print("=" * 60)
    
    # 初始化转换器
    fun_s2p = FuncSPTrans()
    
    # 准备测试数据 (串联关节数据)
    q_d_s = np.array([0.1, 0.2, -0.1, -0.2])  # 左右脚踝的pitch和roll
    qdot_d_s = np.array([0.01, 0.02, -0.01, -0.02])
    tor_d_s = np.array([1.0, 2.0, -1.0, -2.0])
    
    # 多次运行测试性能
    num_runs = 1000
    times = []
    
    print(f"输入数据:")
    print(f"  q_d_s: {q_d_s}")
    print(f"  qdot_d_s: {qdot_d_s}")
    print(f"  tor_d_s: {tor_d_s}")
    print()
    
    # 预热运行
    for _ in range(10):
        fun_s2p.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        fun_s2p.calc_joint_pos_ref()
        fun_s2p.calc_joint_tor_des()
        
        q_d_p = np.zeros(4)
        qdot_d_p = np.zeros(4)
        tor_d_p = np.zeros(4)
        fun_s2p.get_p_des(q_d_p, qdot_d_p, tor_d_p)
    
    # 性能测试
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        fun_s2p.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        fun_s2p.calc_joint_pos_ref()
        fun_s2p.calc_joint_tor_des()
        
        q_d_p = np.zeros(4)
        qdot_d_p = np.zeros(4)
        tor_d_p = np.zeros(4)
        fun_s2p.get_p_des(q_d_p, qdot_d_p, tor_d_p)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # 输出结果
    print(f"输出数据:")
    print(f"  q_d_p: {q_d_p}")
    print(f"  qdot_d_p: {qdot_d_p}")
    print(f"  tor_d_p: {tor_d_p}")
    print()
    
    # 性能统计
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    print(f"性能统计 (运行 {num_runs} 次):")
    print(f"  平均时间: {avg_time*1000:.4f} ms")
    print(f"  最短时间: {min_time*1000:.4f} ms")
    print(f"  最长时间: {max_time*1000:.4f} ms")
    print(f"  时间标准差: {std_time*1000:.4f} ms")
    print(f"  平均频率: {1.0/avg_time:.2f} Hz")
    print()

def test_round_trip_conversion():
    """
    测试往返转换的精度
    """
    print("=" * 60)
    print("测试往返转换精度")
    print("=" * 60)
    
    fun_s2p = FuncSPTrans()
    
    # 原始并联数据
    original_q_p = np.array([0.1, 0.2, -0.1, -0.2])
    original_qdot_p = np.array([0.01, 0.02, -0.01, -0.02])
    original_tor_p = np.array([1.0, 2.0, -1.0, -2.0])
    
    print("原始并联数据:")
    print(f"  q_p: {original_q_p}")
    print(f"  qdot_p: {original_qdot_p}")
    print(f"  tor_p: {original_tor_p}")
    print()
    
    # 并联 -> 串联
    fun_s2p.set_p_est(original_q_p, original_qdot_p, original_tor_p)
    fun_s2p.calcFK()
    fun_s2p.calcIK()
    
    q_s = np.zeros(4)
    qdot_s = np.zeros(4)
    tor_s = np.zeros(4)
    fun_s2p.get_s_state(q_s, qdot_s, tor_s)
    
    print("转换为串联数据:")
    print(f"  q_s: {q_s}")
    print(f"  qdot_s: {qdot_s}")
    print(f"  tor_s: {tor_s}")
    print()
    
    # 串联 -> 并联
    fun_s2p.set_s_des(q_s, qdot_s, tor_s)
    fun_s2p.calc_joint_pos_ref()
    fun_s2p.calc_joint_tor_des()
    
    recovered_q_p = np.zeros(4)
    recovered_qdot_p = np.zeros(4)
    recovered_tor_p = np.zeros(4)
    fun_s2p.get_p_des(recovered_q_p, recovered_qdot_p, recovered_tor_p)
    
    print("恢复的并联数据:")
    print(f"  q_p: {recovered_q_p}")
    print(f"  qdot_p: {recovered_qdot_p}")
    print(f"  tor_p: {recovered_tor_p}")
    print()
    
    # 计算误差
    q_error = np.abs(original_q_p - recovered_q_p)
    qdot_error = np.abs(original_qdot_p - recovered_qdot_p)
    tor_error = np.abs(original_tor_p - recovered_tor_p)
    
    print("转换误差:")
    print(f"  q_p 误差: {q_error}")
    print(f"  qdot_p 误差: {qdot_error}")
    print(f"  tor_p 误差: {tor_error}")
    print(f"  最大误差: {np.max([np.max(q_error), np.max(qdot_error), np.max(tor_error)])}")
    print()

def main():
    """
    主测试函数
    """
    print("踝关节串并联转换测试")
    print("开始测试...\n")
    
    # 测试并联到串联转换
    test_ankle_parallel_to_serial()
    
    # 测试串联到并联转换
    test_ankle_serial_to_parallel()
    
    # 测试往返转换精度
    test_round_trip_conversion()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()