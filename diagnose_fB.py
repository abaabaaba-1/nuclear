#!/usr/bin/env python3
"""
诊断 f_B 计算：为什么我们的值比论文大 10^5 倍？
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.MOLLM import ConfigLoader
from problem.stellarator_coil.evaluator import RewardingSystem
import json

print("=" * 60)
print("f_B 物理量级诊断")
print("=" * 60)

config = ConfigLoader("stellarator_coil/config_nsga2.yaml")
reward_system = RewardingSystem(config)

print(f"\n等离子体表面参数:")
print(f"  nfp (对称性): {reward_system.surf_plas.nfp}")
print(f"  测试点数 (phi): {reward_system.surf_plas.quadpoints_phi.size}")
print(f"  测试点数 (theta): {reward_system.surf_plas.quadpoints_theta.size}")

# 计算表面积
points = reward_system.surf_plas.gamma()
normal_vec = reward_system.surf_plas.normal()
dS = np.sqrt(np.sum(normal_vec**2, axis=2)).flatten()
total_area = np.sum(dS)

print(f"\n表面积分:")
print(f"  总表面积: {total_area:.2f} m²")
print(f"  单个测试点平均面积: {total_area/dS.size:.4f} m²")
print(f"  测试点总数: {dS.size}")

# 检查背景场
if reward_system.B_ext_n is not None:
    B_ext_rms = np.sqrt(np.mean(reward_system.B_ext_n**2))
    B_ext_max = np.max(np.abs(reward_system.B_ext_n))
    print(f"\n背景TF场 (B_ext):")
    print(f"  RMS: {B_ext_rms:.4f} T")
    print(f"  最大值: {B_ext_max:.4f} T")
    
    # 估算背景场贡献
    f_B_background = 0.5 * np.sum(reward_system.B_ext_n**2 * dS)
    print(f"  背景场单独的f_B: {f_B_background:.2f} T² m²")
else:
    print(f"\n背景场: 未创建")

# 测试一个简单配置
print(f"\n测试计算:")
test_config = {"loops": [{"loop_id": 100, "current": 0.5}]}
current_array = reward_system._loops_to_currents([test_config["loops"][0]])

print(f"  测试配置: 1个闭环，0.5 MA")
print(f"  非零电流的线段数: {np.count_nonzero(current_array)}")
print(f"  最大线段电流: {np.max(np.abs(current_array))/1e6:.3f} MA")

f_B_test = reward_system._evaluate_field_error(current_array)
print(f"  计算得到的 f_B: {f_B_test:.2f} T² m²")

# 与论文对比
print(f"\n" + "=" * 60)
print("与 Hammond 2025 论文对比:")
print("=" * 60)
print(f"论文 f_B 值范围:")
print(f"  初始 (6个平面线圈): 0.12 T² m²")
print(f"  优化后 (λ=10⁻⁶):    3.6e-05 T² m²")
print(f"  优化后 (λ=10⁻⁹):    6.8e-06 T² m²")
print(f"\n我们的 f_B 值范围 (NSGA-II结果):")
print(f"  最小: 14,737 T² m²")
print(f"  平均: 15,258 T² m²")
print(f"  最大: 16,614 T² m²")
print(f"\n量级差异: 我们的值 / 论文的值 ≈ {15000 / 0.12:.0e}")

print(f"\n可能的原因:")
print(f"  1. 网格分辨率不同: 论文用96×100，我们用48×50")
print(f"  2. 表面积不同: 论文的表面积可能不同")
print(f"  3. 背景场处理不同: 论文可能有不同的归一化")
print(f"  4. 计算公式可能有差异")

print(f"\n" + "=" * 60)
print("建议:")
print("=" * 60)
print("1. 暂时使用观测值设置归一化范围: f_B: [10000, 20000]")
print("2. 后续需要深入检查 f_B 计算公式与论文的一致性")
print("3. 可能需要咨询 simsopt 文档或论文作者")
