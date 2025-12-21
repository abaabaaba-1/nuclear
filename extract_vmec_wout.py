#!/usr/bin/env python3
"""
从 VMEC 优化结果中提取特定解的 wout 文件
"""
import pickle
import shutil
import sys

pkl_file = 'moo_results/zgca,gemini-2.5-flash-nothinking/mols/volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_121.pkl'

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

final_pops = data.get('final_pops', [])
if not final_pops:
    print("No final population found!")
    sys.exit(1)

pareto_front = list(final_pops[0])
print(f"Found {len(pareto_front)} solutions in Pareto front")

# 按 volume 排序（或其他指标）
sorted_by_volume = sorted(pareto_front, key=lambda x: x.property.get('volume', 0), reverse=True)

print("\nTop 5 solutions by volume:")
for i, item in enumerate(sorted_by_volume[:5]):
    print(f"\n{i+1}. Properties:")
    for key, val in item.property.items():
        print(f"   {key}: {val:.6f}" if isinstance(val, float) else f"   {key}: {val}")

# 选择最优解
best_item = sorted_by_volume[0]
print(f"\nBest solution properties: {best_item.property}")
print(f"Configuration: {best_item.value}")

# 注意：VMEC 的 value 是配置参数，不是直接的 wout 文件路径
# wout 文件应该在 vmecpp/calculations/ 目录下
# 每次 VMEC 运行都会生成对应的 wout 文件

print("\nNote: The wout file for this configuration should be in:")
print("  problem/stellarator_vmec/vmecpp/calculations/")
print("\nTo use it for coil optimization, update config.yaml:")
print("  wout_file: 'problem/stellarator_vmec/vmecpp/calculations/wout_XXX.nc'")
