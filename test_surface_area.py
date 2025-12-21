#!/usr/bin/env python3
"""测试表面积计算"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simsopt.geo import SurfaceRZFourier

wout_file = "/home/dataset-assist-0/MOLLM-main/problem/stellarator_vmec/vmecpp/calculations/wout_w7x.nc"
plas_n = 32

print("测试 surf_plas 的shape和方法...")
surf_plas = SurfaceRZFourier.from_wout(
    wout_file, s=1.0, nphi=plas_n, ntheta=plas_n, range='half period'
)

print(f"nfp: {surf_plas.nfp}")
print(f"stellsym: {surf_plas.stellsym}")
print(f"quadpoints_phi shape: {surf_plas.quadpoints_phi.shape}")
print(f"quadpoints_theta shape: {surf_plas.quadpoints_theta.shape}")

gamma = surf_plas.gamma()
print(f"\ngamma() shape: {gamma.shape}")

normal = surf_plas.normal()
print(f"normal() shape: {normal.shape}")

unitnormal = surf_plas.unitnormal()
print(f"unitnormal() shape: {unitnormal.shape}")

try:
    darea = surf_plas.darea()
    print(f"darea() shape: {darea.shape}")
except Exception as e:
    print(f"darea() error: {e}")

# 尝试不同的计算方式
print("\n计算表面积元：")
print("方法1: |normal|")
dS1 = np.sqrt(np.sum(normal**2, axis=2)).flatten()
area1 = np.sum(dS1)
print(f"  总表面积: {area1:.2f} m²")
print(f"  dS shape: {dS1.shape}")

# 方法2: 使用simsopt的积分
print("\n方法2: 使用simsopt内置方法")
try:
    # 计算表面积：积分1在整个表面上
    from simsopt.geo import SurfaceRZFourier
    test_func = np.ones_like(gamma[:,:,0])
    area2 = np.sum(test_func * np.sqrt(np.sum(normal**2, axis=2)))
    print(f"  表面积: {area2:.2f} m²")
except Exception as e:
    print(f"  错误: {e}")

# 方法3: 用 simsopt 的 area() 方法
print("\n方法3: 使用 surf.area() 方法")
try:
    area3 = surf_plas.area()
    print(f"  表面积: {area3:.6f} m²")
except Exception as e:
    print(f"  错误: {e}")
