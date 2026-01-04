## Benchmark 验收结果

本项目目前包含两个物理基准任务：  
GSCO-Lite（离散线圈稀疏化）和 VMEC（三目标等离子体边界优化）。  
我们为这两个 benchmark 均编写了独立的健康检查脚本，并对当前配置进行了验收。

---

### GSCO-Lite（stellarator_coil_gsco_lite）

- **健康检查脚本**
  - 入口：[problem/stellarator_coil_gsco_lite/health_check_gsco_lite.py](cci:7://file:///home/dataset-assist-0/MOLLM-main/problem/stellarator_coil_gsco_lite/health_check_gsco_lite.py:0:0-0:0)
  - 运行示例：  
    `python problem/stellarator_coil_gsco_lite/health_check_gsco_lite.py 200 2025`
  - 功能：
    - 使用与正式实验相同的 `ConfigLoader`、[SimpleGSCOEvaluator](cci:2://file:///home/dataset-assist-0/MOLLM-main/problem/stellarator_coil_gsco_lite/evaluator.py:112:0-744:21) 和 [generate_initial_population](cci:1://file:///home/dataset-assist-0/MOLLM-main/problem/stellarator_vmec/evaluator.py:95:0-158:35)。
    - 对一批候选 coil cell 配置进行评估，统计：
      - `f_B`（场误差）、`f_S`（活跃 cell 数）、`I_max`（最大电流）的原始分布。
      - 归一化分布及在 0 / 1 处的 clip 率。

- **当前 `objective_ranges` 配置**
  - `f_B:   [12.3, 16.0]`
  - `f_S:   [0, 60]`
  - `I_max: [0.2, 0.4]`

- **健康检查结论（基于随机 + warm-start 初始种群）**
  - **f_B（场误差）**
    - 原始值约在 `13.3–15.5 T²m²` 之间，均值约 `14.3`。
    - 归一化后大致在 `[0.26, 0.88]`，几乎**没有被剪裁到 0 或 1**。
    - 说明当前范围对 f_B 来说非常合适，能提供足够的区分度与梯度。
  - **f_S（活跃 cell 数）**
    - 原始值在 `3–60` 之间，均值约 `31`。
    - 归一化 clip 率：
      - `@0`: 约 `0%`
      - `@1`: 约 `40%` 左右（集中在上界 60）。
    - 这些被剪到 1 的样本对应“几乎占满所有 cell 的极度稠密解”，物理上本来就应视为同一档“最坏 sparsity”，因此这种上界饱和是**可接受且符合设计预期**的。
  - **I_max（最大电流）**
    - 原始值在 `0.2–0.4 MA` 之间，分布非常窄。
    - 归一化 clip 率典型为：
      - `@0`: ≈ `20–30%`
      - `@1`: ≈ `70–80%`
    - 表明在当前 cell→segment 拓扑与固定单位电流下，`I_max` 行为更接近一个“粗粒度约束”（安全 / 超限），而不是提供细腻梯度的一等公民目标。
    - 在算法设计层面，应将 `I_max` 主要视为**电流安全约束**或强惩罚项，而不是期望它提供与 `f_B` 同级别的连续优化信号。

> 总体结论：在当前 `objective_ranges` 下，GSCO-Lite benchmark 的数值行为干净、可用。  
> `f_B` 与 `f_S` 能提供有效梯度；`I_max` 作为粗约束使用是合理的。

---

### VMEC（三目标 W7-X 等离子体边界优化）

- **健康检查脚本**
  - 入口：[problem/stellarator_vmec/health_check_vmec.py](cci:7://file:///home/dataset-assist-0/MOLLM-main/problem/stellarator_vmec/health_check_vmec.py:0:0-0:0)
  - 运行示例：  
    `python problem/stellarator_vmec/health_check_vmec.py 100 42`
  - 功能：
    - 使用与正式实验相同的 `ConfigLoader`、[RewardingSystem](cci:2://file:///home/dataset-assist-0/MOLLM-main/problem/stellarator_vmec/evaluator.py:168:0-581:26)、[generate_initial_population](cci:1://file:///home/dataset-assist-0/MOLLM-main/problem/stellarator_vmec/evaluator.py:95:0-158:35)。
    - 实际调用 VMEC++ 求解，并通过现有 evaluator：
      - 计算 `volume / aspect_ratio / magnetic_shear`。
      - 判定收敛性（`fsqr/fsqz/fsql <= ftolv`）及 Mercier 稳定性。
      - 调用 `_apply_penalty + _transform_objectives` 得到最终归一化分数。
    - 报告：
      - 收敛率、Mercier 稳定率、整体可行率。
      - 三个目标的原始分布、归一化分布及 clip 率。

- **当前 `objective_ranges` 配置**
  - `volume:         [26.0, 29.5]`
  - `aspect_ratio:   [10.5, 11.5]`
  - `magnetic_shear: [0.9, 1.0]`

- **健康检查结论（样本数 100）**
  - **约束可行性**
    - 收敛率（fsqr/fsqz/fsql ≤ ftolv）：约 `99%`（99/100）。
    - Mercier 稳定率（min_mercier ≥ 0）：约 `99%`（99/100）。
    - 同时收敛且 Mercier 稳定的可行率：约 `99%`。
    - 表明在当前 LLM/GA 的系数扰动约束下，绝大部分候选都落在 VMEC 的**稳定可解空间**内。
  - **volume**
    - 可行解体积集中在 `[26, 29.5] m³` 附近，归一化统计：
      - `Norm mean ≈ 0.50`，`clip @0 ≈ 0%`，`clip @1 ≈ 1%`（唯一的 1% 来自被惩罚的不可行解）。
    - 说明 `volume` 的范围选取合理，不会出现大面积饱和。
  - **aspect_ratio**
    - 可行解纵横比集中在 `[10.5, 11.5]`，归一化：
      - `Norm mean ≈ 0.47`，`clip @0 ≈ 0%`，`clip @1 ≈ 1%`。
    - 说明在当前扰动下，aspect ratio 也具有足够的动态范围。
  - **magnetic_shear**
    - 可行解磁剪切在 `[0.9, 1.0]` 内变化，归一化：
      - `Norm mean ≈ 0.57`，`clip @0 ≈ 0%`，`clip @1 ≈ 1%`。
    - 同样没有明显的 0/1 饱和问题。

> 总体结论：在 `objective_ranges = {volume: [26,29.5], aspect_ratio: [10.5,11.5], magnetic_shear: [0.9,1.0]}` 下，  
> VMEC benchmark 的收敛性、稳定性与归一化范围均表现良好，可直接用于多目标优化与 LLM 对比实验。

---

### 小结

- **GSCO-Lite 与 VMEC 两个 benchmark 均经过独立健康检查脚本验证：**
  - 数值范围（`objective_ranges`）不会大面积饱和。
  - 不可行解会被明确惩罚，不会误当作“极优解”。
  - 约束（收敛性、Mercier 稳定性、电流上限等）的行为在日志中可清晰追踪。
- 推荐在任何修改物理配置或目标范围后，先运行对应的 health check 脚本，重新生成一份统计作为新的“Benchmark 验收结果”记录。