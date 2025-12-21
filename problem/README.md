# Problem 目录总览（唯一保留文档）

本仓库的 `problem/` 目录包含 MOLLM 的不同优化任务（problem definitions）。

本文档作为 `problem/` 下**唯一总入口**，其余 Markdown 文档可按需删除/归档。

---

## 1. 你的当前主线（推荐顺序）

### 1.1 Part 1：VMEC 等离子体边界优化（已完成/在用）

- 任务：优化等离子体边界形状（连续参数）
- 目录：`problem/stellarator_vmec/`
- 输出：`wout_*.nc`（等离子体边界与平衡结果）

### 1.2 Part 2：线圈外壳优化（当前推进）

固定 VMEC 的 `wout_file` 后，在绕组面/wireframe 上优化外壳电流，使等离子体边界法向场误差 `B·n` 尽量小，同时追求稀疏/可制造。

这一部分天然包含两类优化：

- **组合优化**：选择哪些 cells/loops/segments 激活（拓扑/布置）
- **连续优化**：电流大小的连续调节（幅值/分配）

#### 路线 1（先做，独立 baseline 对比）

- **RCLS（连续电流 baseline）**：在固定 wireframe 上求连续电流最优解，用作参考水平。
- **组合优化（MOLLM/NSGA-II 等）**：在离散表示上搜索稀疏结构，输出 Pareto front。

> 路线1不要求 RCLS→GSCO 的 warm-start 连接，但两条路线需要共享同一套物理评估口径（`f_B`、采样、约束定义）。

#### 路线 2（路线1成功后再做）

- 将连续电流微调/投影优化嵌入到组合搜索中
- 或用 RCLS 作为组合搜索 warm-start（更工程、更接近论文 modular coil 的初始化需求）

---

## 2. 线圈问题的现有实现（建议使用顺序）

### 2.1 GSCO-Lite（最简单、最可解释）

- 目录：`problem/stellarator_coil_gsco_lite/`
- 表示：12×12 cells，状态 {-1,0,+1}，单位电流固定
- 优点：最直观、易验证组合优化能力

### 2.2 Hybrid（更工程、更灵活）

- 目录：`problem/stellarator_coil_hybrid/`
- 表示：约 150 个有物理意义的 loops + 连续电流范围
- 优点：更接近真实线圈设计（可作为路线1的更强 baseline，也便于路线2扩展）

### 2.3 Original（不推荐）

- 目录：`problem/stellarator_coil/`
- 历史上存在搜索空间过大/实现问题等，不建议作为当前主线

---

## 3. 运行提示（最小化版本）

### 3.1 校准（重要）

对于线圈类任务，建议先运行各自目录下的 `calibrate_objectives.py`，用来确定 `objective_ranges`，否则多目标优化的尺度会失真。

### 3.2 小规模测试

- 将 `pop_size`、`eval_budget` 设小（例如 10 / 50）
- 确保无报错、`f_B` 有下降趋势

---

## 4. 文档清理建议（你要求“只保留一个总的”）

我建议：

- 保留：本文件 `problem/README.md`
- 删除候选（你确认后我再给出删除命令）：
  - `problem/COIL_PART2_ROADMAP.md`
  - `problem/COMPARISON_GUIDE.md`
  - `problem/QUICK_START.md`
  - `problem/stellarator_coil_gsco_lite/README.md`
  - `problem/stellarator_coil_hybrid/README.md`

不建议删除（属于 VMEC++ 子项目/第三方文档，后续排错可能用到）：
- `problem/stellarator_vmec/vmecpp/AGENTS.md`
- `problem/stellarator_vmec/vmecpp/README.md`
- `problem/stellarator_vmec/vmecpp/VMECPP_NAMING_GUIDE.md`
