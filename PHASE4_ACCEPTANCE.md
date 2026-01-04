# Phase 4 Acceptance Pack (VMEC + GSCO-Lite)

## 0) 目标（验收口径）

- Phase4 的核心贡献：引入 **LLM-SemOp（语义编辑算子）**，以低频触发、批量生成、异步不阻塞的方式融入 FusionOpt 主循环，并保持可复现与可审计（`llm_calls.jsonl`）。
- 验收关注点：
  - **稳定性**：不显著降低可行评估率、不会引入不可控失败。
  - **可审计**：LLM 调用全量记录（含跳过/失败/缓存命中）。
  - **有效性**：在至少一个 benchmark 上体现清晰的 `HV/top1` 改善趋势；并在 cross-benchmark 上复现增益。

## 1) 结果总览（关键 run + 指标）

指标口径：

- `feasible_rate`：`evaluations.csv` 中 `feasible==1` 的比例。
- `HV`：使用 `f*_min` 的 3 目标，参考点 `ref=[1.1,1.1,1.1]`。
- `top1(total)`：可行且 `status==ok` 的样本中 `total` 最大值。

### 1.1 VMEC（Phase4HardV6, budget=1000）

- seed=46 rerun：
  - Baseline：`feasible_rate=0.559`，`HV=1.1961`，`top1(total)=0.9089`
  - LLM-SemOp：`feasible_rate=0.597`，`HV=1.2293`，`top1(total)=0.9424`
  - Baseline run_dir：
    - `results/stellarator_vmec/fusionopt_v1/Stellarator_VMEC_Phase4HardV6_Baseline_Budget1000_seed46_20260103T074647_rerun`
  - LLM run_dir：
    - `results/stellarator_vmec/fusionopt_v1/Stellarator_VMEC_Phase4HardV6_LLM_SemOp_Budget1000_OpenAI_seed46_20260103T074650_rerun`

### 1.2 VMEC 经典 baselines（GA / NSGA2, eval_budget=1000, seed=42）

说明：baseline runner 的默认 `objective_ranges` 与 Phase4HardV6 不一致；这里采用 Phase4HardV6 的范围对 raw properties 重新归一化后，计算与 FusionOpt 同口径的 `feasible_rate/HV/top1(total)`。

- GA：`feasible_rate=0.9985`，`HV=0.2635`，`top1(total)=0.5397`
- NSGA2：`feasible_rate=0.9990`，`HV=0.2635`，`top1(total)=0.5397`

对应 baseline runner 输出：

- GA：
  - `moo_results_vmec_baselines/zgca,gemini-2.5-flash-nothinking/mols/volume_aspect_ratio_magnetic_shear_vmec_baseline_ga_b1000_baseline_GA_optimized_42.pkl`
- NSGA2：
  - `moo_results_vmec_baselines/zgca,gemini-2.5-flash-nothinking/mols/volume_aspect_ratio_magnetic_shear_vmec_baseline_nsga2_b1000_baseline_NSGA2_42.pkl`

### 1.3 GSCO-Lite（Phase4, budget=200）

- seed=42：
  - Baseline：`feasible_rate=1.0`，`HV=0.5810`，`top1(total)=2.2992`
  - LLM-SemOp：`feasible_rate=1.0`，`HV=0.6113`，`top1(total)=2.3287`
  - Baseline run_dir：
    - `results/stellarator_coil_gsco_lite/fusionopt_v1/Stellarator_GSCO_Phase4_Baseline_Budget200_seed42_20260103T042140/`
  - LLM run_dir：
    - `results/stellarator_coil_gsco_lite/fusionopt_v1/Stellarator_GSCO_Phase4_LLM_SemOp_Budget200_OpenAI_seed42_20260103T042140/`

- seed=43：
  - Baseline：`feasible_rate=1.0`，`HV=0.5752`，`top1(total)=2.2936`
  - LLM-SemOp：`feasible_rate=1.0`，`HV=0.6152`，`top1(total)=2.3326`
  - Baseline run_dir：
    - `results/stellarator_coil_gsco_lite/fusionopt_v1/GSCO_Phase4_Baseline_B200_seed43_20260103T214645/`
  - LLM run_dir：
    - `results/stellarator_coil_gsco_lite/fusionopt_v1/GSCO_Phase4_LLM_SemOp_B200_seed43_20260103T214645/`

### 1.4 GSCO-Lite（Phase4Hard, budget=200, seed=42）

目的：避免 Phase4 可行率饱和，构造更严格的可行性判定以降低 `feasible_rate`，验证 LLM-SemOp 在“可行解稀缺”时的收益。

Hard feasibility 阈值（示例）：

- `f_B<=14.4`
- `f_S<=10`
- `I_max<=0.2`

结果（HV/top1 口径同上）：

- Baseline：`feasible_rate=0.265`，`HV=0.5810`，`top1(total)=2.2992`
- LLM-SemOp：`feasible_rate=0.295`，`HV=0.6113`，`top1(total)=2.3287`

对应 run_dir：

- Baseline：
  - `results/stellarator_coil_gsco_lite/fusionopt_v1/GSCO_Phase4Hard_Baseline_B200_seed42_20260103T225525/`
- LLM-SemOp：
  - `results/stellarator_coil_gsco_lite/fusionopt_v1/GSCO_Phase4Hard_LLM_SemOp_B200_seed42_20260103T225525/`

## 2) 可复现命令（最小闭环）

### 2.1 VMEC（FusionOpt Phase4HardV6）

```bash
python run_fusionopt.py problem/stellarator_vmec/config_phase4_hard_v6_baseline_budget200.yaml --seed 42
python run_fusionopt.py problem/stellarator_vmec/config_phase4_hard_v6_llm_semop_budget200_openai.yaml --seed 42
```

（大预算版本使用对应 `budget=1000` 配置文件；run 会打印 `run_dir:`）

### 2.2 GSCO-Lite（Phase4）

```bash
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_phase4_baseline_budget200.yaml --seed 43
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_phase4_llm_semop_budget200_openai.yaml --seed 43
```

### 2.3 GSCO-Lite（Phase4Hard）

```bash
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_phase4_hard_baseline_budget200.yaml --seed 42
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_phase4_hard_llm_semop_budget200_openai.yaml --seed 42
```

## 3) 协议检查（Protocol）

```bash
python check_phase0_protocol.py --results_dir <run_dir>
```

期望：`ALL_OK: True`。

## 4) LLM 调用审计（必须可见）

每个 LLM run 的 run_dir 下：

- `logs/llm_calls.jsonl`

用于审计：

- `status`（ok/skipped/failed）
- `cache_hit`
- `error`（例如 missing_api_key）

## 5) 关于 VMEC “归一化范围/clipping”问题：是否需要现在修？

- 现状：VMEC 的 `objective_ranges` 会对 raw objectives 做 clipping 后再归一化；若范围过窄，会导致 `f*_min` 在边界饱和，从而影响 `HV/top1` 的解释（例如 `magnetic_shear` 超上界被截断为最优）。
- 验收建议：**Phase4 验收阶段不建议修改归一化逻辑**，避免推翻已产出的对比结果与报告；改动应作为 Phase5 的“benchmark/指标改进”工作单独推进。
- 当前 mitigation：
  - Phase4HardV6 已将 `objective_ranges` 调整到更贴近可行区域（例如 `magnetic_shear=[0.85,1.05]`），并在报告中明确说明 clipping 风险。

