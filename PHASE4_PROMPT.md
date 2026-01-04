# Phase 4 外部大模型实现任务：FusionOpt LLM-SemOp（语义算子 + 异步批量 + 缓存 + 全量日志）

你将为本仓库的 **FusionOpt** 代码路径实现 Phase4：**LLM-SemOp**（语义级编辑算子）。

目标：把 LLM 放在“语义级编辑提案”位置（低频触发、批量生成、异步不阻塞），并且在没有 API Key/网络或 LLM 输出不合法时 **自动回退** 到现有 StdOp/HeuRepairOp，保证稳定性。

## 0. 重要约束（必须遵守）

- **不要引入/调用 legacy MOLLM runner/codepath**：不要 import 或依赖 `model/MOLLM.py`、`algorithm/MOO.py` 等旧路径。
- **保持现有 Phase0/Phase3 行为不回归**：不开启 SemOp 时，结果目录结构与日志字段保持一致。
- **不要在 repo 中写入任何 api_key**：输出到 `run_dir/config.yaml` 的配置已在 `run_fusionopt.py` 做了 `api_key` 清空，但你新增的任何日志/缓存也不得包含 api_key。
- **输出必须可复现**：固定 `temperature=0.0`；缓存 key 必须确定性。

## 1. 现有入口与关键文件（你需要在这些位置集成）

- 入口：`run_fusionopt.py`
  - 负责读取 YAML config、构建 `FusionOptEngine`、以及写入 run_dir/config.yaml（已清空 api_key）。
- 主循环：`fusionopt/engine.py`（`FusionOptEngine`）
  - 当前 operator 只有：`std_crossover/std_mutation/std_resample`
  - 当前已有 Phase4 占位：`run_dir/logs/llm_calls.jsonl` 文件创建
  - Gate/HeuRepair 统一在 `_gate_and_repair()`
- 领域适配：
  - VMEC：`fusionopt/adapters_vmec.py`（gate + heu_repair + std ops）
  - GSCO-Lite：`fusionopt/adapters_gsco.py`
- 协议字段：`fusionopt/protocol.py` + `eval_logger.py`

## 2. 你要实现的功能（功能需求）

### 2.1 新增 LLM-SemOp operator

在 `FusionOptEngine` 中新增一个 operator：`llm_semop`，并能够通过 config 中的权重参与 offspring 生成（与 std operators 同一层级抽样）。

推荐做法：
- 在 `fusionopt/engine.py`：
  - 解析 `fusionopt.operator_weights.llm_semop`（默认 0.0，即不启用）
  - 在 `_sample_operator()` 返回值中加入 `llm_semop`
  - 在 offspring 生成分支中，若 op==`llm_semop`：
    - 从 population 采样 1 或 2 个 parent（建议先 1 个 parent，降低实现复杂度）
    - 调用一个新的模块（例如 `fusionopt/llm_semop.py`）去生成 child decision_json
    - child 仍需经过 `_gate_and_repair()` + dedup

### 2.2 低频触发（Trigger）

Phase4 目标是“低频触发”，因此 **不能** 每个 child 都调用 LLM。

要求：至少支持一种简单触发策略（先固定间隔即可）：
- `fixed_interval`：每 N 代（generation）最多触发一次 batch 请求，batch 中生成 K 个候选；其余时间返回 `None` 让 engine 回退 std ops。

建议 config：
- `fusionopt.llm_semop.enabled: bool`
- `fusionopt.llm_semop.trigger.type: "fixed_interval"`
- `fusionopt.llm_semop.trigger.every_n_generations: int`
- `fusionopt.llm_semop.batch_size: int`

### 2.3 异步批量（Async + Batch）

要求：LLM 生成不能阻塞主优化循环。

建议实现：
- 用 `ThreadPoolExecutor` 在后台提交一个 batch 请求任务
- engine 在需要 child 时：
  - 若后台已有 ready 的结果，就消费一个；
  - 若没有 ready，就返回 `None`（回退 std ops），**不要等待**

### 2.4 严格 schema 与校验 + 回退策略

LLM 输出必须是 JSON，并能被严格解析与校验。

最小可行 schema（推荐直接让 LLM 输出完整 decision object，而不是 diff）：

- VMEC：
  - `{"new_coefficients": {"<mode_key>": <float>, ...}}`
- GSCO-Lite：
  - `{"cells": [[phi:int, theta:int, state:int], ...]}`

校验规则：
- 必须是 JSON object
- 字段类型必须匹配（dict/list/int/float）
- 通过 adapter 的 `gate()`（否则视为无效）
- （可选）再走 `heu_repair()` 以提升可行率

回退策略（必须实现）：
- 任何解析失败 / schema 不合法 / gate 返回 None / repair 返回 None：
  - 记录一次 llm_calls 日志（见 2.6）
  - 返回 `None`，让 engine 用 std operators 继续产出 offspring

### 2.5 确定性缓存（Deterministic Cache）

要求：缓存 key 必须由以下内容确定性生成：
- `prompt` 字符串
- `model.name`（或 model id）
- `temperature`（固定 0.0）
- `schema_version`（你定义一个常量字符串，如 `"v1"`）

建议：
- `cache_key = sha256(prompt + "\n" + model + "\n" + str(temperature) + "\n" + schema_version)`
- 缓存存盘位置：
  - 推荐 `run_dir/cache/llm/`（每次 run 独立，方便复现与打包）
- 若 cache_hit：仍然必须写入 llm_calls.jsonl，并标注 `cache_hit=true`

### 2.6 全量日志：`run_dir/logs/llm_calls.jsonl`

每一次“尝试调用/读取缓存/失败回退”都必须写一条 JSONL。

建议字段（你可以增减，但需包含关键信息）：
- `ts`（UTC ISO string）
- `generation`
- `operator: "llm_semop"`
- `model`（string）
- `temperature`（0.0）
- `schema_version`
- `cache_key`（string）
- `cache_hit`（bool）
- `status`：`"ok" | "skipped" | "error" | "invalid_json" | "validation_failed"`
- `latency_sec`（float，可选）
- `error`（string，可选，异常信息）
- `prompt_hash`（string，可选）
- `prompt_preview`/`response_preview`（可选：建议截断到 1-2KB，避免日志爆炸）

注意：日志里 **绝对不能** 包含 api_key。

## 3. LLM 调用实现（工程要求）

仓库内可能存在旧的 LLM 调用代码（例如 `model/LLM.py`），但本 Phase4 实现必须与 FusionOpt 路径解耦，不依赖 legacy runner。

你可以：
- 新写一个最小的 client（例如 `fusionopt/llm_client.py`）
- 或者抽象一个接口，允许未来接入不同 provider

最低要求：
- 若 `model.api_key` 为空：
  - 不进行真实网络调用
  - 记录一次 `status="skipped"` 的 llm_calls
  - 返回 `None`（回退）

## 4. 配置（Config）要求

需要在不破坏现有 config 的基础上增加可选配置。

建议新增（示例，仅供参考）：
```yaml
model:
  name: gemini-2.5-pro
  api_key: ""  # 运行时由用户提供

fusionopt:
  operator_weights:
    llm_semop: 0.05
  llm_semop:
    enabled: true
    trigger:
      type: fixed_interval
      every_n_generations: 5
    batch_size: 8
    temperature: 0.0
    schema_version: v1
```

## 5. 验收标准（必须提供可运行命令）

### 5.1 稳定性

- 在 `model.api_key: ""` 的情况下，运行应当不报错，SemOp 自动跳过并回退。

### 5.2 协议合规

- 任意一次 run 输出目录必须符合：`results/{problem_id}/{algo_id}/{run_id}/`
- `evaluations.csv` 字段满足 `check_phase0_protocol.py` 检查

### 5.3 可追踪

- `run_dir/logs/llm_calls.jsonl` 必须存在
- 即使 api_key 为空，也应记录 `status="skipped"` 的条目（便于验证 trigger/逻辑是否跑到）

### 5.4 至少一个 benchmark 的“低预算更快更好”

- 在至少一个 benchmark（VMEC 或 GSCO-Lite）上，给出一个小预算实验（例如 eval_budget=200 或 500），展示：
  - HV/可行率曲线在早期优于不启用 SemOp 的对照

（注意：这是 Phase4 最终验收项；本任务实现重点是把工程管线搭好，确保可控、可回退、可复现。）

## 6. 你需要提交的产物（Deliverables）

- 代码：实现 LLM-SemOp 所需的新增/修改文件（含缓存与日志）。
- 至少 2 个 config（建议）：
  - `problem/stellarator_vmec/config_phase4_llm_semop_smoke.yaml`
  - `problem/stellarator_coil_gsco_lite/config_phase4_llm_semop_smoke.yaml`
- 一份最小复现说明（可以追加到 `fusionopt.md` 或新增 `PHASE4_REPRODUCE.md`）：
  - 运行命令
  - protocol check 命令
  - 预期输出文件列表（包含 llm_calls.jsonl）

## 7. 你可以参考的现有行为（不要破坏）

- `fusionopt/engine.py` 目前通过 `_gate_and_repair()` 统一处理 gate + HeuRepair。
- `run_fusionopt.py` 会把 config 中的 `model.api_key` 清空后再写入 run_dir/config.yaml。

---

请按以上要求实现，并确保提交内容不包含任何敏感信息（尤其是 api_key）。
