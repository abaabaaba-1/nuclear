 # FusionOpt Roadmap (Phase 0–5)
 
 本文件用于长期保存“FusionOpt”整体推进计划（0–5 阶段）。每个阶段都有可独立收尾的产出与验收标准，便于并行协作与阶段性复盘。
 
 ---
 
 ## 总体目标
 
 - **目标**：构建并验证一个面向核聚变设计（高评估成本、多目标、强约束）的 **FusionOpt** 优化器。
   - 适配本仓库中两个核心问题：
     - `problem/stellarator_vmec`
     - `problem/stellarator_coil_gsco_lite`
 - **协议与检查工具（权威约束）**：
   - `EVAL_PROTOCOL.md`
   - `check_phase0_protocol.py`
 - **问题权威配置（输入规范）**：
   - `problem/stellarator_vmec/config.yaml`
   - `problem/stellarator_coil_gsco_lite/config.yaml`
 
 ### 验收标准（Acceptance）
 - 任意 baseline 能在统一协议下输出可解析结果目录（含 `evaluations.csv` / `config.yaml` / `run_meta.json`）。
 - 相同 seed + 配置可复现关键统计。
 
 ---
 
 ## 关键决策（已确认）
 
 - `protocol.algo_id`：Phase 2 固定为 `fusionopt_v1`。
 - LLM（Phase 4 才使用）：Gemini 2.5，`temperature=0.0`。
   - 记录所有调用（含失败、降级、cache_hit）到：`results/{problem_id}/{algo_id}/{run_id}/logs/llm_calls.jsonl`
   - 必须实现确定性缓存：`cache_key = hash(prompt + model + temperature + schema_version)`
 - 与历史 MOLLM 完全解耦：FusionOpt 新实现不得 import/依赖：
   - `model/MOLLM.py`
   - `algorithm/MOO.py`
   允许复用的基础设施（不算 MOLLM 路径）：
   - `algorithm/base.py`（`Item`, `ItemFactory`）
   - `eval_logger.py`（协议对齐日志）
   - `model/util.py`（`nsga2_selection` 等通用工具）
 - `objective_ranges`：两个 benchmark 的对比实验必须使用同一套 `objective_ranges`（不允许在不同算法/不同 run 间随意改动范围）。
 - 开源安全：写入 run_dir 的 `config.yaml` 必须对 `model.api_key` 做脱敏/清空。
 
 ---
 
 ## Phase 0：评估协议与问题基线（Status: DONE）
 
 ### 目标
 - 明确评估预算口径、结果目录结构、日志 schema，并确保两个 benchmark 的 baseline 输出可以被 `check_phase0_protocol.py` 验证。
 
 ### 关键工作
 - 对齐 `EVAL_PROTOCOL.md`：
   - `results/{problem_id}/{algo_id}/{run_id}/...`
   - `evaluations.csv` / `generations/gen_XXXX_pop.csv` / `final/pareto_front.csv`
 - 确认两个问题的 evaluator 路径与初始种群接口：
   - `evalutor_path: problem.stellarator_vmec.evaluator`
   - `evalutor_path: problem.stellarator_coil_gsco_lite.evaluator`
 
 ### 验收标准（Acceptance）
 - 两个问题至少各 1 个 baseline run：
   - `check_phase0_protocol.py --results_dir <run_dir>` 返回 `ALL_OK: True`。
 
 ---
 
 ## Phase 1：对比基线套件与实验编排（Status: DONE）
 
 ### 目标
 - 让“跑基线、对比、画图”这件事成本足够低，后续迭代才能快。
 
 ### 关键工作
 - 统一 runner：`run_all_baselines.py`。
 - 确保两个问题都至少有：
   - 经典进化/启发式基线（可复现）
   - LLM 驱动的现有优化器基线（用于对比）
 
 ### 验收标准（Acceptance）
 - 可以一键跑出多算法、多 seed 的可对比结果（至少 smoke budget）。
 
 ---
 
 ## Phase 2：FusionOpt v1（多算子骨架 + 统一档案）（Status: IN PROGRESS）
 
 ### 目标
 - 抽象出一个“优化器级别”的主循环：
   - 多算子产生候选
   - 可行性门控（cheap gate）
   - 昂贵评估
   - 统一档案/选择
   - 统一日志（协议对齐）
 
 ### 关键工作
 - 定义算子接口（输入/输出统一为当前 repo 里已有的 JSON 编码）。
 - 先用固定比例混合算子（不做复杂调度）。
 - 目标是：**稳定、可复现、可对比**。
 
 ### 产出（Deliverables）
 - `run_fusionopt.py`
 - `fusionopt/`（或等效模块）
 - `problem/*/config_fusionopt_v1.yaml`（两个问题各一份）
 
 ### 验收标准（Acceptance）
 - 两个 benchmark 上都能跑满预算、输出协议对齐结果。
 - 可行率与无效评估比例不显著劣于最强现有基线。
 
 ### 工程规格（实现级：FusionOpt v1）
 
 目标：把这一节直接喂给大模型，它可以按规格实现 `run_fusionopt.py` + `fusionopt/`，并在 VMEC/GSCO-Lite 上跑通协议对齐的 smoke。
 
 #### 2.1 v1 范围（必须做 / 暂不做）
 
 v1 必须做：
 
 - **完全与 MOLLM runner 解耦**（不得 import `model/MOLLM.py` / `algorithm/MOO.py`）。
 - **按代（generation）主循环**：
   - 初始化种群（复用问题模块中的 `generate_initial_population(config, seed)`）
   - offspring 生成（多算子框架，但 v1 只实现 StdOp：crossover/mutation/resample）
   - cheap gate（JSON 校验/规范化 + 去重）
   - 昂贵评估（调用 `RewardingSystem.evaluate(items)`）
   - 环境选择（复用 `model/util.py: nsga2_selection`）
   - 协议对齐日志（复用 `eval_logger.py: EvalLogger`）
 - **预算口径**：`optimization.eval_budget` 只统计 `eval_tier=true` 的昂贵评估次数。
 
 v1 暂不做：
 
 - 不调用 LLM（LLM-SemOp 放 Phase 4；但 v1 要预留 `logs/llm_calls.jsonl` 的目录位置）。
 - 不做 surrogate / cheap analytic tier（`eval_tier` 先统一为 `true`）。
 - 不做复杂算子自适应调度（先固定权重混合）。
 - 不做严格 resume（可后续加；v1 优先稳定与可复现）。
 
 #### 2.2 CLI / Runner 约定
 
 - **入口脚本**：`run_fusionopt.py`（repo 根目录）。
 - **命令行**：
   - `python run_fusionopt.py <config_path> --seed <int>`
   - 可选：`--run_id <str>`（不传则自动生成）。
 - **运行时必须打印**：
   - `problem_id`
   - `algo_id`
   - `run_dir`（用于直接跑 `check_phase0_protocol.py`）。
 
 `run_id` 推荐格式：`{exper_name}_seed{seed}_{YYYYMMDDTHHMMSS}`。
 
 #### 2.3 结果目录与日志契约（严格对齐 `EVAL_PROTOCOL.md`）
 
 FusionOpt v1 的结果目录必须是：
 
 - `run_dir = {logging.results_base_dir}/{protocol.algo_id}/{run_id}`
 
 且包含（至少）：
 
 ```text
 results/{problem_id}/{algo_id}/{run_id}/
   config.yaml
   run_meta.json
   evaluations.csv
   generations/
     gen_0000_pop.csv
     ...
   final/
     pareto_front.csv
   logs/
     llm_calls.jsonl  # Phase 4 必写；Phase 2 可为空文件或不创建文件但必须有 logs/ 目录
 ```
 
 **开源安全（硬约束）**：因为 `EvalLogger` 会把传入的 `config_data` 原样写入 `run_dir/config.yaml`，所以 runner 在创建 `EvalLogger` 前必须对 `model.api_key` 做脱敏/清空。
 
 #### 2.4 最小配置 Schema（v1）
 
 FusionOpt v1 复用现有问题配置键，并新增 `fusionopt:` 段。建议 runner 使用一个 dotted-key 的 Config wrapper（参考 `baseline_gsco_pymoo.py: Config.get("a.b.c")`），以兼容现有 evaluator 的 `config.get('optimization.pop_size')` 风格。
 
 v1 必需键（两个问题通用）：
 
 - `protocol.problem_id: str`
 - `protocol.algo_id: str`（v1 必须为 `fusionopt_v1`）
 - `evalutor_path: str`（注意仓库里就是这个拼写）
 - `goals: List[str]`
 - `optimization_direction: List[str]`
 - `objective_ranges: Dict[str, [low, high]]`
 - `optimization.pop_size: int`
 - `optimization.eval_budget: int`
 - `optimization.log_freq: int`（建议保留）
 - `seed.master: int`
 - `logging.results_base_dir: str`
 
 v1 新增键：
 
 - `fusionopt.offspring_per_gen: int`（默认 = pop_size）
 - `fusionopt.batch_eval_size: int`（默认 = pop_size；可小于 pop_size）
 - `fusionopt.max_attempts_per_child: int`（避免 gate 太严导致死循环）
 - `fusionopt.operator_weights: {std_crossover, std_mutation, std_resample}`
 - `fusionopt.gate.dedup_scope: run|generation`
 - `fusionopt.gate.canonicalize_json: bool`
 
 最小示例（通用结构；问题特定字段如 `vmec:` / `coil_design:` 原样保留）：
 
 ```yaml
 protocol:
   problem_id: "stellarator_vmec"  # 或 stellarator_coil_gsco_lite
   algo_id: "fusionopt_v1"
 
 seed:
   master: 42
 
 optimization:
   pop_size: 20
   eval_budget: 2000
   log_freq: 20
 
 logging:
   results_base_dir: "./results/stellarator_vmec"
 
 fusionopt:
   offspring_per_gen: 20
   batch_eval_size: 10
   max_attempts_per_child: 50
 
   operator_weights:
     std_crossover: 0.7
     std_mutation: 0.3
     std_resample: 0.0
 
   gate:
     dedup_scope: "run"
     canonicalize_json: true
 ```
 
 #### 2.5 与 `EvalLogger` 的对接硬要求（否则协议会悄悄错）
 
 `EvalLogger.log_batch()` 在写 `evaluations.csv` 时，会优先读取 `item.eval_record`，否则读取：
 
 - `item.property`（raw objectives dict）
 - `item.scores`（min objectives list）
 - `item.constraints`（dict，可为空，但 v1 必须补齐协议字段）
 
 因此 **FusionOpt runner 必须在每次评估后** 为每个 `Item` 写入：
 
 - `item.constraints["status"] ∈ {"ok","sim_fail","invalid_input","internal_error"}`
 - `item.constraints["sim_message"]: str`
 - `item.constraints["cv"]: float`（可行时 0；失败时建议 `1e6`）
 - `item.constraints["g1"]: float`（可直接等于 `cv`）
 - `item.constraints["feasible"]: int`（可行 1，不可行 0）
 
 注意：VMEC evaluator 当前只写 `constraint_results.is_feasible/is_converged/...`，GSCO evaluator 可能只写 `gradient_hints`；二者都不会自动生成上述协议字段。因此需要在 FusionOpt runner 中统一补齐。

 ##### 2.5.1 推荐的 `status/cv/feasible/sim_message` 映射（v1）

 - VMEC：
   - evaluator 会写 `item.constraints["is_feasible"] ∈ {0.0, 1.0}`。
   - 建议映射：
     - 若 `is_feasible == 1.0`：
       - `status="ok"`, `feasible=1`, `cv=0.0`, `sim_message=""`
     - 否则：
       - `status="sim_fail"`, `feasible=0`, `cv=1e6`, `sim_message="infeasible_or_failed"`
 - GSCO-Lite：
   - evaluator 对 forbidden cell 会在 `item.gradient_hints` 中写入 `"VIOLATION: ..."`。
   - 建议映射：
     - 若存在 `h.startswith("VIOLATION:")`：
       - `status="invalid_input"`, `feasible=0`, `cv=1e6`, `sim_message=<该 violation 文本>`
     - 否则：
       - `status="ok"`, `feasible=1`, `cv=0.0`, `sim_message=""`

 ##### 2.5.2 evaluator 可能丢弃无效/重复 item 的处理（防御性约束）

 - GSCO 的 `RewardingSystem.evaluate(items)` 在内部 `_sanitize_and_validate` 可能会丢弃 invalid/repeated 候选并返回更短的 `items`。
 - v1 推荐通过 gate + run-scope 去重尽量避免触发该路径。
 - 若仍发生“返回 items 数量 < 输入数量”，为了保持预算口径与日志一致性，runner 应当为缺失的 decision 补齐 penalty Item：
   - `status="invalid_input"`, `feasible=0`, `cv=1e6`
   - `scores = [1.0]*len(goals)`（确保被支配）
   - `property` 可填为各 objective 的 worst-case（或留空，但会影响 `*_raw` 列）。

 #### 2.6 Decision JSON 与去重（cheap gate 的核心）

 - FusionOpt 内部统一用 `Item.value` 存储 `decision_json`（一个 JSON 字符串）。
 - 去重粒度：以 `decision_json` 的 **canonical 表达**为准。
   - canonicalize：`json.dumps(obj, sort_keys=True, separators=(",", ":"))`
   - 去重 key：`decision_id = md5(decision_json)`（与 `EvalLogger._decision_id` 一致）。
 - `dedup_scope`：
   - `run`：整次 run 内不重复昂贵评估同一个 `decision_id`（推荐）。
   - `generation`：只对当前代 offspring 去重。

 #### 2.7 建议的接口契约（用于解耦 VMEC/GSCO 与 engine）

 （v1 可以不严格按类实现，但建议保持这些函数边界，Phase 3/4 扩展会省很多事。）

 - `ProblemAdapter`：
   - `problem_id: str`
   - `goals: List[str]`
   - `make_item_factory(goals) -> ItemFactory`
   - `generate_initial_decisions(config, seed) -> List[str]`
   - `gate(decision_json: str, config, rng) -> Optional[str]`（返回 canonical JSON；失败返回 None）
   - `std_resample(config, rng) -> str`
   - `std_mutation(parent_json: str, config, rng) -> str`
   - `std_crossover(parent_a_json: str, parent_b_json: str, config, rng) -> str`

 - `FusionOptEngine`：
   - 输入：`config_data`, `seed`, `run_id`
   - 输出：协议对齐的 run_dir（runner 打印）
   - 责任：budget 控制、offspring 生成调度、评估、选择、日志。

 #### 2.8 cheap gate 细则（v1 必须实现）

 gate 的目标：在不消耗昂贵评估预算的情况下，尽量把 candidate 投影到“输入合法 + 更不容易让 evaluator 崩”的区域。

 通用 gate（两问题都做）：

 - JSON 必须可解析为 dict。
 - canonicalize 输出（用于稳定去重）。
 - **不在 gate 阶段调用昂贵求解器**。

 VMEC gate（建议规则）：

 - schema 必须是：`{"new_coefficients": { ... }}`。
 - `new_coefficients` 必须是 dict；value 必须可转 float。
 - key 规范化：`key.strip().replace(" ", "")`。
 - clamp：直接调用 evaluator 实例上的 `RewardingSystem._sanitize_new_coefficients(new_coefficients)`。
 - gate 输出：`json.dumps({"new_coefficients": sanitized}, sort_keys=True, separators=(",", ":"))`。

 GSCO-Lite gate（建议规则）：

 - schema 必须是：`{"cells": [[phi, theta, state], ...]}`。
 - 每个 cell 必须能解析出 `(phi:int, theta:int, state:int)`：
   - `phi ∈ [0, coil_design.wf_nPhi-1]`
   - `theta ∈ [0, coil_design.wf_nTheta-1]`
   - `state ∈ {-1, 1}`（若给了 0 则视为 inactive 丢弃；若给了其他值则投影到 -1/1）。
 - 去重：同一 `(phi, theta)` 只保留最后出现的 state。
 - 数量约束：
   - `min_active_cells = llm_constraints.min_active_cells`
   - `max_active_cells = llm_constraints.max_active_cells`
   - 超过上限：截断
   - 不足下限：随机补齐（避免 evaluator `_sanitize_and_validate` 把空 cells 判失败）。
 - forbidden_cells：Phase2 建议直接移除 forbidden cells（并在不足 min_cells 时补齐）。

 #### 2.9 StdOp（v1：只实现标准算子）

 v1 的 StdOp 目标：不依赖 LLM 的前提下，能稳定地产生多样化 offspring，且大部分能通过 gate。

 VMEC StdOp（建议）：

 - `std_resample`：调用问题模块的 `generate_initial_population` 或从 baseline coeffs 随机扰动生成。
 - `std_mutation(parent)`：
   - 解析 parent 的 `new_coefficients` dict。
   - 随机选 `k` 个 key（`k <= llm_constraints.max_coeff_changes`），对 value 做相对扰动。
   - 输出必须再走 VMEC gate（保证 clamp）。
 - `std_crossover(parent_a, parent_b)`：
   - 解析两者 delta dict。
   - key-level 混合（例如 union 后随机采样子集；重叠 key 二选一或均值）。
   - 输出必须再走 VMEC gate。

 GSCO-Lite StdOp（建议，参考 `baseline_gsco_pymoo.py`）：

 - `std_mutation`：四类之一（flip / move / add / remove），再走 GSCO gate。
 - `std_crossover`：对 cell list 排序后 one-point crossover，再走 GSCO gate。
 - `std_resample`：调用 `generate_initial_population` 或随机采样 cells，再走 GSCO gate。

 #### 2.10 主循环伪代码（v1）

 ```text
 load config
 set random seeds (random, numpy)
 import module = __import__(evalutor_path)
 evaluator = module.RewardingSystem(ConfigWrapper(config_data))
 init ItemFactory(goals)
 build run_dir = logging.results_base_dir / algo_id / run_id
 sanitize config_data (clear model.api_key) then init EvalLogger(run_dir,...)
 ensure run_dir/logs exists

 # init population
 pop_jsons = adapter.generate_initial_decisions(config, seed)
 pop_jsons = [gate(x) ...] and resample until size == pop_size
 pop_items = evaluate(pop_jsons); fill item.constraints protocol fields
 logger.log_batch(pop_items, generation=0, tier=true)
 population = pop_items
 n_true_evals = len(population)

 generation = 0
 while n_true_evals < eval_budget:
   generation += 1
   offspring_jsons = []
   attempts = 0
   while len(offspring_jsons) < offspring_per_gen and attempts < max_attempts_per_child * offspring_per_gen:
     op = sample_by_weight(operator_weights)
     parents = sample parents from population
     child = op(parents)
     child = gate(child)
     if child is None: attempts += 1; continue
     if dedup_scope=run and child_id in seen: attempts += 1; continue
     offspring_jsons.append(child); mark seen
     attempts += 1

   # expensive eval (may need to truncate to remaining budget)
   offspring_jsons = offspring_jsons[: (eval_budget - n_true_evals)]
   offspring_items = evaluate(offspring_jsons); fill item.constraints
   logger.log_batch(offspring_items, generation=generation, tier=true)
   n_true_evals += len(offspring_items)

   pool = population + offspring_items
   population = nsga2_selection(pool, pop_size)

 logger.close()
 ```

 #### 2.11 最小 smoke 与验收命令

 - 两个问题各准备 1 个 smoke config：
   - `eval_budget=20, pop_size=10, log_freq=10`
 - 运行：
   - `python run_fusionopt.py problem/stellarator_vmec/config_fusionopt_v1_smoke.yaml --seed 42`
   - `python check_phase0_protocol.py --results_dir <printed_run_dir>`
   - `python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_fusionopt_v1_smoke.yaml --seed 42`
   - `python check_phase0_protocol.py --results_dir <printed_run_dir>`

 ---
 
 ## Phase 3：HeuRepairOp（领域修复算子）（Status: DONE）
 
 ### 目标
 - 在不调用 LLM 的前提下，用“领域规则”提高有效评估率。
 
 ### 关键工作
 - VMEC：形状平滑/高阶抑制/步长限制等可行修复。
 - GSCO-Lite：去重、间距/禁区约束修复、曲率尖峰抑制等可行修复。
 
 ### 验收标准（Acceptance）
 - 在相同评估预算下，FusionOpt 的可行率与 HV/收敛曲线出现可解释提升趋势。
 
 ---
 
 ## Phase 4：LLM-SemOp（语义算子 + 异步批量）（Status: DONE）
 
 ### 目标
 - 将 LLM 放在“语义级编辑提案”位置：低频触发、批量生成、异步不阻塞。
 
 ### 关键工作
 - 统一 LLM 输出 schema（JSON action/edit），并做严格校验与投影。
 - 触发策略：停滞检测/异常事件/固定间隔（先固定间隔）。
 - 固定模型超参（`temperature=0.0`），并实现确定性缓存（见“关键决策”）。
 - 缓存与降级策略：
   - 如果 response 无法解析为目标 schema，或校验失败，则回退 StdOp/HeuRepairOp。
   - 即使 cache_hit 也要写入 `llm_calls.jsonl`（标注 cache_hit=true）。

### 当前进展（Progress）

- VMEC（Phase4HardV6）在 `budget=1000`（seed=42）下，LLM-SemOp 出现明显增益（`HV/top1` 更高，且 `feasible_rate` 略高），说明“大预算更好”的趋势成立。
- VMEC（Phase4HardV6）在 `budget=1000`（seed=46）下也复现增益（该 seed 首次运行因手动关闭 IDE 中断，以下为 rerun 结果）：
  - Baseline：`feasible_rate=0.559`，`HV=1.1961`，`top1(total)=0.9089`
  - LLM-SemOp：`feasible_rate=0.597`，`HV=1.2293`，`top1(total)=0.9424`
  - 我们将不再继续其他 seed 的 `budget=1000` 计算。
- GSCO-Lite（budget=200, seed=42）已完成 baseline vs LLM-SemOp 对比：
  - Baseline：`feasible_rate=1.0`，`HV=0.5810`，`top1(total)=2.2992`
  - LLM-SemOp：`feasible_rate=1.0`，`HV=0.6113`，`top1(total)=2.3287`
- GSCO-Lite（budget=200, seed=43）已完成 baseline vs LLM-SemOp 对比：
  - Baseline：`feasible_rate=1.0`，`HV=0.5752`，`top1(total)=2.2936`
  - LLM-SemOp：`feasible_rate=1.0`，`HV=0.6152`，`top1(total)=2.3326`
- GSCO-Lite（Phase4Hard, budget=200, seed=42）为避免可行率饱和，引入 hard feasibility（例如 `f_B<=14.4`, `f_S<=10`, `I_max<=0.2`）并复跑：
  - Baseline：`feasible_rate=0.265`，`HV=0.5810`，`top1(total)=2.2992`
  - LLM-SemOp：`feasible_rate=0.295`，`HV=0.6113`，`top1(total)=2.3287`
- LLM 调用可追踪：GSCO-Lite 该次 run 的 `logs/llm_calls.jsonl` 有明确记录（`status=ok`）。

- VMEC 经典 baselines（GA / NSGA2，`eval_budget=1000`, `seed=42`）已跑完；为与 FusionOpt 的 Phase4HardV6 指标口径一致（`HV` 使用 `f*_min`，ref=`[1.1,1.1,1.1]`），我们基于保存的 raw properties 重新归一化并计算：
  - GA：`feasible_rate=0.9985`，`HV=0.2635`，`top1(total)=0.5397`
  - NSGA2：`feasible_rate=0.9990`，`HV=0.2635`，`top1(total)=0.5397`

- 备注：VMEC 的 `magnetic_shear` 在不同配置中若使用过窄的 `objective_ranges`，会导致 clipping（饱和）从而影响 `HV/top1` 的解释；Phase4HardV6 已使用更贴近可行区域的范围（如 `magnetic_shear=[0.85,1.05]`），但仍可能出现一定比例的 clipping，验收时需说明该风险。

 ### 验收标准（Acceptance）
 - 不显著降低运行稳定性。
 - LLM 调用次数可追踪、可控。
 - 在至少一个 benchmark 上达到“低预算段更快更好”的明确增益。
 
 ---
 
 ## Phase 5：冲榜与论文（Status: PENDING）
 
 ### 目标
 - 以“昂贵评估优化器”为主叙事，完成系统性实验与论文。
 
 ### 关键工作
 - 多 seed 统计、anytime 曲线、消融（StdOp / +HeuRepairOp / +LLM-SemOp）。
 - 目标导向：尽可能在两个 benchmark 的主指标上争取第一；若无法同时第一，明确主张（样本效率/鲁棒性/可行率）。
 
 ### 验收标准（Acceptance）
 - 全链路可复现（配置 → 结果 → 图表）。
 - 结论稳健（跨 seed 统计支持）。
 
 ---
 
 ## 附：协作与里程碑建议
 
 - 建议每个 Phase 结束产出一个 `PHASE_X_REPORT.md`：
   - 本阶段做了什么
   - 可复现实验命令
   - 主要结果与下一阶段风险
 - 建议每次大改动都绑定一个最小 smoke test，避免回归。
