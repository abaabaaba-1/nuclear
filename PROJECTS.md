# 工程项目与脚本分类映射

## 导管架几何优化（SACS Jacket Geometry）
- 入口脚本：`baseline_nsga2_sacs_geo_jk.py`
- 配置文件：`problem/sacs_geo_jk/config.yaml`
- 评估与接口：
  - `problem/sacs_geo_jk/evaluator.py`
  - `problem/sacs_geo_jk/sacs_file_modifier.py`
  - `problem/sacs_geo_jk/sacs_interface_weight_improved.py`
  - `problem/sacs_geo_jk/sacs_interface_uc.py`

运行示例：
```bash
python baseline_nsga2_sacs_geo_jk.py problem/sacs_geo_jk/config.yaml
```

## 海上平台几何优化（SACS Offshore Platform Geometry）
- 入口脚本：`baseline_nsga2_sacs_geo_pf.py`、`baseline_nsga2_sacs_geo_pf_v3.py`
- 配置文件：`problem/sacs_geo_pf/config.yaml`

运行示例：
```bash
python baseline_nsga2_sacs_geo_pf.py problem/sacs_geo_pf/config.yaml
```

## 通用基线（按配置决定工程项目）
- GA 基线：`baseline_ga.py`（默认指向导管架几何，可替换为任意配置）
- NSGA-II 基线（通用）：`baseline_nsga2.py`
- SMS-EMOA 基线（通用）：`baseline_sms.py`
- MOEA/D 基线（通用）：`baseline_moead.py`
- 随机搜索基线（通用）：`baseline_rs.py`
- LLM 通用基线：`baseline_llm_generic.py`

运行示例：
```bash
# 导管架几何（示例）
python baseline_ga.py problem/sacs_geo_jk/config.yaml

# 海上平台几何（示例）
python baseline_ga.py problem/sacs_geo_pf/config.yaml
```

## 核聚变（Stellarator/VMEC）
- 配置目录：`problem/stellarator_vmec/`
- 示例配置：`problem/stellarator_vmec/config.yaml`
- 入口脚本：按需复用通用基线脚本（如 `baseline_ga.py`），通过传入该目录下配置文件运行。

运行示例：
```bash
python baseline_ga.py problem/stellarator_vmec/config.yaml
```

## 命名与路径约定
- 入口脚本统一以 `baseline_<算法>[_<问题>]` 命名。
- 问题域配置放在 `problem/<问题>/config.yaml`，脚本默认参数已修正为该路径结构。
- 若需新增工程项目，建议新建 `problem/<新问题>/` 并使用通用基线脚本传入配置运行。
