import time
import logging
import sys
import yaml
import numpy as np
from pathlib import Path
import json

# --- 1. 设置环境 ---

# 将 MOLLM-main 根目录添加到 Python 路径
# 这允许我们导入 'problem.stellarator_vmec.evaluator'
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    import vmecpp # 确保 vmecpp 已经安装在您的 mollm_env 环境中
    from problem.stellarator_vmec.evaluator import RewardingSystem, generate_initial_population, _analyze_wout_file
    from problem.stellarator_vmec.vmec_file_modifier import VmecFileModifier
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保您已激活 'mollm_env' 虚拟环境 (pip install vmecpp)")
    print("并且此脚本位于 MOLLM-main 根目录中。")
    sys.exit(1)


# --- 2. 从您的框架中复制/模拟必要的类 ---

# 复制自您的 MOLLM.py 以加载配置
class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        self.config = self._load_config(self.config_path)

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# 一个模拟的 'Item' 类，以匹配您的算法框架
class MockItem:
    def __init__(self, value_json_string):
        self.value = value_json_string
        self.results = None
        # print(f"  > 创建 MockItem...")
            
    def assign_results(self, results_dict):
        self.results = results_dict
        feasible = results_dict.get('constraint_results', {}).get('is_feasible', 0.0)
        print(f"  > 结果已分配: Feasible={feasible}")

# --- 3. 定义测试函数 ---

def run_baseline_test(config, candidate_json):
    """
    测试 1: 基准测试 (您当前的 evaluator.py)
    - 冷启动
    - 默认 (多) 线程
    - 严格的公差 (来自文件)
    """
    print("\n--- TEST 1: BASELINE (As-is evaluator.py) ---")
    print("    策略: 冷启动, 默认线程, 严格公差")
    
    # 每次测试都使用新的 RewardingSystem 实例
    rs = RewardingSystem(config)
    item = MockItem(candidate_json)
    
    start_time = time.time()
    try:
        # 调用您文件中的原始 evaluate 方法
        rs.evaluate([item])
        end_time = time.time()
        print(f"    原始 evaluator.py 运行完成。")
        print(f"    Time taken for 1 run: {end_time - start_time:.4f} seconds")
    except Exception as e:
        end_time = time.time()
        print(f"    运行失败: {e} (耗时: {end_time - start_time:.4f}s)")
        
    if item.results:
        print(f"    最终结果: {item.results.get('constraint_results')}")
    else:
        print("    最终结果: 运行未成功分配结果。")


def run_parallel_optimized_test(config, candidate_json, N=3):
    """
    测试 2: 并行优化测试 (模拟多进程)
    - 冷启动
    - max_threads = 1
    - 放宽的公差
    """
    print(f"\n--- TEST 2: PARALLEL-OPTIMIZED (N={N} 次冷启动) ---")
    print("    策略: 冷启动, max_threads=1, 放宽公差 (1e-11)")
    
    rs = RewardingSystem(config)
    total_time = 0
    
    for i in range(N):
        print(f"  Run {i+1}/{N}...")
        item = MockItem(candidate_json) # 每次都用同一个候选解
        start_time = time.time()
        
        # --- 这是 evaluator.py 逻辑的 *修改* 版本 ---
        try:
            modifications = json.loads(item.value)
            new_coefficients = modifications.get("new_coefficients")
            rs.modifier.replace_coefficients(new_coefficients)
            
            vmec_input = vmecpp.VmecInput.from_file(rs.modifier.input_file_path)
            
            # 关键修复 1: 放宽公差
            vmec_input.ftol_array = np.array([3.0E-11])
            
            run_args = {
                "input": vmec_input,
                "max_threads": 1 # 关键修复 2: 优化并行
            }
            
            vmec_output = vmecpp.run(**run_args)
            vmec_output.wout.save(rs.output_file_path)
            
            analysis_res = _analyze_wout_file(rs.output_file_path)
            
            is_converged = analysis_res.get('is_converged', False)
            min_mercier = analysis_res.get('min_mercier', -999.0)
            is_stable = min_mercier > 0
            is_feasible = is_converged and is_stable
            
            item.assign_results({
                'constraint_results': {'is_feasible': 1.0 if is_feasible else 0.0, 'is_converged': is_converged, 'is_stable': is_stable}
            })

        except Exception as e:
            print(f"    > Run {i+1} failed: {e}")
        
        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        print(f"    > Run {i+1} time: {run_time:.4f}s")
            
    print(f"    Total time for {N} *独立冷启动*: {total_time:.4f} seconds")
    print(f"    Average time per run: {total_time / N:.4f} seconds")

        
def run_hot_start_test(config, candidate_json, N=3):
    """
    测试 3: 热启动测试
    - 链式热启动 (第1次冷, 之后热)
    - max_threads = 1
    - 放宽的公差
    - 健壮的锚点保存
    """
    print(f"\n--- TEST 3: HOT-START (N={N} 次链式运行) ---")
    print("    策略: 链式热启动, max_threads=1, 放宽公差, 健壮锚点")
    
    rs = RewardingSystem(config)
    last_good_output = None # 热启动锚点
    total_time = 0
    
    for i in range(N):
        print(f"  Run {i+1}/{N}...")
        # 在真实GA中, 这里的 candidate_json 每次都会*略有不同*
        # 为简单起见，我们重复使用同一个
        item = MockItem(candidate_json) 
        start_time = time.time()
        
        # --- 这是 *完整修复* 后的 evaluator.py 逻辑 ---
        try:
            modifications = json.loads(item.value)
            new_coefficients = modifications.get("new_coefficients")
            rs.modifier.replace_coefficients(new_coefficients)
            
            vmec_input = vmecpp.VmecInput.from_file(rs.modifier.input_file_path)
            
            # 关键修复 1: 放宽公差
            vmec_input.ftol_array = np.array([3.0E-11])
            
            run_args = {
                "input": vmec_input,
                "max_threads": 1 # 关键修复 2: 优化并行
            }
            
            # 关键修复 3: 应用热启动
            if last_good_output is not None:
                print("    > 应用热启动...")
                run_args['restart_from'] = last_good_output
            else:
                print("    > 冷启动 (Run 1)...")
            
            vmec_output = vmecpp.run(**run_args)
            
            # 关键修复 4: 健壮的锚点保存
            # 无论是否可行，都保存锚点以供下一次启动
            print("    > 存储锚点用于下一次热启动。")
            last_good_output = vmec_output 
            
            vmec_output.wout.save(rs.output_file_path)
            
            analysis_res = _analyze_wout_file(rs.output_file_path)
            
            is_converged = analysis_res.get('is_converged', False)
            min_mercier = analysis_res.get('min_mercier', -999.0)
            is_stable = min_mercier > 0
            is_feasible = is_converged and is_stable

            item.assign_results({
                'constraint_results': {'is_feasible': 1.0 if is_feasible else 0.0, 'is_converged': is_converged, 'is_stable': is_stable}
            })

        except Exception as e:
            print(f"    > Run {i+1} failed: {e}")
            # 如果失败, 不更新 last_good_output, 下次将重用上一个*成功*的锚点
            
        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        print(f"    > Run {i+1} time: {run_time:.4f}s")
            
    print(f"    Total time for {N} *链式* 运行: {total_time:.4f} seconds")
    print(f"    Average time per run: {total_time / N:.4f} seconds (Run 1是冷的, 2-{N}是热的)")

# --- 4. 主执行函数 ---
def main():
    # 将日志级别调高，减少 evaluator.py 中的 "Infeasible" 警告刷屏
    logging.basicConfig(level=logging.ERROR) 
    
    config_path = "problem/stellarator_vmec/config.yaml"
    print(f"正在加载配置文件: {config_path}")
    try:
        config = ConfigLoader(config_path)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 at '{config_path}'")
        print("请确保此脚本与 'problem' 文件夹位于同一目录 (MOLLM-main 根目录)。")
        return

    # 将 project_path 解析为绝对路径，以提高健壮性
    try:
        abs_path = str(Path(config.get('vmec.project_path')).resolve())
        config.config['vmec']['project_path'] = abs_path
        print(f"VMEC 项目路径解析为: {abs_path}")
        if not Path(abs_path).exists():
            print(f"警告: 路径 '{abs_path}' 不存在。")
    except Exception as e:
        print(f"配置 vmec.project_path 时出错: {e}")
        return

    print("正在生成一个初始候选解用于测试...")
    # 使用固定的 seed=42 保证可复现性
    candidates_list = generate_initial_population(config, 42)
    if not candidates_list:
        print("错误: 生成初始种群失败。")
        print("请检查 evaluator.py 中的 'generate_initial_population' 函数和文件路径。")
        return
        
    test_candidate_json = candidates_list[0]
    
    # --- 运行所有测试 ---
    run_baseline_test(config, test_candidate_json)
    run_parallel_optimized_test(config, test_candidate_json, N=3)
    run_hot_start_test(config, test_candidate_json, N=3)

if __name__ == "__main__":
    main()