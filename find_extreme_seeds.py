import yaml
import logging
import json
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# --- 动态添加项目路径以导入模块 ---
sys.path.append(str(Path.cwd()))
try:
    from problem.sacs.evaluator import generate_initial_population, RewardingSystem
except ImportError as e:
    print(f"导入错误: {e}\n请确保此脚本位于项目根目录 (MOLLM-main) 下。")
    sys.exit(1)

# --- 模拟主程序中的Item和Config对象 ---
class MockItem:
    def __init__(self, value_str):
        self.value = value_str
        self.results = {}
    def assign_results(self, results_dict):
        self.results = results_dict

class ConfigWrapper:
    def __init__(self, config_dict):
        self._config = config_dict
    def get(self, key, default=None):
        keys, val = key.split('.'), self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

def find_extreme_seeds():
    """
    通过大规模随机抽样、评估和筛选，寻找在各个目标维度上表现极端的初始种子。
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("ExtremeSeedFinder")

    # 1. 加载配置文件
    config_path = 'problem/sacs/config.yaml'
    logger.info(f"从 {config_path} 加载配置...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件时出错: {e}"); return

    # 2. 生成大规模随机样本
    num_samples = 2000
    seed = 123  # 使用一个固定的种子以保证抽样过程可复现
    
    # 临时修改配置中的种群大小以生成所需数量的样本
    config_data['optimization']['pop_size'] = num_samples
    config = ConfigWrapper(config_data)

    logger.info(f"开始生成 {num_samples} 个随机候选样本...")
    population_jsons = generate_initial_population(config, seed)
    
    # 移除第一个未经修改的基线模型，只评估随机生成的样本
    if len(population_jsons) > 1:
        population_jsons = population_jsons[1:]
    
    logger.info(f"成功生成 {len(population_jsons)} 个独特的候选样本。")
    if not population_jsons:
        logger.error("未能生成任何候选样本，程序中止。"); return

    # 3. 全面评估所有样本
    items_to_evaluate = [MockItem(json_str) for json_str in population_jsons]
    rewarding_system = RewardingSystem(config)
    logger.info(f"开始全面评估 {len(items_to_evaluate)} 个样本，这可能需要数小时，请耐心等待...")

    # 使用tqdm添加进度条
    for item in tqdm(items_to_evaluate, desc="正在评估样本"):
        rewarding_system.evaluate([item]) # 每次评估一个以显示进度

    # 4. 分析结果，筛选极端种子
    logger.info("评估完成。正在分析结果以筛选极端种子...")
    results_list = []
    for item in items_to_evaluate:
        if 'error_reason' not in item.results: # 筛选掉评估失败的样本
            res = item.results['original_results']
            res['max_uc'] = item.results['constraint_results']['max_uc']
            res['json'] = item.value
            results_list.append(res)

    if not results_list:
        logger.error("没有成功评估的有效候选方案，无法筛选种子。"); return

    # 使用Pandas DataFrame进行高效筛选
    df = pd.DataFrame(results_list)
    logger.info(f"在 {len(df)} 个有效方案中进行筛选...")

    # 筛选出各个维度的最优者
    seed_lightest = df.loc[df['weight'].idxmin()]
    seed_lowest_overall_uc = df.loc[df['max_uc'].idxmin()]
    seed_lowest_axial_uc = df.loc[df['axial_uc_max'].idxmin()]
    seed_lowest_bending_uc = df.loc[df['bending_uc_max'].idxmin()]

    # 5. 打印最终筛选出的种子
    def print_seed(name, seed_series):
        logger.info(f"\n{'='*10} 种子: {name} {'='*10}")
        logger.info(f"  性能 -> 重量: {seed_series['weight']:.2f} 吨 | 综合UC: {seed_series['max_uc']:.4f} | 轴向UC: {seed_series['axial_uc_max']:.4f} | 弯曲UC: {seed_series['bending_uc_max']:.4f}")
        logger.info("  可以直接复制用于 'evaluator.py' 的 SEED_BASELINE 格式:")
        try:
            parsed_json = json.loads(seed_series['json'])
            # 构造一个可以直接粘贴的Python字典格式字符串
            seed_name = "SEED_" + name.replace(" ", "_").upper()
            print(f"\n{seed_name} = " + json.dumps(parsed_json, indent=4))
        except Exception as e:
            logger.error(f"打印JSON时出错: {e}")
            print(seed_series['json'])

    print("\n\n")
    logger.info("*"*25 + " 发现的极端种子 " + "*"*25)
    print_seed("重量最轻", seed_lightest)
    print_seed("综合UC最低", seed_lowest_overall_uc)
    print_seed("轴向UC最低", seed_lowest_axial_uc)
    print_seed("弯曲UC最低", seed_lowest_bending_uc)
    logger.info("*"*65)
    logger.info("操作建议: 将上方打印出的 SEED_* 字典复制到 'evaluator.py' 文件中，并将它们添加到 INITIAL_SEEDS 列表中，例如 `INITIAL_SEEDS = [SEED_BASELINE, SEED_LIGHTEST_WEIGHT, ...]`。")

if __name__ == "__main__":
    find_extreme_seeds()