# run_rs_baseline.py

"""
独立的随机搜索（Random Search）基准测试脚本。

该脚本旨在作为 MOLLM 优化框架的一个基础基准。
它执行一个简单的随机搜索，直到达到评估预算，并以与 MOLLM 框架
完全兼容的格式保存结果（.pkl 数据文件和 .json 指标文件）。

修改：增加了周期性日志记录功能，以生成用于绘图的性能曲线。
"""
# =========================================================================
# 零部分：路径修正
# =========================================================================
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# =========================================================================
# 第一部分：模块导入
# =========================================================================
import argparse
import random
import json
import pickle
import time
import numpy as np
from tqdm import tqdm

from model.MOLLM import ConfigLoader
from algorithm.base import ItemFactory
from problem.sacs_geo_jk.evaluator import RewardingSystem, generate_initial_population
from model.util import nsga2_so_selection, top_auc, cal_hv


# =========================================================================
# 第二部分：指标计算辅助函数
# =========================================================================

# =========================================================================
# 第二部分：指标计算辅助函数 (已修正，与MOO对齐)
# =========================================================================

def calculate_and_log_metrics(
    all_evaluated_items,
    history_values,
    failed_evals,
    evaluated_count,
    config,
    start_time,
    json_filename,
    final_metrics_results_list,
    is_final_log=False
):
    """
    计算当前所有评估点的性能指标，并更新JSON文件。
    HV计算逻辑已修正，与MOO框架对齐（基于Top 100）。
    """
    eval_budget = config.get('optimization.eval_budget')
    total_generated = len(all_evaluated_items)
    unique_count = len(history_values)
    
    if total_generated == 0:
        return # 避免除零错误

    uniqueness = unique_count / total_generated
    validity = (total_generated - failed_evals) / total_generated

    # AUC计算仍然使用完整的历史数据，步长与主框架保持一致（freq_log=100）
    auc1 = top_auc(all_evaluated_items, top_n=1, finish=is_final_log, freq_log=100, max_oracle_calls=eval_budget)
    auc10 = top_auc(all_evaluated_items, top_n=10, finish=is_final_log, freq_log=100, max_oracle_calls=eval_budget)
    auc100 = top_auc(all_evaluated_items, top_n=100, finish=is_final_log, freq_log=100, max_oracle_calls=eval_budget)

    valid_items = [item for item, _ in all_evaluated_items if item.scores is not None]
    
    if valid_items:
        # --- 核心修正点 ---
        # 1. 按 'total' score 降序排序所有有效解
        sorted_items = sorted(valid_items, key=lambda item: item.total, reverse=True)
        
        # 2. 取 Top 100 用于计算HV，与 MOO.py 保持一致
        top100_for_hv = sorted_items[:100]
        scores_for_hv = np.array([item.scores for item in top100_for_hv])
        hypervolume = cal_hv(scores_for_hv)
        # --- 修正结束 ---

        # avg_top 指标仍然基于完整排序的列表计算
        avg_top1 = sorted_items[0].total if sorted_items else 0
        avg_top10 = np.mean([i.total for i in sorted_items[:10]]) if len(sorted_items) >= 10 else (np.mean([i.total for i in sorted_items]) if sorted_items else 0)
        avg_top100 = np.mean([i.total for i in sorted_items[:100]]) if len(sorted_items) >= 100 else (np.mean([i.total for i in sorted_items]) if sorted_items else 0)
    else:
        hypervolume, avg_top1, avg_top10, avg_top100 = 0.0, 0.0, 0.0, 0.0
        
    current_metrics = {
        'all_unique_moles': unique_count, 'Uniqueness': uniqueness, 'Validity': validity,
        'Training_step': evaluated_count, 'avg_top1': avg_top1, 'avg_top10': avg_top10,
        'avg_top100': avg_top100, 'top1_auc': auc1, 'top10_auc': auc10, 'top100_auc': auc100,
        'hypervolume': hypervolume, 'div': 0, 'generated_num': total_generated,
        'running_time[s]': time.time() - start_time
    }
    
    # 后续保存逻辑不变...
    final_metrics_results_list.append(current_metrics)
    final_json_data = {'params': config.to_string(), 'results': final_metrics_results_list}

    with open(json_filename, 'w') as f:
        json.dump(final_json_data, f, indent=4)
    
    if is_final_log:
        print("\n--- 最终指标摘要 ---")
        for key, value in current_metrics.items():
            if isinstance(value, float): print(f"{key:<20}: {value:.4f}")
            else: print(f"{key:<20}: {value}")
    else:
        print(f"\n[Log @ Eval {evaluated_count}] avg_top10: {avg_top10:.4f}, HV: {hypervolume:.4f}, Unique: {unique_count}")



# =========================================================================
# 第三部分：主驱动脚本
# =========================================================================

def run_rs_baseline(config_path: str, seed: int):
    """
    执行随机搜索主函数。
    """
    # ---------------------------------------------------------------------
    # 初始化阶段
    # ---------------------------------------------------------------------
    print("=" * 30)
    print(">>> 1. 初始化随机搜索基准 <<<")
    print("=" * 30)

    # 若未以 problem/ 开头，补上目录前缀（本脚本的 ConfigLoader 不会自动加）
    cfg_path = config_path
    if not (cfg_path.startswith('problem/') or os.path.isabs(cfg_path)):
        cfg_path = os.path.join('problem', cfg_path)
    config = ConfigLoader(cfg_path)
    random.seed(seed)
    np.random.seed(seed)
    print(f"配置 '{config_path}' 已加载，随机种子设置为: {seed}")

    start_time = time.time()

    exper_name = config.get('exper_name') + "_RS_Baseline"
    eval_budget = config.get('optimization.eval_budget')
    log_freq = config.get('optimization.log_freq', 50) # 从配置读取日志频率, 默认50
    goals = config.get('goals')
    print(f"实验名称: {exper_name}")
    print(f"评估预算: {eval_budget}, 日志记录频率: {log_freq}")

    item_factory = ItemFactory(goals)
    reward_system = RewardingSystem(config=config)
    print("ItemFactory 和 RewardingSystem 初始化完成。")

    model_name_folder = config.get('model.name').split(',')[-1]
    base_save_dir = os.path.join(config.get('save_dir'), model_name_folder)
    
    mols_save_dir = os.path.join(base_save_dir, 'mols')
    results_save_dir = os.path.join(base_save_dir, 'results')
    os.makedirs(mols_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)
    
    save_suffix = config.get('save_suffix')
    pkl_filename = os.path.join(mols_save_dir, f"{'_'.join(goals)}_{save_suffix}_{seed}.pkl")
    json_filename = os.path.join(results_save_dir, f"{'_'.join(goals)}_{save_suffix}_{seed}.json")
    print(f"PKL 结果将保存至: {pkl_filename}")
    print(f"JSON 指标将保存至: {json_filename}")

    # ---------------------------------------------------------------------
    # 主搜索循环
    # ---------------------------------------------------------------------
    print("\n" + "=" * 30)
    print(">>> 2. 开始随机搜索循环 <<<")
    print("=" * 30)

    print(f"正在一次性生成 {eval_budget} 个随机候选解...")
    original_pop_size = config.config['optimization']['pop_size']
    config.config['optimization']['pop_size'] = eval_budget * 2 # 生成2倍预算以处理重复项
    all_candidate_strings = generate_initial_population(config, seed)
    config.config['optimization']['pop_size'] = original_pop_size # 恢复原始pop_size
    unique_candidate_strings = list(dict.fromkeys(all_candidate_strings))
    random.shuffle(unique_candidate_strings)
    print(f"生成完成，得到 {len(unique_candidate_strings)} 个唯一的候选解。开始评估...")

    all_evaluated_items = []
    history_values = set()
    failed_evals = 0
    final_metrics_results = [] # <--- 用于存储每个时间点的指标

    pbar = tqdm(total=eval_budget, desc="随机搜索评估")
    
    evaluated_count = 0
    for candidate_string in unique_candidate_strings:
        if evaluated_count >= eval_budget:
            break

        new_item = item_factory.create(candidate_string)
        if new_item.value in history_values: continue

        reward_system.evaluate([new_item])
        evaluated_count += 1
        pbar.update(1)

        if new_item.total is None or new_item.total <= 0:
            failed_evals += 1
        
        all_evaluated_items.append((new_item, evaluated_count))
        history_values.add(new_item.value)

        # *** 核心修改：周期性记录指标 ***
        if evaluated_count % log_freq == 0 and evaluated_count > 0:
            calculate_and_log_metrics(
                all_evaluated_items, history_values, failed_evals, evaluated_count,
                config, start_time, json_filename, final_metrics_results, is_final_log=False
            )

    pbar.close()
    
    if evaluated_count < eval_budget:
        print(f"\n警告: 唯一候选解不足，仅评估了 {evaluated_count}/{eval_budget} 个。")
        
    running_time_hours = (time.time() - start_time) / 3600
    print(f"\n随机搜索完成。总耗时: {running_time_hours:.3f} 小时。")

    # ---------------------------------------------------------------------
    # 收尾与最终结果保存
    # ---------------------------------------------------------------------
    print("\n" + "=" * 30)
    print(">>> 3. 保存最终结果 <<<")
    print("=" * 30)
    
    # *** 核心修改：进行最后一次指标记录 ***
    print("正在计算并保存最终指标...")
    calculate_and_log_metrics(
        all_evaluated_items, history_values, failed_evals, evaluated_count,
        config, start_time, json_filename, final_metrics_results, is_final_log=True
    )
    print(f".json 文件已更新并最终保存。")
    
    # 保存包含所有原始数据的 .pkl 文件 (只保存一次)
    valid_items = [item for item, _ in all_evaluated_items if item.total is not None and item.total > 0]
    pop_size = config.get('optimization.pop_size')
    final_pareto_front = nsga2_so_selection(valid_items, pop_size=pop_size) if valid_items else []

    results_data = {
        "exper_name": exper_name,
        "seed": seed,
        "config_dict": config.config,
        "all_mols": all_evaluated_items,
        "final_pareto_front": final_pareto_front,
        "running_time_hours": running_time_hours,
        "final_metrics": final_metrics_results # 将整个指标历史保存到pkl中
    }

    with open(pkl_filename, 'wb') as f:
        pickle.dump(results_data, f)
    print(f".pkl 文件已成功保存。")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行随机搜索基准测试。")
    parser.add_argument("--config", type=str, default="sacs_geo_jk/config.yaml", help="指向特定问题配置文件的路径。")
    parser.add_argument("--seed", type=int, default=42, help="用于可复现性的随机种子。")
    args = parser.parse_args()
    run_rs_baseline(config_path=args.config, seed=args.seed)

