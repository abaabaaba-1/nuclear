#!/usr/bin/env python3
"""
目标函数范围校准工具

使用方法:
    python problem/stellarator_coil_gsco_lite/calibrate_objectives.py

功能:
1. 生成100个随机配置
2. 评估所有配置
3. 统计目标函数的实际分布
4. 推荐合理的 objective_ranges 配置
"""

import sys
import os
import json
import numpy as np
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.MOLLM import ConfigLoader
from problem.stellarator_coil_gsco_lite.evaluator import SimpleGSCOEvaluator, generate_initial_population


class DummyItem:
    """临时的Item类用于评估"""
    def __init__(self, value):
        self.value = value
        self.property = {}
        self.total = 0.0
    
    def assign_results(self, results):
        self.property = results['original_results']
        self.total = results['overall_score']


def calibrate_objectives():
    """校准目标函数范围"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("Calibrator")
    
    # 加载配置
    config_path = "stellarator_coil_gsco_lite/config.yaml"  # ConfigLoader会自动添加problem/前缀
    config = ConfigLoader(config_path)
    
    logger.info("="*70)
    logger.info("Objective Function Calibration Tool")
    logger.info("="*70)
    
    # 创建evaluator
    logger.info("Initializing evaluator...")
    evaluator = SimpleGSCOEvaluator(config)
    
    # 生成随机样本
    logger.info("Generating 100 random configurations...")
    n_samples = 100
    random_configs = generate_initial_population(config, seed=42)[:n_samples]
    
    # 转换为Item对象
    items = [DummyItem(cfg) for cfg in random_configs]
    
    # 评估
    logger.info("Evaluating configurations (this may take a while)...")
    evaluated_items, log_dict = evaluator.evaluate(items)
    
    # 收集结果
    f_B_values = [item.property['f_B'] for item in evaluated_items]
    f_S_values = [item.property['f_S'] for item in evaluated_items]
    I_max_values = [item.property['I_max'] for item in evaluated_items]
    
    # 统计分析
    logger.info("\n" + "="*70)
    logger.info("Calibration Results")
    logger.info("="*70)
    
    def print_stats(name, values):
        values = np.array(values)
        logger.info(f"\n{name}:")
        logger.info(f"  Min:  {np.min(values):.4e}")
        logger.info(f"  Max:  {np.max(values):.4e}")
        logger.info(f"  Mean: {np.mean(values):.4e}")
        logger.info(f"  Std:  {np.std(values):.4e}")
        logger.info(f"  5th percentile:  {np.percentile(values, 5):.4e}")
        logger.info(f"  95th percentile: {np.percentile(values, 95):.4e}")
        return values
    
    f_B_arr = print_stats("f_B (Magnetic Field Error) [T²m²]", f_B_values)
    f_S_arr = print_stats("f_S (Number of Active Cells)", f_S_values)
    I_max_arr = print_stats("I_max (Maximum Current) [MA]", I_max_values)
    
    # 推荐范围（使用5th-95th百分位数）
    logger.info("\n" + "="*70)
    logger.info("Recommended objective_ranges for config.yaml:")
    logger.info("="*70)
    
    recommended_ranges = {
        'f_B': [float(np.percentile(f_B_arr, 5)), float(np.percentile(f_B_arr, 95))],
        'f_S': [int(np.percentile(f_S_arr, 5)), int(np.percentile(f_S_arr, 95))],
        'I_max': [float(np.percentile(I_max_arr, 5)), float(np.percentile(I_max_arr, 95))]
    }
    
    logger.info("\nobjective_ranges:")
    for obj, (min_val, max_val) in recommended_ranges.items():
        if obj == 'f_B':
            logger.info(f"  {obj}: [{min_val:.2e}, {max_val:.2e}]")
        elif obj == 'f_S':
            logger.info(f"  {obj}: [{min_val}, {max_val}]")
        else:
            logger.info(f"  {obj}: [{min_val:.2f}, {max_val:.2f}]")
    
    # 保存到文件
    output_file = project_root / "problem" / "stellarator_coil_gsco_lite" / "calibrated_ranges.json"
    with open(output_file, 'w') as f:
        json.dump(recommended_ranges, f, indent=2)
    
    logger.info(f"\nCalibrated ranges saved to: {output_file}")
    logger.info("\nPlease update config.yaml with these values!")
    
    return recommended_ranges


if __name__ == "__main__":
    calibrate_objectives()
