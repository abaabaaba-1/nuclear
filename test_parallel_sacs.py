# test_sacs_evaluation_v2.py
import logging
import json
import os
from pathlib import Path
import shutil

# --- 确保从您的项目根目录正确导入组件 ---
try:
    from problem.sacs.evaluator import RewardingSystem
    from model.MOLLM import ConfigLoader
    from algorithm.base import ItemFactory
except ImportError as e:
    print(f"Import Error: {e}")
    print("\n[ERROR] 此脚本必须从您的 'MOLLM-main' 项目根目录运行。")
    print("请切换到 'MOLLM-main' 目录并运行: python test_sacs_evaluation_v2.py")
    exit()

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    """
    执行一次对SACS评估流程的单一、可控的测试。
    """
    logging.info("--- 开始 SACS 评估组件测试 V2 ---")

    # 1. 加载配置
    try:
        config_loader = ConfigLoader('sacs/config.yaml')
        config = config_loader.config
        logging.info("✅ 配置加载成功。")
    except Exception as e:
        logging.error(f"❌ 加载配置失败: {e}")
        return

    # 2. 定义一个简单的硬编码测试用例
    # 我们将修改单个节点的坐标
    test_case_json = {
        "new_code_blocks": {
            # 提供一个具体的、有效的修改作为测试
            "JOINT_201": "JOINT  201  -25.00 -38.50-165.00        -1"
        }
    }
    logging.info(f"🧪 创建测试用例: 修改 'JOINT_201'。")

    # 创建一个虚拟的 'Item' 对象，就像您的 MOO 算法所做的那样
    item_factory = ItemFactory(config.get('goals'))
    test_item = item_factory.create(json.dumps(test_case_json))

    # 3. 初始化 RewardingSystem
    try:
        reward_system = RewardingSystem(config=config_loader)
        logging.info("✅ RewardingSystem 初始化成功。")
    except Exception as e:
        logging.error(f"❌ 初始化 RewardingSystem 失败: {e}")
        return

    # 4. 执行评估
    logging.info("🚀 开始评估单个测试用例...")
    evaluated_items = []
    try:
        evaluated_items, _ = reward_system.evaluate([test_item])
        logging.info("✅ 评估方法调用完成。")
    except Exception as e:
        logging.error(f"❌ 在 'evaluate' 调用期间发生错误: {e}", exc_info=True)
        # 尝试清理临时目录
        if hasattr(reward_system, 'modifier') and hasattr(reward_system.modifier, 'temp_dir'):
             temp_dir = Path(reward_system.modifier.temp_dir)
             if temp_dir.exists():
                 shutil.rmtree(temp_dir)
                 logging.info(f"🧹 已清理临时目录: {temp_dir}")
        return

    # 5. 分析并报告结果
    logging.info("--- 测试结果 ---")
    if not evaluated_items:
        logging.error("❌ 评估没有返回任何项目。")
        return

    # --- 【已修正】---
    # 直接访问 .results 属性，而不是调用 .get_results() 方法
    final_results = evaluated_items[0].results
    # --- 修正结束 ---

    if not final_results:
        logging.error("❌ 评估后的项目没有附加结果。")
        return

    logging.info(f"原始结果字典: {json.dumps(final_results, indent=2)}")

    if final_results.get('error_reason'):
        logging.error(f"❌ 评估失败，原因: {final_results['error_reason']}")
    elif 'original_results' in final_results and final_results['original_results']:
        original = final_results['original_results']
        weight = original.get('weight', 'N/A')
        axial_uc = original.get('axial_uc_max', 'N/A')
        bending_uc = original.get('bending_uc_max', 'N/A')
        
        logging.info("✅ 成功: 评估流程完成并解析了结果。")
        logging.info(f"  - 重量 (Weight): {weight}")
        logging.info(f"  - 最大轴向应力比 (Max Axial UC): {axial_uc}")
        logging.info(f"  - 最大弯曲应力比 (Max Bending UC): {bending_uc}")
    else:
        logging.warning("🤔 评估已完成，但结果字典的格式与预期不符。")

    logging.info("--- 测试结束 ---")

if __name__ == "__main__":
    run_test()