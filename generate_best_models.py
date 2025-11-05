# generate_best_models.py

import os
import pickle
import pandas as pd
import json
import shutil
import logging

# --- 1. 配置区域 ---
# 请根据您的实际情况修改以下三个路径变量

# 指向您想要分析的 pkl 文件
PKL_FILE_PATH = os.path.join('moo_results', 'check.pkl')

# 指向您的SACS项目文件夹 (包含 sacinp.demo06 文件的目录)
SACS_PROJECT_PATH = "/mnt/d/wsl_sacs_exchange/demo06_project/Demo06"

# 指定生成的模型文件要保存到的目录
OUTPUT_DIR = "best_models_output"

# --- 脚本依赖 ---
# 确保此脚本可以找到 problem.sacs.sacs_file_modifier
try:
    from problem.sacs.sacs_file_modifier import SacsFileModifier
except ImportError:
    print("错误: 无法导入 SacsFileModifier。")
    print("请确保此脚本与您的项目结构保持一致，或者将 problem 文件夹的路径添加到 PYTHONPATH。")
    exit()

# 为了让脚本独立运行，定义最小化的 Item 类结构
try:
    from algorithm.base import Item, HistoryBuffer
except ImportError:
    class Item: pass
    class HistoryBuffer: pass

# 设置日志记录，以便看到 SacsFileModifier 的详细输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    """
    主函数：读取 pkl, 查找最优解, 并生成对应的 SACS 模型文件。
    """
    print("="*50)
    print("=== SACS 最优模型文件生成脚本 ===")
    print("="*50)

    # --- 步骤 1: 验证路径和文件 ---
    if not os.path.exists(PKL_FILE_PATH):
        print(f"错误: PKL 文件未找到于 '{PKL_FILE_PATH}'")
        return

    if not os.path.exists(SACS_PROJECT_PATH):
        print(f"错误: SACS 项目路径不存在于 '{SACS_PROJECT_PATH}'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输入 PKL 文件: {PKL_FILE_PATH}")
    print(f"SACS 项目路径: {SACS_PROJECT_PATH}")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")

    # --- 步骤 2: 读取并解析 PKL 文件 ---
    print("\n--- 正在读取并解析 PKL 文件... ---")
    try:
        with open(PKL_FILE_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"读取或解包 pkl 文件时出错: {e}")
        return

    if 'all_mols' not in data or not data['all_mols']:
        print("错误: 文件中没有找到 'all_mols' 数据。脚本终止。")
        return

    all_candidates = data['all_mols']
    extracted_data = []
    for entry in all_candidates:
        item = entry[0] if isinstance(entry, (list, tuple)) else entry
        if not (hasattr(item, 'value') and hasattr(item, 'property') and item.property):
            continue
        
        prop = item.property
        info = {'candidate_string': item.value}
        
        # 兼容新旧两种数据格式
        if 'original_results' in prop:
            info.update(prop.get('original_results', {}))
            info.update(prop.get('constraint_results', {}))
        else: # 旧格式兼容
            info.update(prop)
            max_uc = None
            if info.get('axial_uc_max') is not None and info.get('bending_uc_max') is not None and info.get('axial_uc_max', 999) < 100:
                max_uc = max(info['axial_uc_max'], info['bending_uc_max'])
            info['max_uc'] = max_uc
            info['is_feasible'] = 1.0 if max_uc is not None and max_uc <= 1.0 else 0.0
            
        extracted_data.append(info)

    df = pd.DataFrame(extracted_data)
    df.dropna(subset=['weight', 'axial_uc_max', 'bending_uc_max'], how='all', inplace=True)
    print(f"成功解析 {len(df)} 条有效候选方案。")
    
    # --- 步骤 3: 筛选可行解并寻找单项最优 ---
    df_feasible = df[df['is_feasible'] == 1.0].copy()
    if df_feasible.empty:
        print("\n错误: 在所有候选方案中未找到任何可行解 (UC <= 1.0)。无法生成模型文件。")
        return
        
    print(f"\n--- 从 {len(df_feasible)} 个可行解中寻找单项最优方案... ---")

    objectives_to_find = {
        'weight': 'min',
        'axial_uc_max': 'min',
        'bending_uc_max': 'min'
    }
    best_solutions = {}

    for obj, mode in objectives_to_find.items():
        if obj in df_feasible.columns:
            best_idx = df_feasible[obj].idxmin() if mode == 'min' else df_feasible[obj].idxmax()
            best_solutions[obj] = df_feasible.loc[best_idx]
            print(f"找到 '{obj}' 最优解: {best_solutions[obj][obj]:.4f}")

    # --- 步骤 4: 生成模型文件 ---
    print("\n--- 正在生成 SACS 模型文件... ---")
    modifier = SacsFileModifier(SACS_PROJECT_PATH)
    
    # 创建一个主备份，用于每次操作前恢复，确保独立性
    master_backup_path = modifier._create_backup()
    if not master_backup_path:
        print("\n错误: 创建主备份文件失败，为安全起见，脚本终止。")
        return
    
    print(f"已创建主备份文件: {master_backup_path.name}")

    try:
        for obj, best_row in best_solutions.items():
            print(f"\n正在处理 '{obj}' 的最优方案...")
            
            # 1. 恢复到原始状态
            modifier._restore_from_backup(master_backup_path)
            
            # 2. 获取并解析JSON
            candidate_json_str = best_row['candidate_string']
            try:
                new_code_blocks = json.loads(candidate_json_str).get("new_code_blocks")
                if not new_code_blocks:
                    raise ValueError("JSON中缺少 'new_code_blocks' 键")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  -> 错误: 解析候选方案的JSON失败: {e}。跳过此方案。")
                continue
            
            # 3. 应用修改
            success = modifier.replace_code_blocks(new_code_blocks)
            
            # 4. 如果成功，将修改后的文件复制到输出目录
            if success:
                output_filename = f"sacinp_best_{obj}.demo06"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                # 从SACS项目目录复制已修改的文件到输出目录
                shutil.copy2(modifier.input_file, output_path)
                print(f"  -> ✅ 成功生成模型文件: {os.path.abspath(output_path)}")
            else:
                print(f"  -> ❌ 修改 SACS 文件失败。请检查日志。")

    finally:
        # 5. 无论成功与否，最后都将原始文件恢复，保持项目清洁
        modifier._restore_from_backup(master_backup_path)
        print("\n--- 操作完成，已将原始 SACS 文件恢复到初始状态。 ---")
        
if __name__ == '__main__':
    main()