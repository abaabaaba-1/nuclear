# read_checkpoint.py (V5 - 最终增强版)

import pickle
import pandas as pd
import os
import json
import numpy as np
from collections import defaultdict, deque

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    print("警告: 未安装 matplotlib 或 seaborn。将跳过数据可视化部分。")
    print("建议安装: pip install matplotlib seaborn pandas")
    PLOT_AVAILABLE = False

# 为了让脚本独立运行，即使在没有项目环境的情况下也能解包pkl
# 我们定义一个最小化的 Item 类结构
try:
    # 尝试从项目中导入，如果项目结构完整
    from algorithm.base import Item, HistoryBuffer
except ImportError:
    print("提示: 未在标准项目路径下找到 'algorithm.base'。将使用本地定义的最小化 Item 类。")
    class Item:
        def __init__(self):
            self.value = ""
            self.property = {}
            self.total = 0.0
    class HistoryBuffer: pass

try:
    from model.MOLLM import ConfigLoader
except ImportError:
    ConfigLoader = None

def analyze_final_pops(final_pops_data):
    """
    专门分析 'final_pops' 数据，找出最好和最差的解。
    """
    if not final_pops_data or not isinstance(final_pops_data, list):
        print("\n'final_pops' 数据为空或格式不正确。")
        return

    # 确保列表中的每个元素都是 Item 对象
    valid_pops = [p for p in final_pops_data if hasattr(p, 'total') and p.total is not None]

    if not valid_pops:
        print("\n在 'final_pops' 中没有找到有效的 Item 对象。")
        return

    print(f"\n--- 分析最终精英种群 (final_pops, 共 {len(valid_pops)} 个) ---")

    # 1. 找到 total 分数最高的 Item (最优解)
    best_item = max(valid_pops, key=lambda item: item.total)
    
    # 2. 找到 total 分数最低的 Item (最差解)
    worst_item = min(valid_pops, key=lambda item: item.total)

    print("\n[+] 精英种群中的最佳候选方案 (按 'total' 分数):")
    print(f"  - Total Score: {best_item.total:.6f}")
    if hasattr(best_item, 'property') and best_item.property:
        print(f"  - Properties: {best_item.property}")
    print(f"  - Candidate Value (JSON): \n{best_item.value}")

    print("\n[-] 精英种群中的最差候选方案 (按 'total' 分数):")
    print(f"  - Total Score: {worst_item.total:.6f}")
    if hasattr(worst_item, 'property') and worst_item.property:
        print(f"  - Properties: {worst_item.property}")
    print(f"  - Candidate Value (JSON): \n{worst_item.value}")


def analyze_coil_all_mols(all_candidates):
    if ConfigLoader is not None:
        try:
            cfg = ConfigLoader('stellarator_coil/config.yaml')
            nPhi = cfg.get('coil_design.wf_nPhi', 48)
            nTheta = cfg.get('coil_design.wf_nTheta', 50)
        except Exception:
            nPhi, nTheta = 48, 50
    else:
        nPhi, nTheta = 48, 50

    def _compute_connectivity(active_segments):
        if not active_segments:
            return None
        edges = []
        for seg_idx in active_segments:
            try:
                seg_idx = int(seg_idx)
            except Exception:
                continue
            if seg_idx < 0 or seg_idx >= nPhi * nTheta * 2:
                continue
            if seg_idx < nPhi * nTheta:
                phi = seg_idx // nTheta
                theta = seg_idx % nTheta
                phi2 = (phi + 1) % nPhi
                node_a = (phi, theta)
                node_b = (phi2, theta)
            else:
                offset = seg_idx - nPhi * nTheta
                phi = offset // nTheta
                theta = offset % nTheta
                theta2 = (theta + 1) % nTheta
                node_a = (phi, theta)
                node_b = (phi, theta2)
            edges.append((node_a, node_b))

        if not edges:
            return None

        deg = defaultdict(int)
        adj = defaultdict(list)
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
            adj[a].append(b)
            adj[b].append(a)

        visited = set()
        comp_edges = []
        comp_open_ends = []
        for start in deg.keys():
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            nodes = []
            while stack:
                u = stack.pop()
                nodes.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            edges_in_comp = int(sum(deg[u] for u in nodes) / 2)
            open_ends = sum(1 for u in nodes if deg[u] == 1)
            comp_edges.append(edges_in_comp)
            comp_open_ends.append(open_ends)

        total_edges = len(edges)
        n_components = len(comp_edges)
        largest_comp_edges = max(comp_edges) if comp_edges else 0
        total_open_ends = sum(comp_open_ends)
        has_open_ends = total_open_ends > 0
        frac_largest = largest_comp_edges / total_edges if total_edges > 0 else 0.0
        return {
            'edges_total': total_edges,
            'n_components': n_components,
            'largest_comp_edges': largest_comp_edges,
            'fraction_in_largest': frac_largest,
            'open_ends': total_open_ends,
            'has_open_ends': has_open_ends,
        }

    all_rows = []
    conn_rows = []
    for candidate_entry in all_candidates:
        item = candidate_entry[0] if isinstance(candidate_entry, (list, tuple)) and candidate_entry else candidate_entry
        if not hasattr(item, 'property') or not hasattr(item, 'total') or not hasattr(item, 'value'):
            continue
        prop = item.property or {}
        if not {'f_B', 'f_S', 'I_max'}.issubset(set(prop.keys())):
            continue
        all_rows.append({
            'f_B': prop.get('f_B'),
            'f_S': prop.get('f_S'),
            'I_max': prop.get('I_max'),
            'total_score': item.total,
        })

        try:
            cfg = json.loads(item.value)
            active_segments = cfg.get('active_segments', [])
            metrics = _compute_connectivity(active_segments)
            if metrics is not None:
                conn_rows.append(metrics)
        except Exception:
            continue

    if not all_rows:
        print("\n未能从 'all_mols' 中提取到任何仿星器线圈 (f_B, f_S, I_max) 数据。")
        return

    df = pd.DataFrame(all_rows)
    print("\n--- Stellarator Coil 目标统计 (f_B, f_S, I_max, total_score) ---")
    with pd.option_context('display.float_format', lambda x: f"{x:.3e}"):
        print(df[['f_B', 'f_S', 'I_max', 'total_score']].agg(['mean', 'std', 'min', 'max']))

    if conn_rows:
        dfc = pd.DataFrame(conn_rows)
        print("\n--- Stellarator Coil 连通性统计 (基于 active_segments 拓扑) ---")
        print(dfc[['edges_total', 'n_components', 'fraction_in_largest', 'open_ends']].agg(['mean', 'min', 'max']))
        closed_frac = 1.0 - dfc['has_open_ends'].mean()
        print(f"  无开口链条的配置比例 (all_mols): {closed_frac:.3f}")

        N = min(100, len(dfc) // 2)
        if N > 0:
            early = dfc.head(N)
            late = dfc.tail(N)
            early_closed = 1.0 - early['has_open_ends'].mean()
            late_closed = 1.0 - late['has_open_ends'].mean()
            print(f"  早期 {N} 个样本中无开口比例: {early_closed:.3f}")
            print(f"  后期 {N} 个样本中无开口比例: {late_closed:.3f}")

    if PLOT_AVAILABLE and not df.empty:
        print("\n--- 正在生成 Stellarator Coil 目标分布图... ---")
        plt.style.use('seaborn-v0_8-whitegrid')
        df_plot = df.copy()
        df_plot['log10_f_B'] = np.log10(df_plot['f_B'].clip(lower=1e-20))
        g = sns.pairplot(
            df_plot,
            vars=['log10_f_B', 'f_S', 'I_max'],
            diag_kind='hist',
            plot_kws={'alpha': 0.6, 's': 30},
            height=3
        )
        g.fig.suptitle('Stellarator Coil Objective Distribution', y=1.02, fontsize=16)
        save_path = 'coil_checkpoint_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Coil 可视化图表已保存至: {os.path.abspath(save_path)}")


def analyze_checkpoint(filepath):
    """
    主分析函数，读取 pkl 文件并进行全面分析。
    """
    if not os.path.exists(filepath):
        print(f"错误: 文件未找到于 '{filepath}'")
        return

    print(f"--- 正在读取文件: {filepath} ---")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"读取或解包 pkl 文件时出错: {e}")
        return

    print("\n文件包含以下主要部分 (keys):")
    print(list(data.keys()))
    
    # ===================================================================
    # 新增模块：分析 'final_pops'
    if 'final_pops' in data:
        analyze_final_pops(data['final_pops'])
    else:
        print("\n文件中未找到 'final_pops' 键。")
    # ===================================================================

    if 'all_mols' not in data or not data['all_mols']:
        print("\n文件中没有找到 'all_mols' 数据，跳过历史数据分析。")
        return
        
    all_candidates = data['all_mols']

    sample_entry = all_candidates[0] if all_candidates else None
    sample_item = sample_entry[0] if isinstance(sample_entry, (list, tuple)) and sample_entry else sample_entry
    if hasattr(sample_item, 'property') and isinstance(sample_item.property, dict):
        keys = set(sample_item.property.keys())
        if {'f_B', 'f_S', 'I_max'}.issubset(keys):
            analyze_coil_all_mols(all_candidates)
            return

    print(f"\n--- 分析所有历史评估数据 (all_mols, 共 {len(all_candidates)} 条记录) ---")

    extracted_data = []
    for candidate_entry in all_candidates:
        item = candidate_entry[0] if isinstance(candidate_entry, (list, tuple)) and candidate_entry else candidate_entry
        if not hasattr(item, 'value') or not hasattr(item, 'property'): continue

        prop = item.property or {}
        info = {'candidate_string': item.value, 'total_score': item.total}
        
        # 智能解析新旧两种数据格式
        if 'original_results' in prop: # 新格式
            original_results = prop.get('original_results', {})
            constraint_results = prop.get('constraint_results', {})
            info.update({
                'weight': original_results.get('weight'),
                'axial_uc_max': original_results.get('axial_uc_max'),
                'bending_uc_max': original_results.get('bending_uc_max'),
                'is_feasible': constraint_results.get('is_feasible'),
                'max_uc': constraint_results.get('max_uc'),
            })
        else: # 旧格式
            info.update({
                'weight': prop.get('weight'),
                'axial_uc_max': prop.get('axial_uc_max'),
                'bending_uc_max': prop.get('bending_uc_max'),
            })
            max_uc = None
            if info['axial_uc_max'] is not None and info['bending_uc_max'] is not None and info['axial_uc_max'] < 100:
                max_uc = max(info['axial_uc_max'], info['bending_uc_max'])
            info['max_uc'] = max_uc
            info['is_feasible'] = 1.0 if max_uc is not None and max_uc <= 1.0 else 0.0
            
        extracted_data.append(info)

    if not extracted_data:
        print("\n未能从 'all_mols' 中成功提取任何数据。")
        return

    df = pd.DataFrame(extracted_data)
    df.dropna(subset=['weight', 'axial_uc_max', 'bending_uc_max'], how='all', inplace=True)
    
    print(f"\n种群整体统计 (共 {len(df)} 条有效记录):")
    print(df[['weight', 'axial_uc_max', 'bending_uc_max', 'max_uc']].agg(['mean', 'std', 'min', 'max']).round(4))
    
    df_feasible = df[df['is_feasible'] == 1.0].copy()
    
    if df_feasible.empty:
        print("\n警告: 在所有候选方案中，没有找到任何可行解 (UC <= 1.0)。")
    else:
        print(f"\n恭喜！在历史记录中找到 {len(df_feasible)} 个可行解。")
        print("\n--- 各个单项最优的可行候选方案 (从全部历史记录中筛选) ---")
        for col in ['weight', 'axial_uc_max', 'bending_uc_max']:
            if col in df_feasible.columns and not df_feasible[col].empty:
                best_row = df_feasible.loc[df_feasible[col].idxmin()]
                print(f"\n- 目标 '{col}' 的最优解 (最小值):")
                # 打印数值部分
                print(best_row[['weight', 'axial_uc_max', 'bending_uc_max', 'max_uc']].to_string())
                # 新增：打印完整的 JSON 定义
                print("- Candidate Value (JSON):")
                print(best_row['candidate_string'])
    if PLOT_AVAILABLE and not df.empty:
        # (可视化部分代码不变)
        print("\n--- 正在生成数据分布图... ---")
        plt.style.use('seaborn-v0_8-whitegrid')
        df['Feasibility'] = df['is_feasible'].apply(lambda x: 'Feasible (UC <= 1)' if x == 1.0 else 'Infeasible (UC > 1 or Fail)')
        g = sns.pairplot(
            df.query('max_uc < 10 and weight < 500'),
            vars=['weight', 'axial_uc_max', 'bending_uc_max'],
            hue='Feasibility', palette={'Feasible (UC <= 1)': 'green', 'Infeasible (UC > 1 or Fail)': 'red'},
            diag_kind='hist', plot_kws={'alpha': 0.6, 's': 30}, height=3
        )
        g.fig.suptitle('种群目标分布 (可行 vs 不可行)', y=1.02, fontsize=16)
        save_path = 'checkpoint_analysis_final.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化图表已保存至: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze MOLLM checkpoint pickle file.')
    parser.add_argument('--path', type=str, default=os.path.join('moo_results', 'check.pkl'), 
                        help='Path to the .pkl checkpoint file')
    args = parser.parse_args()
    
    analyze_checkpoint(args.path)