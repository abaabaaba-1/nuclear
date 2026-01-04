#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load candidate solutions from a MOLLM checkpoint (PKL), pick the best few for
each metric, regenerate VmecInput based on the baseline input.w7x, run vmecpp,
and plot the plasma boundary. Save PNG figures plus an optional CSV summary.

Usage:

python problem/stellarator_vmec/plot_vmec_checkpoint.py \
  --pkl moo_results/zgca,gemini-2.5-flash-nothinking/mols/volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_42.pkl \
  --baseline problem/stellarator_vmec/vmecpp/calculations/input.w7x \
  --outdir moo_results/plots \
  --topk 3 \
  --compare_baseline

Dependencies: vmecpp, matplotlib, numpy, pandas (optional summary)
"""
import argparse
import os
import pickle
import json
import sys
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# 添加项目根目录到路径，以便导入项目模块
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

import vmecpp

# 支持作为模块导入或直接运行
try:
    from .vmec_file_modifier import VmecFileModifier
except ImportError:
    from vmec_file_modifier import VmecFileModifier

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pkl(pkl_path: str) -> Any:
    """Load a pkl file while handling encoding or import issues."""
    _project_root = Path(pkl_path).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    
    with open(pkl_path, 'rb') as f:
        try:
            return pickle.load(f)
        except (UnicodeDecodeError, ModuleNotFoundError) as e:
            logger.warning(f"Failed to load with default encoding: {e}; trying latin1")
            try:
                f.seek(0)
                return pickle.load(f, encoding='latin1')
            except Exception as e2:
                logger.error(f"Failed to load with latin1 encoding: {e2}")
                raise


def iter_items_from_data(data: Any) -> List[Any]:
    """Extract candidate Items from a PKL structure as robustly as possible."""
    items: List[Any] = []
    # 常见键位
    for key in ['final_pops', 'init_pops', 'all_mols', 'history', 'results', 'population']:
        v = data.get(key) if isinstance(data, dict) else None
        if isinstance(v, list) and len(v) > 0:
            # all_mols 可能是 [ [Item, ...], ... ]
            for entry in v:
                if isinstance(entry, (list, tuple)) and entry:
                    cand = entry[0]
                    if hasattr(cand, 'value'):
                        items.append(cand)
                else:
                    if hasattr(entry, 'value'):
                        items.append(entry)
    # 兜底：如果顶层就是 list
    if not items and isinstance(data, list):
        for entry in data:
            if hasattr(entry, 'value'):
                items.append(entry)
    # 去重
    uniq = []
    seen = set()
    for it in items:
        sig = getattr(it, 'value', None)
        if sig and sig not in seen:
            uniq.append(it)
            seen.add(sig)
    return uniq


def parse_candidate_coeffs(item) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Parse new_coefficients and metric dictionaries from an Item."""
    coeffs: Dict[str, float] = {}
    metrics: Dict[str, Any] = {}

    # 先从 value(JSON) 解析 new_coefficients
    raw = getattr(item, 'value', '') or ''
    try:
        if '<candidate>' in raw:
            raw = raw.split('<candidate>', 1)[1].rsplit('</candidate>', 1)[0].strip()
        obj = json.loads(raw)
        if isinstance(obj, dict) and 'new_coefficients' in obj and isinstance(obj['new_coefficients'], dict):
            coeffs = {k: float(v) for k, v in obj['new_coefficients'].items()}
    except Exception as e:
        logger.debug(f"Failed to parse candidate coefficients: {e}")

    # 再从 item.property 中解析指标
    prop = getattr(item, 'property', {}) or {}
    if 'original_results' in prop:
        metrics.update(prop.get('original_results', {}))
        cr = prop.get('constraint_results', {}) or {}
        metrics['is_feasible'] = cr.get('is_feasible', 1.0)
    else:
        # 旧格式兜底
        for key in ['volume', 'aspect_ratio', 'magnetic_shear']:
            if key in prop:
                metrics[key] = prop[key]
        metrics['is_feasible'] = prop.get('is_feasible', 1.0)

    # 额外：若 item.total 存在，记录
    if hasattr(item, 'total') and getattr(item, 'total') is not None:
        metrics['total'] = float(getattr(item, 'total'))

    return coeffs, metrics


def pick_best_by_metrics(items: List[Any], require_feasible: bool = True, topk: int = 1):
    """Select the top-k solutions for each metric."""
    parsed: List[Tuple[Any, Dict[str, Any], Dict[str, float]]] = []
    for it in items:
        coeffs, metrics = parse_candidate_coeffs(it)
        if not coeffs:
            continue
        if require_feasible:
            feasible_flag = metrics.get('is_feasible', 1.0)
            try:
                if float(feasible_flag) <= 0.0:
                    continue
            except Exception:
                pass
        parsed.append((it, metrics, coeffs))

    def top_by(metric: str, reverse: bool) -> List[Tuple[Any, Dict[str, Any], Dict[str, float]]]:
        ranked = [x for x in parsed if metric in x[1]]
        ranked.sort(key=lambda x: float(x[1][metric]), reverse=reverse)
        return ranked[:topk]

    result = {
        'volume': top_by('volume', True),
        'aspect_ratio': top_by('aspect_ratio', False),
        'magnetic_shear': top_by('magnetic_shear', True)
    }
    return result


def get_baseline_coeffs(baseline_input: Path) -> Dict[str, float]:
    """Extract coefficients from the baseline input file."""
    modifier = VmecFileModifier(str(baseline_input.parent), baseline_input.name)
    return modifier.extract_coefficients()


def run_vmec_safe(
    baseline_input: Path,
    coeffs: Dict[str, float],
    tmp_dir: Path,
) -> Optional[Any]:
    """Run VMEC safely and return vmec_output, or None if it fails."""
    local_input = None
    try:
        import shutil
        # Ensure temp directory exists
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local_input = tmp_dir / baseline_input.name

        # Copy baseline file
        if not baseline_input.exists():
            logger.error(f"Baseline file does not exist: {baseline_input}")
            return None
        
        logger.debug(f"Copying file: {baseline_input} -> {local_input}")
        shutil.copy2(baseline_input, local_input)
        
        # Verify copy succeeded
        if not local_input.exists():
            logger.error(f"File copy failed: {local_input}")
            return None
        
        logger.debug(f"Creating VmecFileModifier: project_path={tmp_dir}, input_file={baseline_input.name}")
        modifier = VmecFileModifier(str(tmp_dir), baseline_input.name)
        
        # Filter coefficients to those present in baseline
        before_coeffs = modifier.extract_coefficients()
        existing_keys = set(before_coeffs.keys())
        filtered_coeffs: Dict[str, float] = {}
        for k, v in coeffs.items():
            k_norm = k.strip().replace(" ", "")
            if k_norm in existing_keys:
                filtered_coeffs[k_norm] = v

        if not filtered_coeffs:
            logger.warning("No valid coefficients can be replaced")
            return None

        logger.debug(f"Replacing {len(filtered_coeffs)} coefficients")
        modifier.replace_coefficients(filtered_coeffs)
        
        # Run VMEC using absolute path
        local_input_abs = local_input.resolve()
        logger.debug(f"Running VMEC: {local_input_abs}")
        if not local_input_abs.exists():
            logger.error(f"VMEC input file does not exist: {local_input_abs}")
            return None
        vmec_input = vmecpp.VmecInput.from_file(str(local_input_abs))
        vmec_output = vmecpp.run(vmec_input)
        logger.debug("VMEC run succeeded")
        return vmec_output
    except RuntimeError as e:
        error_msg = str(e)
        if "JACOBIAN" in error_msg or "FATAL ERROR" in error_msg:
            logger.warning(f"VMEC run failed (physically invalid): {error_msg[:100]}")
        else:
            logger.warning(f"VMEC run failed: {error_msg[:100]}")
        return None
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info(f"Debug - tmp_dir: {tmp_dir} (exists: {tmp_dir.exists()})")
        logger.info(f"Debug - local_input: {local_input} (exists: {local_input.exists() if local_input else 'N/A'})")
        logger.info(f"Debug - baseline_input: {baseline_input} (exists: {baseline_input.exists()})")
        return None
    except Exception as e:
        logger.warning(f"VMEC raised an exception: {type(e).__name__}: {str(e)[:200]}")
        logger.info(f"Debug - tmp_dir: {tmp_dir}, local_input: {local_input}, baseline_input: {baseline_input}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def plot_plasma_geometry(
    vmec_output: Any,
    out_png: Path,
    surface_fracs: List[float] = None,
    title_suffix: str = "",
    compare_with: Optional[Any] = None,
    compare_label: str = "Baseline",
    metrics: Optional[Dict[str, Any]] = None,
):
    """Plot the plasma geometry, optionally comparing against another design."""
    if vmec_output is None:
        logger.error(f"Cannot plot {out_png.stem}: vmec_output is None")
        return False

    ns = vmec_output.wout.ns
    xm = vmec_output.wout.xm
    xn = vmec_output.wout.xn
    rmnc = vmec_output.wout.rmnc
    zmns = vmec_output.wout.zmns

    num_theta = 161
    num_phi = 241
    grid_theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=True)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, num_phi, endpoint=True)

    # 确定要绘制的磁面
    if surface_fracs is None or not surface_fracs:
        surface_fracs = [1.0]
    surface_indices = []
    for frac in surface_fracs:
        f = min(max(float(frac), 0.0), 1.0)
        idx = int(round(f * (ns - 1)))
        if idx not in surface_indices:
            surface_indices.append(idx)
    surface_indices.sort()

    # 创建图形 - 总是创建对比视图以突出差异
    if compare_with is not None:
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 0.8])
        ax3d_main = fig.add_subplot(gs[0, 0], projection='3d')
        ax3d_compare = fig.add_subplot(gs[0, 1], projection='3d')
        ax2d = None  # 关闭重复的截面图
        ax_diff = fig.add_subplot(gs[1, 0])  # 差异放大视图
        ax_multi = fig.add_subplot(gs[1, 1])  # 多截面视图
        ax_info = fig.add_subplot(gs[:, 2])  # 信息面板跨越上下两行
        ax_info.axis('off')
    else:
        fig = plt.figure(figsize=(12, 5.5))
        ax3d_main = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)
        ax_diff = None
        ax_multi = None
        ax_info = None
        ax3d_compare = None

    cmap = plt.get_cmap("viridis", len(surface_indices) + 2)
    phi_idx = 0  # phi=0 截面

    # 存储主设计的截面数据用于差异计算
    main_cross_sections = {}
    main_3d_data = {}  # 存储主设计的3D数据用于多截面视图

    # 绘制主设计
    for order, j in enumerate(surface_indices):
        color = cmap(order + 1)
        x = np.zeros([num_theta, num_phi])
        y = np.zeros([num_theta, num_phi])
        z = np.zeros([num_theta, num_phi])
        for idx_t, theta in enumerate(grid_theta):
            kernel_base = xm * theta
            for idx_p, phi in enumerate(grid_phi):
                kernel = kernel_base - xn * phi
                r = np.dot(rmnc[:, j], np.cos(kernel))
                x[idx_t, idx_p] = r * np.cos(phi)
                y[idx_t, idx_p] = r * np.sin(phi)
                z[idx_t, idx_p] = np.dot(zmns[:, j], np.sin(kernel))

        main_3d_data[j] = (x, y, z)  # 保存用于多截面视图

        ax3d_main.plot_surface(
            x, y, z,
            linewidth=0,
            antialiased=True,
            alpha=0.6 if len(surface_indices) > 1 else 0.9,
            color=color,
        )

        # 绘制截面
        r_cross = np.sqrt(x[:, phi_idx] ** 2 + y[:, phi_idx] ** 2)
        z_cross = z[:, phi_idx]
        main_cross_sections[j] = (r_cross, z_cross)
        label = f"Optimized s={j/(ns-1):.2f}" if ns > 1 else "Optimized s=0.0"
        if ax2d is not None:
            ax2d.plot(r_cross, z_cross, color=color, label=label, linewidth=2.5)
        
        # 多截面视图 - 绘制主设计的多个截面
        if ax_multi is not None:
            for phi_idx_multi in [0, num_phi//4, num_phi//2, 3*num_phi//4]:
                r_multi = np.sqrt(x[:, phi_idx_multi] ** 2 + y[:, phi_idx_multi] ** 2)
                z_multi = z[:, phi_idx_multi]
                ax_multi.plot(
                    r_multi,
                    z_multi,
                    color=color,
                    linestyle='-',
                    alpha=0.6,
                    linewidth=1.5,
                    label=f"Optimized phi={phi_idx_multi*360/num_phi:.0f}deg" if order == 0 and phi_idx_multi == 0 else "",
                )

    # 如果有对比设计，也绘制
    compare_cross_sections = {}
    if compare_with is not None:
        ns_comp = compare_with.wout.ns
        xm_comp = compare_with.wout.xm
        xn_comp = compare_with.wout.xn
        rmnc_comp = compare_with.wout.rmnc
        zmns_comp = compare_with.wout.zmns

        for order, j in enumerate(surface_indices):
            j_comp = j if j < ns_comp else ns_comp - 1
            color = cmap(order + 1)
            x = np.zeros([num_theta, num_phi])
            y = np.zeros([num_theta, num_phi])
            z = np.zeros([num_theta, num_phi])
            for idx_t, theta in enumerate(grid_theta):
                kernel_base = xm_comp * theta
                for idx_p, phi in enumerate(grid_phi):
                    kernel = kernel_base - xn_comp * phi
                    r = np.dot(rmnc_comp[:, j_comp], np.cos(kernel))
                    x[idx_t, idx_p] = r * np.cos(phi)
                    y[idx_t, idx_p] = r * np.sin(phi)
                    z[idx_t, idx_p] = np.dot(zmns_comp[:, j_comp], np.sin(kernel))

            ax3d_compare.plot_surface(
                x, y, z,
                linewidth=0,
                antialiased=True,
                alpha=0.6 if len(surface_indices) > 1 else 0.9,
                color=color,
            )

            # 在同一个2D图上绘制对比
            r_cross = np.sqrt(x[:, phi_idx] ** 2 + y[:, phi_idx] ** 2)
            z_cross = z[:, phi_idx]
            compare_cross_sections[j] = (r_cross, z_cross)
            label_comp = f"{compare_label} s={j_comp/(ns_comp-1):.2f}" if ns_comp > 1 else f"{compare_label} s=0.0"
            if ax2d is not None:
                ax2d.plot(r_cross, z_cross, color=color, linestyle='--', label=label_comp, linewidth=2.5, alpha=0.8)
            
            # 差异放大视图
            if ax_diff is not None and j in main_cross_sections:
                r_main, z_main = main_cross_sections[j]
                r_comp, z_comp = compare_cross_sections[j]
                # 计算差异
                r_diff = r_main - r_comp
                z_diff = z_main - z_comp
                # 放大显示差异（放大10倍）
                ax_diff.plot(
                    r_main,
                    z_main + z_diff * 10,
                    color=color,
                    label=f"Optimized (diff x10) s={j/(ns-1):.2f}",
                    linewidth=2,
                )
                ax_diff.plot(r_comp, z_comp, color=color, linestyle='--', label=f"{compare_label} s={j_comp/(ns_comp-1):.2f}", linewidth=2, alpha=0.8)
            
            # 多截面视图（不同phi角度）- 只绘制baseline的多个截面
            if ax_multi is not None:
                for phi_idx_multi in [0, num_phi//4, num_phi//2, 3*num_phi//4]:
                    r_multi = np.sqrt(x[:, phi_idx_multi] ** 2 + y[:, phi_idx_multi] ** 2)
                    z_multi = z[:, phi_idx_multi]
                    ax_multi.plot(
                        r_multi,
                        z_multi,
                        color=color,
                        alpha=0.5,
                        linewidth=1.5,
                        label=f"{compare_label} phi={phi_idx_multi*360/num_phi:.0f}deg" if order == 0 and phi_idx_multi == 0 else "",
                    )

        ax3d_compare.set_aspect('auto')
        ax3d_compare.set_title(f"{compare_label} (3D)")
        ax3d_compare.set_xlabel("X [m]")
        ax3d_compare.set_ylabel("Y [m]")
        ax3d_compare.set_zlabel("Z [m]")

    # 设置标签和标题
    ax3d_main.set_aspect('auto')
    title_suffix_str = f" | {title_suffix}" if title_suffix else ""
    ax3d_main.set_title(f"Optimized{title_suffix_str} (3D)")
    ax3d_main.set_xlabel("X [m]")
    ax3d_main.set_ylabel("Y [m]")
    ax3d_main.set_zlabel("Z [m]")

    if ax2d is not None:
        ax2d.set_title(f"Cross-section comparison (phi = 0){title_suffix_str}", fontsize=11)
        ax2d.set_xlabel("R [m]")
        ax2d.set_ylabel("Z [m]")
        ax2d.grid(True, linestyle='--', alpha=0.5)
        ax2d.legend(loc='best', fontsize=8)
        ax2d.set_aspect('equal')
    
    # 设置差异放大视图
    if ax_diff is not None:
        ax_diff.set_title("Amplified difference view (Z delta x10)", fontsize=11)
        ax_diff.set_xlabel("R [m]")
        ax_diff.set_ylabel("Z [m]")
        ax_diff.grid(True, linestyle='--', alpha=0.5)
        ax_diff.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        ax_diff.set_aspect('equal')
    
    # 设置多截面视图
    if ax_multi is not None:
        ax_multi.set_title("Multi-angle cross sections", fontsize=11)
        ax_multi.set_xlabel("R [m]")
        ax_multi.set_ylabel("Z [m]")
        ax_multi.grid(True, linestyle='--', alpha=0.5)
        ax_multi.set_aspect('equal')
    
    # 信息面板
    if ax_info is not None and metrics is not None:
        info_text = "Optimization metrics:\n"
        if 'volume' in metrics:
            info_text += f"Volume: {metrics['volume']:.3f} m^3\n"
        if 'aspect_ratio' in metrics:
            info_text += f"Aspect ratio: {metrics['aspect_ratio']:.3f}\n"
        if 'magnetic_shear' in metrics:
            info_text += f"Magnetic shear: {metrics['magnetic_shear']:.4f}\n"
        if 'total' in metrics:
            info_text += f"Total score: {metrics['total']:.4f}\n"
        if compare_with is not None:
            info_text += "\n--- Baseline comparison ---\n"
            if 'volume' in metrics:
                # 安全地获取baseline体积
                baseline_vol = 0
                if hasattr(compare_with.wout, 'volume_p'):
                    vol_p = compare_with.wout.volume_p
                    if hasattr(vol_p, '__len__') and len(vol_p) > 0:
                        baseline_vol = float(vol_p[-1])
                    else:
                        baseline_vol = float(vol_p) if vol_p is not None else 0
                elif hasattr(compare_with.wout, 'volume'):
                    baseline_vol = float(compare_with.wout.volume) if compare_with.wout.volume is not None else 0
                
                vol_diff = metrics['volume'] - baseline_vol
                vol_pct = (vol_diff / baseline_vol * 100) if baseline_vol > 0 else 0
                info_text += f"Volume delta: {vol_diff:+.3f} m^3 ({vol_pct:+.2f}%)\n"
        ax_info.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description='Plot VMEC plasma geometry from a MOLLM checkpoint')
    parser.add_argument('--pkl', required=True, help='checkpoint pkl file')
    parser.add_argument('--baseline', required=False, 
                       default=str(Path(__file__).parent / 'vmecpp' / 'calculations' / 'input.w7x'),
                       help='baseline input.w7x path')
    parser.add_argument('--outdir', required=False, default='moo_results/plots', help='output directory for plots/files')
    parser.add_argument('--topk', type=int, default=3, help='number of candidates per metric')
    parser.add_argument('--allow_infeasible', action='store_true', help='include infeasible candidates')
    parser.add_argument('--surfaces', type=lambda s: [float(x) for x in s.split(',') if x.strip()],
        default=[1.0, 0.7, 0.4],
                       help='normalized flux surfaces to plot (0-1), e.g. "1.0,0.7,0.4"')
    parser.add_argument('--compare_baseline', action='store_true',
                       help='compare against the baseline design')
    parser.add_argument('--skip_failed', action='store_true', default=True,
                       help='skip candidates whose VMEC run fails (default)')
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    baseline_input = Path(args.baseline)
    outdir = Path(args.outdir)

    if not pkl_path.exists():
        raise FileNotFoundError(f'pkl not found: {pkl_path}')
    if not baseline_input.exists():
        raise FileNotFoundError(f'baseline input.w7x not found: {baseline_input}')

    logger.info(f"Loading pkl file: {pkl_path}")
    data = load_pkl(str(pkl_path))
    items = iter_items_from_data(data)
    if not items:
        raise RuntimeError('Failed to extract any candidate Item from the pkl.')
    logger.info(f"Extracted {len(items)} candidates")

    # 获取最优解
    best = pick_best_by_metrics(items, require_feasible=not args.allow_infeasible, topk=args.topk)
    
    # 总是运行baseline用于对比（这样更容易看出优化效果）
    logger.info("Running baseline design for comparison...")
    tmp_baseline = outdir / "baseline_tmp"
    tmp_baseline.mkdir(parents=True, exist_ok=True)
    baseline_coeffs = get_baseline_coeffs(baseline_input)
    baseline_output = run_vmec_safe(baseline_input, baseline_coeffs, tmp_baseline)
    if baseline_output is None:
        logger.warning("Baseline run failed; skipping comparison")
        baseline_output = None
    else:
        logger.info("Baseline run succeeded; will be used for comparison")

    # 汇总表
    summary_rows = []
    tmp_base = outdir / "vmec_tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)

    # 绘制最优解
    for metric, entries in best.items():
        for rank, (item, metrics, coeffs) in enumerate(entries, start=1):
            tag = f'{metric}_top{rank}'
            out_png = outdir / f'{tag}.png'
            
            logger.info(f"Processing {tag}...")
            vmec_output = run_vmec_safe(baseline_input, coeffs, tmp_base / tag)
            
            if vmec_output is None:
                if args.skip_failed:
                    logger.warning(f"Skipping {tag} (VMEC run failed)")
                    continue
                else:
                    raise RuntimeError(f"{tag} VMEC run failed")
            
            # 绘制（总是与baseline对比以突出优化效果）
            success = plot_plasma_geometry(
                vmec_output,
                out_png,
                surface_fracs=args.surfaces,
                title_suffix=f"{metric} rank={rank}",
                compare_with=baseline_output,
                compare_label="Baseline",
                metrics=metrics,  # 传递指标用于显示数值
            )
            
            if success:
                row = {
                    'metric': metric,
                    'kind': 'best',
                    'rank': rank,
                    'png': str(out_png),
                }
                row.update({k: metrics.get(k) for k in ['volume', 'aspect_ratio', 'magnetic_shear', 'is_feasible', 'total']})
                summary_rows.append(row)
                logger.info(f"Generated {out_png}")

    # 保存汇总表
    if HAS_PANDAS and summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = outdir / 'summary.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary saved: {csv_path}")

    logger.info(f'Done! Output directory: {outdir.resolve()}')
    logger.info(f'Generated {len(summary_rows)} figures')


if __name__ == '__main__':
    main()
