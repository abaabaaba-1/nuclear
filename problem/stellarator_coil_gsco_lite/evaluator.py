"""
Stellarator Coil Design Evaluator - GSCO-Lite (Simple Cell-Based)
真正模仿 Hammond 2025 的 GSCO 算法思想

关键特性:
1. 每个grid cell只能有一个单位电流环（-1, 0, +1）
2. 所有active cells使用相同的固定电流
3. LLM只需操作12×12=144个cells，而非几千个loop IDs
4. 自动满足KCL（闭环的叠加）
5. 修复了磁场积分计算的bug

作者: AI Assistant
日期: 2025-11-30
基于: Hammond, K.C. 2025. Nucl. Fusion 65 046012
"""

import json
import numpy as np
import logging
import random
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Simsopt 导入
try:
    from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
    from simsopt.field import WireframeField, BiotSavart, Current, coils_via_symmetries
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
    from simsopt.field import WireframeField, BiotSavart, Current, coils_via_symmetries
    import netCDF4
    from scipy.io import netcdf_file
    SIMSOPT_AVAILABLE = True
except ImportError:
    SIMSOPT_AVAILABLE = False
    logging.warning("Simsopt not available. Cannot perform real physics evaluation.")


def generate_initial_population(config, seed):
    """
    生成初始种群（Simple Cell-Based）
    
    支持从文件加载 (Warm Start)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    pop_size = config.get('optimization.pop_size', 50)
    nPhi = config.get('coil_design.wf_nPhi', 12)
    nTheta = config.get('coil_design.wf_nTheta', 12)
    min_cells = config.get('llm_constraints.min_active_cells', 3)
    max_cells = config.get('llm_constraints.max_active_cells', 50)
    
    initial_population = []

    # Check for warm start file
    warm_start_file = config.get('optimization.initial_population_file')
    if warm_start_file:
        if not os.path.isabs(warm_start_file):
             # Try to resolve relative to project root
             project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             warm_start_path = os.path.join(project_root, warm_start_file)
        else:
             warm_start_path = warm_start_file
             
        if os.path.exists(warm_start_path):
            logging.info(f"Loading initial population from {warm_start_path}")
            try:
                with open(warm_start_path, 'r') as f:
                    seeds = json.load(f)
                
                for cells in seeds:
                    config_json = json.dumps({"cells": cells})
                    initial_population.append(config_json)
                
                logging.info(f"Loaded {len(initial_population)} seeds from file.")
            except Exception as e:
                logging.warning(f"Failed to load warm start file: {e}")
        else:
            logging.warning(f"Warm start file not found: {warm_start_path}")

    # If seeds loaded, check if we need more
    current_count = len(initial_population)
    if current_count < pop_size:
        needed = pop_size - current_count
        logging.info(f"Generating {needed} random individuals to fill population.")
        
        # 生成随机cell配置
        for _ in range(needed):
            n_active = random.randint(min_cells, min(max_cells, 20))  # 初始不要太复杂
            cells = []
            
            for _ in range(n_active):
                phi = random.randint(0, nPhi - 1)
                theta = random.randint(0, nTheta - 1)
                state = random.choice([-1, 1])  # 不包括0（0表示inactive）
                cells.append([phi, theta, state])
            
            config_json = json.dumps({"cells": cells})
            initial_population.append(config_json)
    
    # Trim if too many (though rare)
    initial_population = initial_population[:pop_size]
    
    logging.info(f"Final initial population size: {len(initial_population)}")
    logging.info(f"  Grid size: {nPhi} × {nTheta} = {nPhi * nTheta} cells")
    
    return initial_population


class SimpleGSCOEvaluator:
    """
    简单的基于cell的评估器（GSCO-Lite）
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 目标和方向
        self.objs = config.get('goals', [])
        opt_dirs = config.get('optimization_direction', []) or []
        self.obj_directions = {
            obj: opt_dirs[i] if i < len(opt_dirs) else 'min'
            for i, obj in enumerate(self.objs)
        }
        
        # Wireframe参数
        self.wf_nPhi = config.get('coil_design.wf_nPhi', 12)
        self.wf_nTheta = config.get('coil_design.wf_nTheta', 12)
        self.total_segments = self.wf_nPhi * self.wf_nTheta * 2
        self.unit_current = config.get('coil_design.unit_current', 0.2)  # MA
        
        # Forbidden cells
        self.forbidden_cells = set()
        forbidden_list = config.get('coil_design.forbidden_cells', [])
        if forbidden_list:
            for item in forbidden_list:
                if len(item) == 2:
                    self.forbidden_cells.add((item[0], item[1]))
        
        # 等离子体参数
        self.wout_file = self._resolve_wout_path(config)
        self.plas_n = config.get('plasma_boundary.plas_n', 32)
        
        # 初始化表面
        self._initialize_surfaces()
        
        # 创建背景场（可选）
        self.use_background_field = config.get('coil_design.use_background_field', True)
        self.mf_tf = None
        self.B_ext_n = None
        if self.use_background_field:
            self._create_background_field()
        else:
            self.logger.info("Background field disabled (testing pure coil field)")
        
        # 目标范围
        self.objective_ranges = config.get('objective_ranges', {})
        
        # Gradient helper state
        self.gradient_helper_initialized = False
        self.response_matrix = None
        self.dS_flat = None
        self.norm_factor = None
        self.n_grid_points = 0
        
        self.logger.info("="*70)
        self.logger.info("GSCO-Lite Evaluator Initialized")
        self.logger.info("="*70)
        self.logger.info(f"  Grid: {self.wf_nPhi} × {self.wf_nTheta} cells")
        self.logger.info(f"  Total segments: {self.total_segments}")
        self.logger.info(f"  Unit current: {self.unit_current} MA")
        self.logger.info(f"  Forbidden cells: {len(self.forbidden_cells)} cells blocked")
        if self.forbidden_cells:
            self.logger.info(f"    {list(self.forbidden_cells)}")
        self.logger.info(f"  Plasma resolution: {self.plas_n} × {self.plas_n}")
        self.logger.info("="*70)

    def _init_gradient_helper(self):
        """Pre-compute response matrices for fast gradient hints"""
        if not SIMSOPT_AVAILABLE or self.gradient_helper_initialized:
            return
            
        try:
            self.logger.info("Pre-computing gradient response matrices (lazy init)...")
            
            # Surface integration weights
            # dS is the magnitude of the normal vector (Jacobian)
            # f_B = 0.5 * sum(B_n^2 * dS) / (ntheta * nphi)
            normal_vec = self.surf_plas.normal()
            dS = np.sqrt(np.sum(normal_vec**2, axis=2))
            self.dS_flat = dS.flatten()
            self.n_grid_points = self.dS_flat.size
            self.norm_factor = 0.5 / (self.surf_plas.quadpoints_phi.size * self.surf_plas.quadpoints_theta.size)
            
            # Response matrix
            n_cells = self.wf_nPhi * self.wf_nTheta
            self.response_matrix = np.zeros((n_cells, self.n_grid_points))
            
            # Compute response for each cell
            for phi in range(self.wf_nPhi):
                for theta in range(self.wf_nTheta):
                    idx = phi * self.wf_nTheta + theta
                    
                    # Activate just this cell
                    cell = [phi, theta, 1] 
                    currents = self.cells_to_segment_currents([cell])
                    
                    wf = ToroidalWireframe(self.surf_wf, self.wf_nPhi, self.wf_nTheta)
                    wf.currents[:] = currents
                    wf_field = WireframeField(wf)
                    
                    points = self.surf_plas.gamma().reshape((-1, 3))
                    wf_field.set_points(points)
                    B_vec = wf_field.B()
                    normals = self.surf_plas.unitnormal().reshape((-1, 3))
                    B_n = np.sum(B_vec * normals, axis=1)
                    
                    self.response_matrix[idx, :] = B_n
            
            self.gradient_helper_initialized = True
            self.logger.info("Gradient helper initialized.")
            
        except Exception as e:
            self.logger.warning(f"Failed to init gradient helper: {e}")

    def _calculate_gradient_hints(self, cells: List, top_k=5) -> List[str]:
        """
        Calculate gradient hints: Top-k single moves that reduce f_B.
        """
        if not self.gradient_helper_initialized:
            self._init_gradient_helper()
        
        if not self.gradient_helper_initialized:
            return []

        try:
            # Reconstruct current B_n
            if self.B_ext_n is not None:
                current_B_n = self.B_ext_n.flatten().copy()
            else:
                current_B_n = np.zeros(self.n_grid_points)
                
            current_cell_map = {}
            for c in cells:
                if isinstance(c, list) and len(c) == 3:
                    phi, theta, state = c
                    idx = phi * self.wf_nTheta + theta
                    current_cell_map[idx] = state
                    if state != 0:
                        current_B_n += state * self.response_matrix[idx]
            
            # Current f_B
            current_f_B = np.sum(current_B_n**2 * self.dS_flat) * self.norm_factor
            
            # Evaluate all moves
            candidates = []
            
            for phi in range(self.wf_nPhi):
                for theta in range(self.wf_nTheta):
                    idx = phi * self.wf_nTheta + theta
                    current_pol = current_cell_map.get(idx, 0)
                    response = self.response_matrix[idx]
                    
                    for target_pol in [-1, 0, 1]:
                        if target_pol == current_pol:
                            continue
                            
                        # Quick update formula
                        delta_pol = target_pol - current_pol
                        new_B_n = current_B_n + delta_pol * response
                        new_f_B = np.sum(new_B_n**2 * self.dS_flat) * self.norm_factor
                        
                        delta_f = new_f_B - current_f_B
                        
                        if delta_f < 0:
                            # Describe move
                            if current_pol == 0:
                                action = "ADD"
                            elif target_pol == 0:
                                action = "REMOVE"
                            else:
                                action = "FLIP"
                                
                            candidates.append({
                                'phi': phi, 'theta': theta,
                                'action': action,
                                'new_state': target_pol,
                                'delta_f': delta_f
                            })
                            
            # Sort by improvement
            candidates.sort(key=lambda x: x['delta_f'])
            
            # Format top-k
            hints = []
            for item in candidates[:top_k]:
                hints.append(
                    f"{item['action']} ({item['phi']},{item['theta']}) to state {item['new_state']} "
                    f"(d_fB={item['delta_f']:.2e})"
                )
                
            return hints
            
        except Exception as e:
            self.logger.warning(f"Error calculating gradient hints: {e}")
            return []

    def _resolve_wout_path(self, config) -> str:
        """解析 wout 文件路径"""
        wout_config = config.get('plasma_boundary.wout_file')
        if not wout_config:
            raise ValueError("plasma_boundary.wout_file is not specified")
        
        if os.path.isabs(wout_config):
            return wout_config
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            return os.path.join(project_root, wout_config)
    
    def _load_vmec_surface(self, wout_file: str, nphi: int, ntheta: int) -> SurfaceRZFourier:
        """从 VMEC wout 文件加载等离子体表面"""
        with netCDF4.Dataset(wout_file, 'r') as ds:
            nfp = int(ds.variables['nfp'][:])
            mpol = int(ds.variables['mpol'][:])
            ntor = int(ds.variables['ntor'][:])
            rmnc = ds.variables['rmnc'][-1, :]
            zmns = ds.variables['zmns'][-1, :]
            xm = ds.variables['xm'][:].astype(int)
            xn = ds.variables['xn'][:].astype(int)
        
        surf = SurfaceRZFourier(
            nfp=nfp, stellsym=True, mpol=mpol, ntor=ntor,
            quadpoints_phi=np.linspace(0, 1, nphi, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False)
        )
        
        for i, (m, n) in enumerate(zip(xm, xn)):
            n_normalized = n // nfp
            try:
                surf.set_rc(m, n_normalized, rmnc[i])
                surf.set_zs(m, n_normalized, zmns[i])
            except:
                pass
        
        return surf
    
    def _create_winding_surface_from_plasma(self) -> SurfaceRZFourier:
        """从等离子体表面外扩创建绕组表面"""
        expansion_factor = self.config.get('coil_design.winding_surface_expansion', 1.2)
        surf_wf = SurfaceRZFourier(
            nfp=self.surf_plas.nfp, stellsym=self.surf_plas.stellsym,
            mpol=self.surf_plas.mpol, ntor=self.surf_plas.ntor,
            quadpoints_phi=self.surf_plas.quadpoints_phi,
            quadpoints_theta=self.surf_plas.quadpoints_theta
        )
        
        for m in range(self.surf_plas.mpol + 1):
            for n in range(-self.surf_plas.ntor, self.surf_plas.ntor + 1):
                try:
                    rc = self.surf_plas.get_rc(m, n)
                    zs = self.surf_plas.get_zs(m, n)
                    surf_wf.set_rc(m, n, rc * expansion_factor)
                    surf_wf.set_zs(m, n, zs * expansion_factor)
                except:
                    pass
        
        return surf_wf
    
    def _initialize_surfaces(self):
        """初始化等离子体表面和绕组表面"""
        try:
            if not Path(self.wout_file).exists():
                raise FileNotFoundError(f"VMEC file not found: {self.wout_file}")
            
            self.surf_plas = self._load_vmec_surface(self.wout_file, self.plas_n, self.plas_n)
            self.surf_wf = self._create_winding_surface_from_plasma()
            
            self.logger.info(f"Loaded plasma surface from: {self.wout_file}")
            self.logger.info(f"  nfp: {self.surf_plas.nfp}")
            self.logger.info(f"  Major radius: {self.surf_plas.get_rc(0,0):.2f} m")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize surfaces: {e}")
            raise
    
    def _create_background_field(self):
        """创建固定的背景场（TF Coils）"""
        if not SIMSOPT_AVAILABLE:
            return
        
        try:
            with netcdf_file(self.wout_file, 'r', mmap=False) as f:
                Rmajor = float(f.variables['Rmajor_p'][()])
                b0 = float(f.variables['b0'][()])
            
            n_tf_coils_hp = 3
            nfp = self.surf_plas.nfp
            mu0 = 4.0 * np.pi * 1e-7
            pol_cur = -2.0 * np.pi * Rmajor * b0 / mu0
            tf_coil_current = pol_cur / (2 * nfp * n_tf_coils_hp)
            
            self.logger.info(f"Background TF coils: {n_tf_coils_hp} per half-period")
            self.logger.info(f"  B0: {b0:.2f} T, R0: {Rmajor:.2f} m")
            self.logger.info(f"  TF current: {tf_coil_current/1e6:.2f} MA")
            
            tf_curves = create_equally_spaced_curves(
                n_tf_coils_hp, nfp, stellsym=True, R0=Rmajor, R1=Rmajor*0.5
            )
            tf_curr = [Current(tf_coil_current) for _ in range(n_tf_coils_hp)]
            tf_coils = coils_via_symmetries(tf_curves, tf_curr, nfp, True)
            self.mf_tf = BiotSavart(tf_coils)
            
            # 预计算背景场
            points = self.surf_plas.gamma().reshape((-1, 3))
            self.mf_tf.set_points(points)
            B_ext = self.mf_tf.B()
            normals = self.surf_plas.unitnormal().reshape((-1, 3))
            self.B_ext_n = np.sum(B_ext * normals, axis=1)
            
            self.logger.info(f"Background field: max|Bn| = {np.max(np.abs(self.B_ext_n)):.4f} T")
            
        except Exception as e:
            self.logger.warning(f"Failed to create background field: {e}")
            self.mf_tf = None
            self.B_ext_n = None
    
    def cells_to_segment_currents(self, cells: List) -> np.ndarray:
        """
        将cell状态转换为完整的segment电流数组
        
        这是核心转换函数：实现了GSCO中的"闭环叠加"思想
        
        Args:
            cells: List of [phi_idx, theta_idx, state] or dict format
        
        Returns:
            current_array: shape (total_segments,), 单位: Ampere
        """
        current_array = np.zeros(self.total_segments)
        
        # 单位电流（Ampere）
        I_unit = self.unit_current * 1e6
        
        # 遍历所有active cells
        for cell in cells:
            try:
                # 支持两种格式
                if isinstance(cell, list):
                    phi_idx, theta_idx, state = cell
                elif isinstance(cell, dict):
                    phi_idx = cell.get('phi', cell.get('phi_idx'))
                    theta_idx = cell.get('theta', cell.get('theta_idx'))
                    state = cell.get('state', cell.get('polarity', 0))
                else:
                    continue
                
                # 验证索引
                if not (0 <= phi_idx < self.wf_nPhi and 0 <= theta_idx < self.wf_nTheta):
                    self.logger.warning(f"Invalid cell index: ({phi_idx}, {theta_idx})")
                    continue
                
                if state == 0:
                    continue  # inactive cell
                
                # 计算该cell的矩形闭环的4条边
                # 使用Hammond论文的segment indexing convention
                nPhiTheta = self.wf_nPhi * self.wf_nTheta
                
                # 下边（phi方向）
                seg_bottom = phi_idx * self.wf_nTheta + theta_idx
                current_array[seg_bottom] += state * I_unit
                
                # 右边（theta方向）
                phi_next = (phi_idx + 1) % self.wf_nPhi
                seg_right = nPhiTheta + phi_next * self.wf_nTheta + theta_idx
                current_array[seg_right] += state * I_unit
                
                # 上边（phi方向，反向）
                theta_next = (theta_idx + 1) % self.wf_nTheta
                seg_top = phi_idx * self.wf_nTheta + theta_next
                current_array[seg_top] -= state * I_unit  # 反向
                
                # 左边（theta方向，反向）
                seg_left = nPhiTheta + phi_idx * self.wf_nTheta + theta_idx
                current_array[seg_left] -= state * I_unit  # 反向
                
            except Exception as e:
                self.logger.warning(f"Error processing cell {cell}: {e}")
                continue
        
        return current_array
    
    def _evaluate_field_error(self, current_array: np.ndarray) -> float:
        """
        评估磁场误差（修复版）
        
        正确实现了论文公式: f_B = (1/2) ∫∫ (B·n)² dS
        
        修复了原实现的bug：
        - 原bug: f_B = 0.5 * [Σ(B_n² * dS) / Σ(dS)] * area （错误的平均×面积）
        - 正确: f_B = 0.5 * Σ(B_n² * dS) * dφ * dθ （正确的积分）
        """
        if not SIMSOPT_AVAILABLE:
            # Fallback: 简单的正则化
            return float(np.sum(current_array**2) * 1e-12)
        
        try:
            # 创建wireframe并设置电流
            wf = ToroidalWireframe(self.surf_wf, self.wf_nPhi, self.wf_nTheta)
            wf.currents[:] = current_array
            wf_field = WireframeField(wf)
            
            # 计算表面上的磁场
            points = self.surf_plas.gamma().reshape((-1, 3))
            wf_field.set_points(points)
            B_wf = wf_field.B()
            
            # 计算法向分量
            normals = self.surf_plas.unitnormal().reshape((-1, 3))
            B_wf_n = np.sum(B_wf * normals, axis=1)
            
            # 叠加背景场
            if self.B_ext_n is not None:
                B_total_n = B_wf_n + self.B_ext_n
            else:
                B_total_n = B_wf_n
            
            # 【修复】正确的磁场误差积分
            ntheta = self.surf_plas.quadpoints_theta.size
            nphi = self.surf_plas.quadpoints_phi.size
            
            # (B·n)²
            B_n_squared = B_total_n ** 2
            B_n_sq_matrix = B_n_squared.reshape((ntheta, nphi))
            
            # 面积元（Jacobian）
            # normal() 返回 ∂r/∂θ × ∂r/∂φ，其模就是参数空间上的面积元
            normal_vec = self.surf_plas.normal()  # shape: (ntheta, nphi, 3)
            dS = np.sqrt(np.sum(normal_vec**2, axis=2))  # shape: (ntheta, nphi)
            
            # 正确的积分
            # quadpoints 定义在 [0,1]×[0,1]，步长 = 1/n
            # 积分 = Σ f(θ_i, φ_j) * |J(θ_i, φ_j)| * Δθ * Δφ
            # 其中 Δθ = 1/ntheta, Δφ = 1/nphi
            # dS 已经包含了物理空间的 Jacobian
            # 所以：f_B = 0.5 * Σ (B·n)² * dS * (1/ntheta) * (1/nphi)
            # 简化：f_B = 0.5 * Σ (B·n)² * dS / (ntheta * nphi)
            f_B = 0.5 * np.sum(B_n_sq_matrix * dS) / (ntheta * nphi)
            
            return float(f_B)
            
        except Exception as e:
            self.logger.error(f"Field evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return 1e5
    
    def evaluate(self, items):
        """
        评估一批候选解
        """
        items, failed_num, repeated_num = self._sanitize_and_validate(items)
        
        original_results = defaultdict(list)
        gradient_hints_list = []
        
        # 逐个评估
        for item in items:
            try:
                config = json.loads(item.value)
                cells = config.get('cells', [])
                
                # Check for forbidden cells
                is_forbidden = False
                for c in cells:
                    if isinstance(c, list):
                        phi, theta, state = c
                    elif isinstance(c, dict):
                        phi = c.get('phi', c.get('phi_idx'))
                        theta = c.get('theta', c.get('theta_idx'))
                        state = c.get('state', c.get('polarity', 0))
                    else:
                        continue
                    
                    if state != 0 and (phi, theta) in self.forbidden_cells:
                        is_forbidden = True
                        break
                
                if is_forbidden:
                    f_B = 1e5
                    f_S = 100
                    I_max = 10.0
                    hints = ["VIOLATION: Forbidden cell used. Remove cells in forbidden zones."]
                else:
                    # 转换为segment currents
                    current_array = self.cells_to_segment_currents(cells)
                    
                    # 计算目标函数
                    f_B = self._evaluate_field_error(current_array)
                    f_S = len([c for c in cells if (c[2] if isinstance(c, list) else c.get('state', 0)) != 0])
                    I_max = (np.max(np.abs(current_array)) / 1e6) if len(current_array) > 0 else 0.0
                    
                    # 计算梯度提示
                    hints = self._calculate_gradient_hints(cells)
                
                gradient_hints_list.append(hints)
                
                original_results['f_B'].append(f_B)
                original_results['f_S'].append(f_S)
                original_results['I_max'].append(I_max)
                
            except Exception as e:
                self.logger.error(f"Evaluation error: {e}")
                import traceback
                traceback.print_exc()
                original_results['f_B'].append(1e5)
                original_results['f_S'].append(100)
                original_results['I_max'].append(10.0)
                gradient_hints_list.append([])
        
        # 转换和分配结果
        transformed_results = {}
        for obj in ['f_B', 'f_S', 'I_max']:
            original_results[obj] = np.array(original_results[obj])
            transformed_results[obj] = self.transform_objectives(obj, original_results[obj])
        
        for idx, item in enumerate(items):
            results = {'original_results': {}, 'transformed_results': {}}
            overall_score = len(['f_B', 'f_S', 'I_max']) * 1.0
            
            for obj in ['f_B', 'f_S', 'I_max']:
                results['original_results'][obj] = original_results[obj][idx]
                results['transformed_results'][obj] = transformed_results[obj][idx]
                overall_score -= results['transformed_results'][obj]
            
            # 添加梯度提示
            results['gradient_hints'] = gradient_hints_list[idx]
            
            results['overall_score'] = overall_score
            item.assign_results(results)
        
        log_dict = {
            'repeated_num': repeated_num,
            'invalid_num': failed_num
        }
        
        return items, log_dict
    
    def _sanitize_and_validate(self, items) -> Tuple[List, int, int]:
        """验证和清理配置"""
        valid_items = []
        failed_count = 0
        repeated_count = 0
        seen = set()
        
        max_cells = self.config.get('llm_constraints.max_active_cells', 50)
        
        for item in items:
            try:
                config = json.loads(item.value)
                cells = config.get('cells', [])
                
                if not isinstance(cells, list):
                    failed_count += 1
                    continue
                
                # 验证和清理cells
                valid_cells_map = {}
                for cell in cells:
                    try:
                        if isinstance(cell, list) and len(cell) == 3:
                            phi, theta, state = cell
                        elif isinstance(cell, dict):
                            phi = cell.get('phi', cell.get('phi_idx', -1))
                            theta = cell.get('theta', cell.get('theta_idx', -1))
                            state = cell.get('state', cell.get('polarity', 0))
                        else:
                            continue
                        
                        # 验证范围
                        if not (0 <= phi < self.wf_nPhi and 0 <= theta < self.wf_nTheta):
                            continue
                        if state not in [-1, 0, 1]:
                            continue
                        if state == 0:  # 跳过inactive cells
                            continue
                        
                        # Deduplicate: overlapping cells overwrite previous ones
                        valid_cells_map[(int(phi), int(theta))] = int(state)
                    except:
                        continue
                
                # Convert back to list
                valid_cells = [[k[0], k[1], v] for k, v in valid_cells_map.items()]
                
                # 限制数量
                if len(valid_cells) > max_cells:
                    valid_cells = valid_cells[:max_cells]
                
                if not valid_cells:
                    failed_count += 1
                    continue
                
                config['cells'] = valid_cells
                item.value = json.dumps(config)
                
                if item.value in seen:
                    repeated_count += 1
                    continue
                
                seen.add(item.value)
                valid_items.append(item)
                
            except Exception as e:
                self.logger.warning(f"Validation error: {e}")
                failed_count += 1
        
        return valid_items, failed_count, repeated_count
    
    def transform_objectives(self, obj: str, values: np.ndarray) -> np.ndarray:
        """目标函数归一化和方向调整"""
        values = self.normalize_objectives(obj, values)
        values = self.adjust_direction(obj, values)
        return values
    
    def normalize_objectives(self, obj: str, values: np.ndarray) -> np.ndarray:
        """归一化到 [0, 1]"""
        ranges = self.objective_ranges
        if obj in ranges:
            min_val, max_val = ranges[obj]
            values = np.clip(values, min_val, max_val)
            if max_val > min_val:
                values = (values - min_val) / (max_val - min_val)
        return values
    
    def adjust_direction(self, obj: str, values: np.ndarray) -> np.ndarray:
        """调整优化方向"""
        direction = self.obj_directions.get(obj, 'min')
        if direction == 'max':
            values = 1.0 - values
        return values


# 导出为统一接口
RewardingSystem = SimpleGSCOEvaluator
