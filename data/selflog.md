1. rcls_basic.py
对应章节： 3.2 节 (Basic example)

脚本作用： 在 8x12 的基础线框上，使用 RCLS (Regularized Constrained Least Squares) 方法优化电流分布。这是最基础的验证实验。

生成结果：

rcls_basic_curr2d.png: 优化后的电流分布 2D 展开图。

rcls_basic_model.vtk: 线框的 3D 模型文件（可用 ParaView 查看）。

2. rcls_ports.py
对应章节： 3.3 节 (Restricting space for other components)

脚本作用： 在 12x22 的线框上运行 RCLS 优化，但增加了空间约束，即在特定位置“挖洞”以预留诊断窗口 (Ports)，强制这些区域的电流为零。

生成结果：

rcls_ports_curr2d.png: 避开窗口后的电流分布图。

rcls_ports_model.vtk: 线框 3D 模型。

rcls_ports_port_geometry.vtk: 诊断窗口的几何模型。

3. gsco_modular.py
对应章节： 4.2 节 (Modular coils)

脚本作用： 使用 GSCO (Greedy Stellarator Coil Optimization) 贪婪算法，在一个高分辨率 (96x100) 线框上优化，目标是生成模块化线圈（类似传统仿星器线圈）。

生成结果：

gsco_modular_curr2d.png: 模块化线圈电流分布图。

gsco_modular.vtk: 结果线框模型。

4. gsco_sector.py
对应章节： 4.3 节 (Sector-confined saddle coils)

脚本作用： 使用 GSCO 算法，但施加了扇区约束。强制线圈只能存在于特定的环向扇区内，不能跨越扇区边界，以便于工程组装和维护。

生成结果：

gsco_sector_curr2d.png: 扇区约束下的电流分布图。

gsco_sector_constraints.png: 显示被约束（禁止电流）区域的示意图。

gsco_sector_model.vtk: 结果线框模型。

5. gsco_multistep.py
对应章节： 4.4 节 (Multiple currents / Multistage GSCO)

脚本作用： 这是一个多阶段优化脚本。它通过逐步降低电流的方式多次运行 GSCO，生成一组包含不同电流大小的鞍形线圈（Saddle Coils），用于修正由外部环向场线圈产生的磁场。

生成结果：

gsco_multistep_curr2d.png: 多电流成分的分布图。

gsco_multistep.vtk: 最终优化的线框模型。

🛠️ 辅助与工具脚本
helper_functions.py: 包含计算弧长 (equalize_arc_length) 和线圈尺寸 (find_coil_sizes) 的通用函数。
plotting.py: 包含绘制 2D/3D 图和 Poincaré 截面图的函数库，被上述脚本调用。