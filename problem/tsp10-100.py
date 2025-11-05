import numpy as np
import matplotlib.pyplot as plt
from coptpy import *
import time
import csv  # 导入CSV模块

# --- 辅助函数：子环路检测 (保持不变) ---
def subtour(vals, nCities):
    """
    Given a tuplelist of edges in a solution, find the shortest subtour.
    (Fixed to be more robust)
    """
    # 筛选出解中被选中的边
    edges = tuplelist((i, j) for i, j in vals.keys()
                        if vals[i, j] > 0.5)
    
    unvisited = list(range(nCities))
    cycle = range(nCities + 1)  # 初始长度（比城市数多1）

    while unvisited:  # 只要还有未访问的城市
        thiscycle = []
        # <--- 修正：neighbors 必须从 unvisited 中获取，否则会出错
        neighbors = unvisited[0:1] # Start with the first unvisited node
        
        while neighbors:
            current = neighbors[0]
            
            # <--- 修正：检查 current 是否真的在 unvisited 中
            if current not in unvisited:
                neighbors.pop(0)
                continue
                
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors.pop(0)
            
            # <--- 修正：必须搜索双向的边，并去重
            # 查找与 current 相连且未访问的邻居
            neighbors_out = [j for i, j in edges.select(current, '*') if j in unvisited]
            neighbors_in  = [i for i, j in edges.select('*', current) if i in unvisited]
            
            # 添加新邻居并保持唯一性
            for n in neighbors_out + neighbors_in:
                if n not in neighbors:
                    neighbors.append(n)
        
        # 寻找最短的子环路
        if len(thiscycle) > 0 and len(cycle) > len(thiscycle):
            cycle = thiscycle
            
    return cycle

# --- COPT 回调类：用于添加惰性约束 (保持不变) ---
class CoptCallback(CallbackBase):
    def __init__(self, vars, nCities):
        super().__init__()
        self.vars = vars
        self.nCities = nCities

    def callback(self):
        # 仅在找到新的整数解时触发
        if self.where() == COPT.CBCONTEXT_MIPSOL:
            try: # <--- 修正：在回调中添加 try...except 是最佳实践
                vars = self.vars
                vals = self.getSolution(vars) # 获取当前整数解
                
                # 寻找最短的子环路
                tour = subtour(vals, self.nCities)
                
                # <--- 修正：添加健壮性检查，防止 [0,0] 索引错误
                # 如果找到的环路不是包含所有城市的完整环路，且环路长度 > 1
                if len(tour) > 1 and len(tour) < self.nCities:
                    # 添加子环路消除约束（惰性约束）
                    self.addLazyConstr(quicksum(vars[tour[i], tour[i+1]] for i in range(len(tour)-1)) + 
                                       vars[tour[-1], tour[0]] <= len(tour) - 1)
            except Exception as e:
                # 如果Python代码出错，打印错误，而不是让求解器崩溃
                print(f"Error in callback: {e}")
                
# --- 辅助函数：求解与绘图 (保持不变) ---
def get_path(model, edges):
    """
    从求解器中提取最优路径
    """
    path = []
    allvars = model.getVars()
    
    for var in allvars:
        # 检查变量是否在最优解中被选中
        if var.x >= 1 - model.Param.IntTol:
            # 只添加一个方向的边用于绘图
            if edges[var.getIdx()][0] < edges[var.getIdx()][1]:
                path.append(edges[var.getIdx()])
            
    return path

def draw_path(cities, path, nCities):
    """
    绘制最终的 TSP 路径
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(cities[:, 0], cities[:, 1], marker='o')
    
    # 绘制城市编号
    for i in range(nCities):
        plt.text(cities[i, 0] + 0.01, cities[i, 1] + 0.01, i, fontsize=9)
        
    # 绘制路径
    for i in path:
        plt.plot([cities[i[0], 0], cities[i[1], 0]],
                 [cities[i[0], 1], cities[i[1], 1]], color='red', linewidth=1)
    
    plt.title(f"TSP Solution ({nCities} Cities)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    # plt.show() # 在批量运行中，我们通常不希望自动显示
    # 如需保存图像，可以使用:
    # plt.savefig(f"tsp_solution_{nCities}_cities.png")
    plt.close() # 关闭图像，防止内存泄漏

# --- 核心求解函数 (已修正) ---
def solve_tsp(nCities):
    """
    为给定数量的城市求解TSP问题，并返回性能指标。
    """
    # 1. 生成问题数据
    # 注意：np.random.seed(1) 已移至 main() 中，
    # 确保每次循环生成的城市坐标是确定且可复现的。
    cities = np.random.rand(nCities, 2)
    
    # 计算距离矩阵
    distances = dict()
    edges_tpl = [] 
    for i in range(nCities):
        for j in range(i + 1, nCities):
            dist = np.linalg.norm(cities[i] - cities[j])
            distances[(i, j)] = dist
            edges_tpl.append((i, j))
    
    # 2. 创建 COPT 环境和模型
    try:
        env = Envr()
        model = env.createModel("ip_stsp")
    except CoptError as e:
        # !--- 此处是修正点 ---!
        # 将跨行的 print 语句合并为一行
        print(f"创建COPT环境失败，请检查COPT 许可证: {e}")
        return {
            "nCities": nCities,
            "Status": "Error",
            "ObjVal": -1,
            "SolveTime": -1,
            "MIPGap": -1,
            "NodeCount": -1,
            "SimplexIter": -1,
            "Error": str(e)
        }

    # 3. 添加变量
    e = model.addVars(edges_tpl, vtype=COPT.BINARY, nameprefix='e')

    # 4. 添加约束
    model.addConstrs(e.sum(i, '*') + e.sum('*', i) == 2 for i in range(nCities))
    
    # 5. 设置目标函数
    model.setObjective(e.prod(distances), sense=COPT.MINIMIZE)

    # 6. 设置回调
    e_symmetric = {} 
    for (i, j), var in e.items():
        e_symmetric[(i, j)] = var
        e_symmetric[(j, i)] = var

    cb = CoptCallback(e_symmetric, nCities)
    model.setCallback(cb, COPT.CBCONTEXT_MIPSOL)

    # 7. 求解模型 (并计时)
    start_time = time.time()
    model.solve()
    end_time = time.time()
    solve_time = end_time - start_time

    # 8. 收集指标
    status = model.status
    obj_val = -1
    mip_gap = -1
    node_count = -1
    simplex_iter = -1
    
    if status == COPT.OPTIMAL or status == COPT.TIMEOUT:
        try:
            obj_val = model.objval
            mip_gap = model.mipgap
            node_count = model.nodecnt
            simplex_iter = model.simplexitercnt
        except CoptError: # 如果模型没有解（例如超时前未找到可行解）
            pass
    
    # 可选：如果需要，可以取消注释以绘制和保存每个解的图像
    # if status == COPT.OPTIMAL:
    #     path = get_path(model, edges_tpl)
    #     draw_path(cities, path, nCities) # 将会保存图像

    return {
        "nCities": nCities,
        "Status": status,
        "ObjVal": obj_val,
        "SolveTime": solve_time,
        "MIPGap": mip_gap,
        "NodeCount": node_count,
        "SimplexIter": simplex_iter,
        "Error": "None"
    }

# --- 实验运行主函数 (新) ---
def main():
    # 设置随机种子，确保整个实验可复现
    np.random.seed(1)
    
    results_list = []
    
    # 城市规模：10, 20, 30, ..., 100
    city_sizes = range(10, 101, 10)
    
    print("--- 开始TSP批量实验 ---")
    
    for n in city_sizes:
        print(f"--- 正在求解 nCities = {n} ---")
        result_data = solve_tsp(n)
        results_list.append(result_data)
        print(f"--- nCities = {n} 求解完成. "
              f"状态: {result_data['Status']}, "
              f"时间: {result_data['SolveTime']:.2f}s, "
              f"目标值: {result_data['ObjVal']:.4f}, "
              f"MIPGap: {result_data['MIPGap']:.4f}")

    # 9. 将结果写入CSV文件
    output_filename = "tsp_benchmark_results.csv"
    print(f"\n--- 实验完成。正在将结果写入 {output_filename} ---")
    
    if not results_list:
        print("没有结果可以写入。")
        return

    # 获取表头（基于第一条结果的键）
    fieldnames = results_list[0].keys()
    
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_list)
        print("...CSV文件写入成功。")
    except IOError as e:
        print(f"写入CSV文件失败: {e}")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    main()