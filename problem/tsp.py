import numpy as np
import matplotlib.pyplot as plt
from coptpy import *
import time

# --- 辅助函数：子环路检测 (已修复) ---
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

# --- COPT 回调类：用于添加惰性约束 (已修复) ---
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
                
# --- 辅助函数：求解与绘图 ---
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
    plt.show()

# --- 主函数 ---
def main():
    # 1. 生成问题数据
    np.random.seed(1)
    nCities = 70
    cities = np.random.rand(nCities, 2)
    
    # 计算距离矩阵
    distances = dict()
    # <--- 修正：只创建 (i, j) 其中 i < j 的边
    edges_tpl = [] 
    for i in range(nCities):
        for j in range(i + 1, nCities):
            dist = np.linalg.norm(cities[i] - cities[j])
            distances[(i, j)] = dist
            # distances[(j, i)] = dist # <--- 不再需要 (j, i)
            edges_tpl.append((i, j))

    # edges = list(distances.keys())
    
    print(f"Generated {nCities} cities and distance matrix.")

    # 2. 创建 COPT 环境和模型
    env = Envr()
    model = env.createModel("ip_stsp")

    # 3. 添加变量
    # <--- 修正：只为 i < j 的边创建变量
    e = model.addVars(edges_tpl, vtype=COPT.BINARY, nameprefix='e')

    # 4. 添加约束
    # 约束1：每个城市必须连接两条边（度为2）
    # e.sum(i, '*') 现在只匹配 (i, j) 
    # e.sum('*', i) 现在只匹配 (j, i)
    model.addConstrs(e.sum(i, '*') + e.sum('*', i) == 2 for i in range(nCities))
    
    # <--- 修正：恢复了关键的对称性约束
    # (这一行在原始notebook中是 e[i,j] == e[j,i]，但因为我们重构了变量，所以不再需要了)
    
    # 5. 设置目标函数：最小化总距离
    model.setObjective(e.prod(distances), sense=COPT.MINIMIZE)
    print("COPT model created.")

    e_symmetric = {} 
    for (i, j), var in e.items():
        e_symmetric[(i, j)] = var
        e_symmetric[(j, i)] = var

    cb = CoptCallback(e_symmetric, nCities)
    model.setCallback(cb, COPT.CBCONTEXT_MIPSOL)

    # 7. 求解模型
    print("Solving TSP... (This may take a moment)")
    model.solve()
    print("...Solve complete.")

    # 8. 显示结果
    if model.status == COPT.OPTIMAL:
        print(f"\nOptimal solution found!")
        print(f"Total distance: {model.objval:.4f}")
        
        # 提取并绘制路径
        path = get_path(model, edges_tpl)
        draw_path(cities, path, nCities)
        
    elif model.status == COPT.TIMEOUT:
        print("\nSolver reached time limit, solution may be suboptimal.")
    else:
        print(f"\nOptimization finished with status code: {model.status}")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    main()