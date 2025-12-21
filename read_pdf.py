import pickle
import numpy as np
import matplotlib.pyplot as plt

pkl_path = "results/stellarator_coil_gsco_lite/baselines/RandomSearch_42.pkl"

# 1. 读取 pkl
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

all_mols = data["all_mols"]  # list of (Item, gen_idx)

# 2. 取出所有有评分的个体
items = [item for item, gen in all_mols if getattr(item, "total", None) is not None]

print(f"总个体数: {len(all_mols)}, 有效个体数: {len(items)}")

# 3. 提取原始目标值
f_B = np.array([item.property["f_B"] for item in items])
f_S = np.array([item.property["f_S"] for item in items])
I_max = np.array([item.property["I_max"] for item in items])

# 4. 画 (f_B, f_S) 散点图，用 I_max 上色
plt.figure(figsize=(6, 5))
sc = plt.scatter(f_B, f_S, c=I_max, cmap="viridis", s=10, alpha=0.7)
plt.colorbar(sc, label="I_max [MA]")
plt.xlabel("f_B [T² m²]")
plt.ylabel("f_S [active cells]")
plt.title("GSCO-Lite RandomSearch population (raw objectives)")
plt.tight_layout()
plt.show()
# 如需保存为文件：
plt.savefig("gsco_randomsearch_population.png", dpi=200)