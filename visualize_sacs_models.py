import os
import glob
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import logging

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_sacs_file(file_path):
    """
    解析SACS输入文件，提取节点、杆件和截面组信息。
    【已修正】使用正则表达式来更稳健地处理JOINT行的不规则格式。
    """
    joints = {}
    members = []
    groups = {}

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 提取节点信息 (JOINT) - 【修正部分】
                if line.startswith('JOINT'):
                    try:
                        # 使用正则表达式查找行内的所有数字（包括整数和浮点数）
                        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                        
                        # 通常第一个是ID，后面三个是X, Y, Z坐标
                        if len(numbers) >= 4:
                            joint_id = numbers[0]
                            x = float(numbers[1])
                            y = float(numbers[2])
                            z = float(numbers[3])
                            joints[joint_id] = (x, y, z)
                        else:
                             logging.warning(f"在JOINT行中未找到足够的坐标数据: {line.strip()} in {file_path}")
                    except (ValueError, IndexError) as e:
                        logging.warning(f"无法解析JOINT行: {line.strip()} in {file_path} (Error: {e})")
                        continue

                # 提取杆件信息 (MEMBER) - 【保持不变】
                elif line.startswith('MEMBER'):
                    try:
                        parts = line.split()
                        if len(parts) >= 4:
                            joint_a = parts[1]
                            joint_b = parts[2]
                            group_id = parts[3]
                            members.append({'endpoints': (joint_a, joint_b), 'group': group_id})
                    except IndexError:
                        logging.warning(f"无法解析MEMBER行: {line.strip()} in {file_path}")
                        continue

                # 提取截面组信息 (GRUP) - 【保持不变】
                elif line.startswith('GRUP'):
                    try:
                        group_id = line[6:10].strip()
                        od = float(line[18:25])
                        groups[group_id] = {'od': od}
                    except (ValueError, IndexError):
                        group_id = line.split()[1] if len(line.split()) > 1 else "UNKNOWN"
                        if group_id not in groups:
                            groups[group_id] = {'od': 10.0} 
                        logging.debug(f"无法解析GRUP行OD，使用默认值: {line.strip()} in {file_path}")
                        continue

    except FileNotFoundError:
        logging.error(f"文件未找到: {file_path}")
        return None, None, None
    except Exception as e:
        logging.error(f"解析文件时发生错误 {file_path}: {e}")
        return None, None, None

    return joints, members, groups



def plot_model(file_path, output_path, joints, members, groups):
    """
    使用Matplotlib绘制并保存3D模型的可视化图像。
    """
    if not joints or not members:
        logging.warning(f"缺少节点或杆件数据，跳过绘图: {os.path.basename(file_path)}")
        return

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同的杆件类型（GRUP ID）创建颜色和粗细映射
    unique_groups = sorted(list(set(m['group'] for m in members)))
    # 使用 'viridis' colormap 创建一组区分明显的颜色
    colors = plt.get_cmap('viridis', len(unique_groups))
    group_styles = {
        group_id: {
            'color': colors(i),
            # 将OD值缩放以获得合适的线条宽度
            'linewidth': max(1, groups.get(group_id, {'od': 10.0})['od'] / 15.0)
        }
        for i, group_id in enumerate(unique_groups)
    }

    # 绘制所有杆件
    for member in members:
        joint_a_id, joint_b_id = member['endpoints']
        group_id = member['group']

        # 确保杆件连接的两个节点都存在
        if joint_a_id in joints and joint_b_id in joints:
            p1 = joints[joint_a_id]
            p2 = joints[joint_b_id]
            style = group_styles.get(group_id, {'color': 'gray', 'linewidth': 1}) # 默认样式

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=style['color'],
                    linewidth=style['linewidth'])

    # 绘制节点（可选，如果想让节点更明显）
    # all_coords = np.array(list(joints.values()))
    # ax.scatter(all_coords[:, 0], all_coords[:, 1], all_coords[:, 2], color='red', s=10)


    # 设置坐标轴标签和标题
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f"SACS Model Visualization\n({os.path.basename(file_path)})", fontsize=16)

    # 自动调整坐标轴比例以获得更好的视觉效果
    all_coords = np.array(list(joints.values()))
    x_coords, y_coords, z_coords = all_coords[:,0], all_coords[:,1], all_coords[:,2]

    # 设置一个合理的坐标轴范围，避免变形
    max_range = np.array([x_coords.max()-x_coords.min(), y_coords.max()-y_coords.min(), z_coords.max()-z_coords.min()]).max() / 2.0
    mid_x = (x_coords.max()+x_coords.min()) * 0.5
    mid_y = (y_coords.max()+y_coords.min()) * 0.5
    mid_z = (z_coords.max()+z_coords.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


    # 保存图像
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"成功保存图像到: {output_path}")
    except Exception as e:
        logging.error(f"保存图像失败: {e}")

    plt.close(fig) # 关闭图形，释放内存


def main():
    """
    主函数，执行整个流程。
    """
    # --- 路径已修改为WSL格式 ---
    # Windows的 D:\ 路径在WSL中对应 /mnt/d/
    backup_dir = "/mnt/d/wsl_sacs_exchange/demo06_project/Demo06/backups"
    
    # 输出目录使用相对路径，它会在你运行脚本的当前目录下创建sacs_visualizations文件夹，无需修改
    output_dir = "./sacs_visualizations"
    # -------------------------

    if not os.path.isdir(backup_dir):
        logging.error(f"指定的备份文件夹不存在: {backup_dir}")
        return

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有SACS输入文件（通常以sacinp开头）
    sacs_files = glob.glob(os.path.join(backup_dir, "*.demo06"))
    if not sacs_files:
        logging.warning(f"在 {backup_dir} 中没有找到任何 'sacs' 文件。")
        return

    logging.info(f"找到 {len(sacs_files)} 个模型文件，开始处理...")

    for file_path in sacs_files:
        logging.info(f"--- 正在处理: {os.path.basename(file_path)} ---")
        joints, members, groups = parse_sacs_file(file_path)

        if joints and members and groups:
            # 定义输出图片的文件名
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{base_filename}.png"
            output_image_path = os.path.join(output_dir, output_filename)

            plot_model(file_path, output_image_path, joints, members, groups)

    logging.info("--- 所有文件处理完毕 ---")


if __name__ == "__main__":
    main()