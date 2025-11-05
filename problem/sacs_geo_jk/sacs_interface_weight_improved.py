# problem/sacs/sacs_interface_weight_improved.py (V7 - Ultimate Stability)
import os
import sqlite3
import re
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SacsWeightCalculator")

STEEL_DENSITY_LBS_PER_IN3 = 0.28356
STEEL_AREAS_IN2 = {
    "W24X55": 16.2, "W24X62": 18.2, "W24X68": 20.0, "W24X76": 22.4, "W24X84": 24.7,
    "W24X94": 27.7, "W24X103": 30.3, "W24X104": 30.6, "W24X117": 34.4, "W24X131": 38.5,
    "W24X146": 43.0, "W24X162": 47.7, "W24X176": 51.8, "W24X192": 56.5, "W24X207": 60.9,
    "W24X229": 67.3
}

def parse_grup_and_pgrup_from_sacinp(sacinp_path: str) -> dict:
    """从sacinp文件解析所有GRUP和PGRUP卡的属性。"""
    properties = {}
    if not os.path.exists(sacinp_path):
        logger.error(f"sacinp文件不存在: {sacinp_path}")
        return properties

    try:
        with open(sacinp_path, 'r', encoding='latin-1') as f:
            for line in f:
                line_stripped = line.strip()
                parts = line_stripped.split()

                if not parts: continue

                if parts[0] == 'GRUP' and len(parts) > 1:
                    group_name = parts[1]
                    # I-Beam
                    w_section_match = re.search(r'(W\d+X\d+)', line_stripped)
                    if w_section_match:
                        section_name = w_section_match.group(1)
                        if section_name in STEEL_AREAS_IN2:
                            properties[group_name] = {'type': 'ibeam', 'area': STEEL_AREAS_IN2[section_name]}
                        continue
                    # Tubular
                    try:
                        od_str = line[18:24].strip()
                        wt_str = line[25:30].strip()
                        od = float(od_str)
                        wt = float(wt_str)
                        # 使用精确的面积公式
                        area = (math.pi / 4) * (od**2 - (od - 2*wt)**2)
                        properties[group_name] = {'type': 'tubular', 'area': area}
                    except (ValueError, IndexError):
                        continue
                
                elif parts[0] == 'PGRUP' and len(parts) > 1:
                    group_name = parts[1]
                    thick_match = re.search(r"(\d+\.\d+)", line[10:])
                    if thick_match:
                        thickness = float(thick_match.group(1))
                        properties[group_name] = {'type': 'plate', 'thickness': thickness}

    except Exception as e:
        logger.error(f"解析sacinp文件时出错: {e}", exc_info=True)
    
    logger.info(f"从 sacinp 解析了 {len(properties)} 个组/板的属性。")
    return properties

def calculate_sacs_weight_from_db(project_path: str) -> dict:
    sacinp_path = os.path.join(project_path, 'sacinp.demo06')
    db_path = os.path.join(project_path, 'sacsdb.db')

    if not os.path.exists(db_path):
        return {'status': 'error', 'error': '数据库文件不存在'}

    group_properties = parse_grup_and_pgrup_from_sacinp(sacinp_path)
    if not group_properties:
        return {'status': 'error', 'error': '无法从sacinp文件解析任何组属性'}

    total_weight_lbs = 0.0

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # --- V7核心修复: 只查询最基本、最不可能改变的列 ---
            # 杆件部分
            query_members = "SELECT DISTINCT MemberName, MemberLength, MemberGroup FROM R_POSTMEMBERRESULTS WHERE MemberLength > 0;"
            cursor.execute(query_members)
            for name, length_ft, group in cursor.fetchall():
                if group in group_properties and group_properties[group]['type'] in ['tubular', 'ibeam']:
                    area_in2 = group_properties[group]['area']
                    volume_in3 = area_in2 * (length_ft * 12.0)
                    total_weight_lbs += volume_in3 * STEEL_DENSITY_LBS_PER_IN3
            
            # 板单元部分 (如果存在)
            query_plates = "SELECT DISTINCT PlateName, PlateGroup, PlateArea FROM R_POSTPLATERESULTS WHERE PlateArea > 0;"
            try:
                cursor.execute(query_plates)
                for name, group, area_ft2 in cursor.fetchall():
                    if group in group_properties and group_properties[group]['type'] == 'plate':
                        thickness_in = group_properties[group]['thickness']
                        volume_in3 = (area_ft2 * 144) * thickness_in
                        total_weight_lbs += volume_in3 * STEEL_DENSITY_LBS_PER_IN3
            except sqlite3.OperationalError:
                logger.warning("数据库中没有板单元结果表 (R_POSTPLATERESULTS)，将忽略板重量。")

    except Exception as e:
        logger.error(f"处理数据库和计算重量时出错: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

    if total_weight_lbs == 0:
        return {'status': 'error', 'error': '计算出的总重量为0'}

    total_weight_tonnes = total_weight_lbs / 2204.62
    logger.info(f"重量计算成功。总重: {total_weight_tonnes:.2f} 吨。")

    return {'status': 'success', 'total_weight_tonnes': total_weight_tonnes}
