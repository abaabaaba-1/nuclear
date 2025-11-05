# problem/sacs/sacs_interface_uc.py (Modified)
import os
import sqlite3
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import logging

class UCExtractor:
    def __init__(self, work_dir: str):
        self.db_path = Path(work_dir) / "sacsdb.db"
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.db_path.exists():
            self.logger.warning(f"数据库文件不存在: {self.db_path}")

    def extract_uc_values_from_db(self) -> Dict[str, Any]:
        self.logger.info("开始提取UC值...")
        if not self.db_path.exists():
            return {"status": "error", "message": f"数据库文件不存在: {self.db_path}"}

        all_member_data = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT MemberName, MaxUC, AxialUC, YYBendingUC, ZZBendingUC
            FROM R_POSTMEMBERRESULTS
            WHERE MaxUC IS NOT NULL AND MaxUC > 0.0
            """
            self.logger.info("执行查询...")
            cursor.execute(query)
            records = cursor.fetchall()
            self.logger.info(f"查询到 {len(records)} 条记录")

            if not records:
                return {"status": "error", "message": "数据库中未找到有效的UC记录"}

            processed_members = set()
            for row in records:
                member_name = row[0]
                if member_name in processed_members:
                    continue
                processed_members.add(member_name)

                all_member_data.append({
                    'member': member_name,
                    'max_uc': float(row[1]),
                    'axial_uc': abs(float(row[2])) if row[2] is not None else 0.0, # 轴向应力比取绝对值
                    'yy_bending_uc': abs(float(row[3])) if row[3] is not None else 0.0,
                    'zz_bending_uc': abs(float(row[4])) if row[4] is not None else 0.0,
                })
            
            conn.close()
            self.logger.info(f"UC值提取完成，处理了 {len(all_member_data)} 个杆件")
            return {"status": "success", "data": all_member_data}

        except sqlite3.Error as e:
            self.logger.error(f"从数据库提取UC值时出错: {e}")
            return {"status": "error", "message": f"数据库查询失败: {e}"}

# --- 核心变更: 重构主接口函数 ---
def get_sacs_uc_summary(work_dir: str = None) -> Dict:
    """
    简化的SACS UC值获取接口，现在返回一个包含各类UC最大值的摘要。
    
    Args:
        work_dir: SACS工作目录
    
    Returns:
        一个字典，包含整体状态和各项UC指标的最大值。
    """
    extractor = UCExtractor(work_dir)
    uc_results = extractor.extract_uc_values_from_db()

    if uc_results["status"] != "success":
        return {
            "status": "failed",
            "message": uc_results["message"],
            "max_uc": 999.0, # 惩罚值
            "axial_uc_max": 999.0,
            "bending_uc_max": 999.0
        }

    member_data = uc_results["data"]
    if not member_data:
        return {
            "status": "failed",
            "message": "提取到的杆件数据为空",
            "max_uc": 999.0,
            "axial_uc_max": 999.0,
            "bending_uc_max": 999.0
        }

    # 计算各项指标的最大值
    all_max_uc = [d['max_uc'] for d in member_data]
    all_axial_uc = [d['axial_uc'] for d in member_data]
    # 弯曲UC是两个方向中较大的一个
    all_bending_uc = [max(d['yy_bending_uc'], d['zz_bending_uc']) for d in member_data]

    summary = {
        "status": "success",
        "total_members": len(member_data),
        "max_uc": max(all_max_uc) if all_max_uc else 0.0,
        "axial_uc_max": max(all_axial_uc) if all_axial_uc else 0.0,
        "bending_uc_max": max(all_bending_uc) if all_bending_uc else 0.0
    }
    
    return summary
