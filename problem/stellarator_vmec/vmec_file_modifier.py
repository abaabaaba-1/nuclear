# problem/stellarator_vmec/vmec_file_modifier.py (新文件)
import re
import logging
from pathlib import Path
from typing import Dict, Optional
import shutil
from datetime import datetime

class VmecFileModifier:
    def __init__(self, project_path: str, input_file: str):
        self.project_path = Path(project_path)
        self.input_file_path = self.project_path / input_file
        self.backup_dir = self.project_path / "backups"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.backup_dir.mkdir(exist_ok=True)
        if not self.input_file_path.exists():
            raise FileNotFoundError(f"VMEC input file not found: {self.input_file_path}")

    def _create_backup(self) -> Optional[Path]:
        """Creates a backup of the current input file."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{self.input_file_path.name}.backup_{ts}"
            shutil.copy2(self.input_file_path, backup_path)
            self.logger.info(f"Created backup: {backup_path.name}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None

    def _restore_from_backup(self, backup_path: Path):
        """Restores the input file from a backup."""
        try:
            shutil.copy2(backup_path, self.input_file_path)
            self.logger.warning(f"Restored file from backup: {backup_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to restore from backup {backup_path.name}: {e}")

    def extract_coefficients(self) -> Dict[str, float]:
        """
        从 input.w7x 文件中提取所有 RBC 和 ZBS 系数。
        """
        coefficients = {}
        try:
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 正则表达式匹配 RBC(m,n)=value 和 ZBS(m,n)=value
            pattern = re.compile(r"([RZ]BS\s*\(\s*[-]?\d+\s*,\s*[-]?\d+\s*\)\s*=\s*([-+]?\d+\.\d+e[-+]?\d+))", re.IGNORECASE)
            
            matches = pattern.finditer(content)
            
            for match in matches:
                full_match = match.group(1) # e.g., "RBC(0,0)=5.5586e+00"
                key_val_pair = full_match.split('=')
                key = key_val_pair[0].strip().replace(" ", "") # "RBC(0,0)"
                val_str = key_val_pair[1].strip() # "5.5586e+00"
                coefficients[key] = float(val_str)
                
            self.logger.info(f"Extracted {len(coefficients)} coefficients from {self.input_file_path.name}")
            return coefficients

        except Exception as e:
            self.logger.error(f"Error extracting coefficients: {e}", exc_info=True)
            return {}

    def replace_coefficients(self, new_coefficients: Dict[str, float]) -> bool:
        """
        使用新的系数值修改 input.w7x 文件。
        LLM 提供的 new_coefficients 字典中的值将替换文件中的值。
        """
        backup_path = self._create_backup()
        if not backup_path:
            return False

        try:
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified_content = content
            num_replaced = 0

            for key, value in new_coefficients.items():
                # 规范化 key，例如 "RBC( 1, 0)" -> "RBC(1,0)"
                norm_key = key.strip().replace(" ", "")
                
                # 匹配 "RBC(1,0)"
                match_key = re.match(r"([RZ]BS)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", norm_key)
                if not match_key:
                    self.logger.warning(f"Invalid coefficient key format: '{key}'. Skipping.")
                    continue
                
                coeff_type = re.escape(match_key.group(1)) # "RBC"
                m = re.escape(match_key.group(2)) # "1"
                n = re.escape(match_key.group(3)) # "0"
                
                # 构建正则表达式以查找 "RBC( 1, 0) = value"
                pattern = re.compile(
                    r"(" + coeff_type + r"\s*\(\s*" + m + r"\s*,\s*" + n + r"\s*\)\s*=\s*)"
                    r"([-+]?\d+\.\d+e[-+]?\d+)",
                    re.IGNORECASE
                )
                
                # 将新值格式化为科学计数法
                new_val_str = f"{value:e}"
                
                if pattern.search(modified_content):
                    modified_content = pattern.sub(r"\g<1>" + new_val_str, modified_content, count=1)
                    num_replaced += 1
                else:
                    self.logger.warning(f"Could not find pattern for key '{key}' in {self.input_file_path.name}.")
            
            with open(self.input_file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            self.logger.info(f"Successfully replaced {num_replaced} coefficients.")
            return True

        except Exception as e:
            self.logger.critical(f"Fatal error during coefficient replacement: {e}", exc_info=True)
            self._restore_from_backup(backup_path)
            return False