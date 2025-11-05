# problem/sacs/sacs_file_modifier.py (V2 - 增加对目标文件的支持)
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

class SacsFileModifier:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.input_file = self.project_path / "sacinp.demo13"
        self.backup_dir = self.project_path / "backups"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.backup_dir.mkdir(exist_ok=True)
        if not self.input_file.exists():
            raise FileNotFoundError(f"SACS input file not found: {self.input_file}")

    def _create_backup(self) -> Optional[Path]:
        """Creates a backup of the current input file."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"sacinp_pre_eval_{ts}.demo13"
            shutil.copy2(self.input_file, backup_path)
            self.logger.info(f"Created backup: {backup_path.name}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None

    def _restore_from_backup(self, backup_path: Path):
        """Restores the input file from a backup."""
        try:
            shutil.copy2(backup_path, self.input_file)
            self.logger.warning(f"Restored file from backup: {backup_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to restore from backup {backup_path.name}: {e}")

    def extract_code_blocks(self, block_prefixes: List[str]) -> Dict[str, str]:
        code_blocks = {}
        try:
            with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            for prefix in block_prefixes:
                found = False
                for line in lines:
                    if line.strip().startswith(prefix):
                        key = prefix.replace(" ", "_")
                        code_blocks[key] = line.rstrip('\n')
                        found = True
                        break
                if not found:
                    self.logger.warning(f"Could not find a unique code block for prefix: '{prefix}'")
        except Exception as e:
            self.logger.error(f"Error extracting code blocks: {e}")
        return code_blocks

    def replace_code_blocks(self, new_code_blocks: Dict[str, str], target_file: Optional[Path] = None) -> bool:
        """
        Replaces entire lines in a SACS file with new code blocks.
        
        Args:
            new_code_blocks: A dictionary of code blocks to replace.
            target_file (Optional): If provided, modifications are written to this file.
                                    If None, the default sacinp.demo13 is modified in place.
        """
        file_to_modify = target_file if target_file is not None else self.input_file
        is_in_place_modification = (target_file is None)

        if not file_to_modify.exists():
            self.logger.error(f"Target file for modification does not exist: {file_to_modify}")
            return False

        backup_path = None
        if is_in_place_modification:
            backup_path = self._create_backup()
            if not backup_path:
                return False

        try:
            with open(file_to_modify, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            lines_replaced = 0
            for identifier, new_line in new_code_blocks.items():
                parts = identifier.split('_')
                if len(parts) != 2:
                    self.logger.warning(f"Invalid identifier format '{identifier}'. Skipping.")
                    continue
                
                keyword, id_val = parts
                pattern = re.compile(r"^\s*" + re.escape(keyword) + r"\s+" + re.escape(id_val))
                
                line_found_and_replaced = False
                for i, line in enumerate(lines):
                    if pattern.search(line):
                        self.logger.info(f"Replacing block '{identifier}' in {file_to_modify.name}:\n  OLD: {line.strip()}\n  NEW: {new_line.strip()}")
                        lines[i] = new_line + '\n'
                        lines_replaced += 1
                        line_found_and_replaced = True
                        break

                if not line_found_and_replaced:
                    self.logger.warning(f"Identifier '{identifier}' from LLM not found in SACS file. Skipping.")

            with open(file_to_modify, 'w', encoding='utf-8', errors='ignore') as f:
                f.writelines(lines)

            if lines_replaced > 0:
                self.logger.info(f"Successfully replaced {lines_replaced} code blocks in {file_to_modify.name}.")
            return True

        except Exception as e:
            self.logger.critical(f"Fatal error during code block replacement on {file_to_modify.name}: {e}")
            if is_in_place_modification and backup_path:
                self._restore_from_backup(backup_path)
            return False