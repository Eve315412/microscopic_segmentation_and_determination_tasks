import os
import json
from datetime import datetime
from PIL import Image
import numpy as np


class HistoryService:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.base_dir)
        self.runs_dir = os.path.join(self.project_root, 'runs')
        os.makedirs(self.runs_dir, exist_ok=True)
        
        # 使用 data 目录下的 history.json 作为统一索引
        self.data_dir = os.path.join(self.project_root, 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.history_index_file = os.path.join(self.data_dir, 'history.json')

    def save_result(self, original_image_np, segmented_image_np, stats, weight_path=None, image_name="unknown", username="unknown"):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 修改文件夹命名规则：分割结果_时间
        out_dir = os.path.join(self.runs_dir, f'分割结果_{ts}')
        os.makedirs(out_dir, exist_ok=True)
        
        base_name = os.path.splitext(image_name)[0]
        orig_path = os.path.join(out_dir, f"{base_name}.png")
        seg_path = os.path.join(out_dir, f"{base_name}_result.png")
        Image.fromarray(original_image_np).save(orig_path)
        Image.fromarray(segmented_image_np).save(seg_path)
        
        # 记录数据
        record_data = {
            'image_name': image_name,
            'time': ts,
            'weight_name': os.path.basename(weight_path) if weight_path else None,
            'weight_path': weight_path,
            'username': username,
            'dir': out_dir,
            **(stats or {})
        }
        
        # 追加到统一索引文件
        self._append_to_index(record_data)
        
        return out_dir
        
    def _load_index(self):
        if not os.path.exists(self.history_index_file):
            return []
        try:
            with open(self.history_index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
            
    def _save_index(self, records):
        with open(self.history_index_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            
    def _append_to_index(self, record):
        records = self._load_index()
        records.append(record)
        self._save_index(records)

    def get_history(self, username=None):
        all_records = self._load_index()
        # 过滤并倒序
        records = [r for r in all_records if not username or r.get('username') == username]
        return records[::-1]

    def clear_history(self, username=None):
        """清空历史记录（删除文件夹并更新索引）"""
        import shutil
        all_records = self._load_index()
        
        new_records = []
        for rec in all_records:
            # 如果指定了用户名且匹配，或者没指定用户名（清空所有），则删除对应文件夹
            if not username or rec.get('username') == username:
                dir_path = rec.get('dir')
                if dir_path and os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        print(f"删除文件夹失败 {dir_path}: {e}")
            else:
                # 不属于当前用户的记录保留
                new_records.append(rec)
                
        self._save_index(new_records)
        return True
