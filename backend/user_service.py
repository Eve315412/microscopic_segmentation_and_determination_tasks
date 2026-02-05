import json
import os

class UserService:
    def __init__(self):
        # 创建 data 文件夹（如果不存在）
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.users_file = os.path.join(self.data_dir, 'users.json')
        self._load_users()

    def register(self, username, password):
        users = self._load_users()
        if username in users:
            return False, "用户已存在"
        
        users[username] = password  # In a real app, hash this password!
        self._save_users(users)
        return True, "注册成功"

    def login(self, username, password):
        users = self._load_users()
        if username not in users:
            return False, "用户不存在"
        if users[username] != password:
            return False, "密码错误"
        return True, "登录成功"

    def _load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_users(self, users):
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
