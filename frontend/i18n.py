import os

LANG = os.environ.get("APP_LANG", "zh")

ZH = {
    "登录": "Login",
    
    "颗粒含量测量": "Particle Content Determination",
    "用户名": "Username",
    "密码": "Password",
    "注册账号": "Register",
    "请输入用户名": "Please enter username",
    "请输入用户名和密码": "Please enter username and password",
    "登录失败": "Login Failed",
    "提示": "Notice",
    "成功": "Success",
    "注册成功，请直接登录": "Registration successful, please login directly",
    "注册失败": "Registration Failed",
    "+ 加载新图像": "+ Load Image",
    "⚡ 执行分割": "⚡ Segment",
    "💾 保存结果": "💾 Save Results",
    "🔧 更换权重": "🔧 Change Weights",
    "📜 历史记录": "📜 History",
    "当前用户": "Current User",
    "更换账号": "Switch Account",
    "退出登录": "Logout",
    "请加载图像": "Please load an image",
    "等待分割...": "Waiting for segmentation...",
    "颗粒面积": "Particle Area",
    "轨迹面积": "Track Area",
    "颗粒占比": "Particle Ratio",
    "已加载模型": "Model loaded",
    "未加载模型": "Model not loaded",
    "已加载图像": "Image loaded",
    "请先加载图像": "Please load an image first",
    "未加载模型权重，请先更换权重": "No model weights loaded, please change weights first",
    "正在分割...": "Segmenting...",
    "分割完成": "Segmentation completed",
    "分割失败": "Segmentation failed",
    "没有可保存的结果": "No results to save",
    "结果已保存到": "Results saved to",
    "选择权重文件": "Select weight file",
    "PyTorch 权重 (*.pth *.pt)": "PyTorch weights (*.pth *.pt)",
    "选择图像": "Select image",
    "← 返回分析": "← Back to Analysis",
    "历史记录": "History",
    "清空记录": "Clear Records",
    "导出 Excel": "Export Excel",
    "时间": "Time",
    "用户": "User",
    "图像名称": "Image Name",
    "颗粒数量": "Particle Count",
    "文件夹路径": "Folder Path",
    "权重模型": "Weight Model",
    "缺少 pandas 库，无法导出 Excel。请安装: pip install pandas openpyxl": "Missing pandas library, cannot export Excel. Please install: pip install pandas openpyxl",
    "没有可导出的历史记录": "No history to export",
    "历史记录已导出到": "History exported to",
    "导出失败": "Export failed",
    "确认清空": "Confirm Clear",
    "确定要清空您的历史记录吗？此操作不可恢复。": "Are you sure to clear your history? This action cannot be undone.",
    "历史记录已清空": "History cleared",
    "清空失败": "Clear failed",
    "分割结果": "Segmentation_Result"
}

# Language names and toggles
ZH.update({
    "语言": "Language",
    "中文": "Chinese",
    "英文": "English"
})

SERVICE = {
    "用户已存在": "User already exists",
    "注册成功": "Registration successful",
    "用户不存在": "User does not exist",
    "密码错误": "Incorrect password",
    "登录成功": "Login successful"
}

def set_lang(lang):
    global LANG
    LANG = lang

def t(s):
    if LANG.startswith("en"):
        if s in ZH:
            return ZH[s]
        if s in SERVICE:
            return SERVICE[s]
    return s

def get_lang():
    return LANG
