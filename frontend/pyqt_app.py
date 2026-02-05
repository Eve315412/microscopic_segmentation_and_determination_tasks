import os
import sys
import csv
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                             QFileDialog, QVBoxLayout, QHBoxLayout, QStatusBar, QLineEdit, 
                             QStackedWidget, QFrame, QSplitter, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QGraphicsDropShadowEffect, QGraphicsOpacityEffect)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QEvent
import numpy as np
try:
    from .i18n import t, set_lang, get_lang
except Exception:
    from i18n import t, set_lang, get_lang

_lang = os.environ.get('APP_LANG')
for _a in sys.argv:
    if _a.startswith('--lang='):
        _lang = _a.split('=', 1)[1]
    elif _a == '--en':
        _lang = 'en'
if _lang:
    set_lang(_lang)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.segmentation_service import SegmentationService
from backend.analysis_service import AnalysisService
from backend.history_service import HistoryService
from backend.user_service import UserService


def np_to_qimage(arr):
    h, w, c = arr.shape
    bytes_per_line = c * w
    return QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)


class MacButtonStyle(QPushButton):
    def __init__(self, color_code, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color_code};
                border-radius: 6px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {color_code}DD;
            }}
        """)

class CrystalButton(QPushButton):
    """
    高拟真玻璃质感按钮
    特点：
    1. 复杂的线性渐变背景模拟光照反射
    2. 细微的半透明边框模拟玻璃厚度
    3. 投影增加立体感
    """
    def __init__(self, text, parent=None, color_tint="white", danger=False):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setFixedHeight(45)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # 投影效果 - 模拟环境光遮蔽
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 6)
        self.setGraphicsEffect(shadow)
        
        # 根据色调微调样式
        text_color = "#1C1C1E"
        if danger:
             # 红色危险按钮
             base_gradient = """
                qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 200, 200, 0.9),
                    stop:0.1 rgba(255, 200, 200, 0.5),
                    stop:0.49 rgba(255, 200, 200, 0.1),
                    stop:0.5 rgba(255, 59, 48, 0.1),
                    stop:1 rgba(255, 59, 48, 0.2))
            """
             border_color = "rgba(255, 59, 48, 0.4)"
             text_color = "#FF3B30"
        elif color_tint == "blue":
            # 蓝色玻璃
            base_gradient = """
                qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(200, 230, 255, 0.9),
                    stop:0.1 rgba(200, 230, 255, 0.5),
                    stop:0.49 rgba(200, 230, 255, 0.1),
                    stop:0.5 rgba(0, 122, 255, 0.1),
                    stop:1 rgba(0, 122, 255, 0.3))
            """
            border_color = "rgba(255, 255, 255, 0.6)"
            text_color = "#004080"
        else:
            # 透明/白色玻璃 (默认)
            base_gradient = """
                qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.95),
                    stop:0.1 rgba(255, 255, 255, 0.6),
                    stop:0.49 rgba(255, 255, 255, 0.1),
                    stop:0.5 rgba(255, 255, 255, 0.05),
                    stop:1 rgba(255, 255, 255, 0.2))
            """
            border_color = "rgba(255, 255, 255, 0.8)"

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {base_gradient};
                color: {text_color};
                border: 1px solid {border_color};
                border-bottom: 1px solid rgba(255, 255, 255, 0.5);
                border-radius: 12px; /* 长方形圆角 */
                padding: 5px 16px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 1.0),
                    stop:0.5 rgba(255, 255, 255, 0.4),
                    stop:1 rgba(255, 255, 255, 0.5));
                border: 1px solid white;
            }}
            QPushButton:pressed {{
                background-color: rgba(200, 200, 200, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                padding-top: 7px; /* 按下位移感 */
            }}
        """)

class ModernButton(QPushButton):
    def __init__(self, text, parent=None, primary=False, danger=False):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        font = QFont("Segoe UI", 10)
        if primary:
            font.setBold(True)
        self.setFont(font)
        self.primary = primary
        self.danger = danger
        self.update_style()
        
        # Glass effect shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def update_style(self):
        if self.danger:
            self.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.7);
                    color: #FF3B30;
                    border: 1px solid rgba(255, 59, 48, 0.3);
                    border-radius: 12px;
                    padding: 10px 20px;
                    text-align: center;
                }
                QPushButton:hover {
                    background-color: rgba(255, 59, 48, 0.1);
                    border: 1px solid rgba(255, 59, 48, 0.5);
                }
                QPushButton:pressed {
                    background-color: rgba(255, 59, 48, 0.2);
                }
            """)
        elif self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 122, 255, 0.85);
                    color: white;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 12px;
                    padding: 10px 20px;
                    text-align: center;
                }
                QPushButton:hover {
                    background-color: rgba(0, 122, 255, 0.95);
                }
                QPushButton:pressed {
                    background-color: rgba(0, 98, 204, 1.0);
                }
                QPushButton:disabled {
                    background-color: rgba(229, 229, 234, 0.5);
                    color: #8E8E93;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.6);
                    color: #1C1C1E;
                    border: 1px solid rgba(255, 255, 255, 0.4);
                    border-radius: 12px;
                    padding: 10px 20px;
                    text-align: center;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.8);
                    border: 1px solid rgba(0, 122, 255, 0.3);
                }
                QPushButton:pressed {
                    background-color: rgba(255, 255, 255, 0.9);
                }
            """)

class GlassCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.65);
                border: 1px solid rgba(255, 255, 255, 0.8);
                border-radius: 16px;
            }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 8)
        self.setGraphicsEffect(shadow)

class LoginWindow(QWidget):
    login_success = pyqtSignal(str)  # Emit username
    lang_change_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.user_service = UserService()
        self.setWindowTitle(f"{t('登录')} - {t('颗粒含量测量')}")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground) # 透明背景
        
        # Determine window size based on background image
        bg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images', '1.png')
        bg_path = bg_path.replace('\\', '/')
        
        window_width = 850
        window_height = 530
        
        if os.path.exists(bg_path):
            pixmap = QPixmap(bg_path)
            if not pixmap.isNull():
                # User requested not to change image size, so we use image size for window
                window_width = pixmap.width()
                window_height = pixmap.height()
        
        self.resize(window_width, window_height)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 根容器 Frame
        self.root_frame = QFrame()
        self.root_frame.setObjectName("RootFrame")
        
        bg_style = ""
        if os.path.exists(bg_path):
            bg_style = f"""
                border-image: url("{bg_path}") 0 0 0 0 stretch stretch;
                background-position: center;
                background-repeat: no-repeat;
            """
        else:
            bg_style = "background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F2F5, stop:1 #E6E9F0);"

        self.root_frame.setStyleSheet(f"""
            QFrame#RootFrame {{
                {bg_style}
                border-radius: 20px;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }}
        """)
        
        # Frame 内部布局 - 使用 QHBoxLayout 将内容分左右
        frame_layout = QHBoxLayout(self.root_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left Content Container (Login Module)
        left_content = QWidget()
        left_content.setFixedWidth(400 if get_lang().startswith('en') else 360)
        # Add semi-transparent background to left content
        left_content.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border-top-left-radius: 20px;
                border-bottom-left-radius: 20px;
            }
        """)
        
        left_layout = QVBoxLayout(left_content)
        left_layout.setAlignment(Qt.AlignCenter)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(30, 40, 30, 40)
        
        # Mac style dots (Top Left)
        dots_container = QWidget()
        dots_layout = QHBoxLayout(dots_container)
        dots_layout.setAlignment(Qt.AlignLeft)
        dots_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_close = MacButtonStyle("#FF5F57") # Red
        self.btn_minimize = MacButtonStyle("#FFBD2E") # Yellow
        self.btn_maximize = MacButtonStyle("#28C840") # Green
        
        self.btn_close.clicked.connect(self.close)
        self.btn_minimize.clicked.connect(self.showMinimized)
        self.btn_maximize.clicked.connect(self.toggle_maximize)

        dots_layout.addWidget(self.btn_close)
        dots_layout.addWidget(self.btn_minimize)
        dots_layout.addWidget(self.btn_maximize)
        
        # Add dots to top of left layout
        left_layout.addWidget(dots_container)
        left_layout.addSpacing(20)

        # Logo/Title Container
        header_container = QWidget()
        header_layout = QVBoxLayout(header_container)
        header_layout.setAlignment(Qt.AlignCenter)

        # School Logo
        self.logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images', '11.png')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            scaled_pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_pixmap)
        else:
            self.logo_label.setText("🏫")
            self.logo_label.setStyleSheet("font-size: 64px;")
        
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("margin-bottom: 10px; background: transparent;")
        header_layout.addWidget(self.logo_label)

        # Title
        title = QLabel(t("颗粒含量测量"))
        _tf = QFont("Segoe UI", 24, QFont.Bold)
        if get_lang().startswith('en'):
            _tf = QFont("Segoe UI", 22, QFont.Bold)
        title.setFont(_tf)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1C1C1E; margin-bottom: 20px; background: transparent;")
        title.setWordWrap(True)
        header_layout.addWidget(title)
        
        left_layout.addWidget(header_container)

        # Input Fields
        self.username = QLineEdit()
        self.username.setPlaceholderText(t("用户名"))
        self.username.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 8px;
                padding: 12px;
                color: #1C1C1E;
                font-size: 14px;
            }
            QLineEdit:focus {
                background-color: white;
                border: 1px solid #007AFF;
            }
        """)
        
        self.password = QLineEdit()
        self.password.setPlaceholderText(t("密码"))
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setStyleSheet(self.username.styleSheet())

        self.btn_login = CrystalButton(t("登录"), color_tint="blue")
        self.btn_login.clicked.connect(self.on_login)

        self.btn_register = CrystalButton(t("注册账号"))
        self.btn_register.clicked.connect(self.on_register)

        left_layout.addWidget(self.username)
        left_layout.addWidget(self.password)
        left_layout.addWidget(self.btn_login)
        left_layout.addWidget(self.btn_register)
        left_layout.addStretch()

        lang_container = QWidget()
        lang_layout = QHBoxLayout(lang_container)
        lang_layout.setContentsMargins(0, 0, 0, 0)
        lang_layout.setAlignment(Qt.AlignRight)
        lbl_lang = QLabel(t("语言"))
        lbl_lang.setStyleSheet("color: #8E8E93; background: transparent;")
        btn_cn = QPushButton(t("中文"))
        btn_en = QPushButton(t("英文"))
        for b in (btn_cn, btn_en):
            b.setFixedHeight(24)
            b.setStyleSheet("color: #8E8E93; background: transparent; border: none; padding: 0 6px;")
        btn_cn.clicked.connect(lambda: self.lang_change_signal.emit('zh'))
        btn_en.clicked.connect(lambda: self.lang_change_signal.emit('en'))
        lang_layout.addWidget(lbl_lang)
        lang_layout.addWidget(btn_cn)
        lang_layout.addWidget(btn_en)
        left_layout.addWidget(lang_container)
        
        frame_layout.addWidget(left_content)
        
        # Right Spacer (Push content to left)
        frame_layout.addStretch(1)
        
        main_layout.addWidget(self.root_frame)

    def on_login(self):
        username = self.username.text().strip()
        password = self.password.text().strip()
        if not username or not password:
            self.username.setPlaceholderText(t("请输入用户名"))
            return
            
        success, msg = self.user_service.login(username, password)
        if success:
            self.login_success.emit(username)
        else:
            QMessageBox.warning(self, t("登录失败"), t(msg))

    def on_register(self):
        username = self.username.text().strip()
        password = self.password.text().strip()
        if not username or not password:
            QMessageBox.warning(self, t("提示"), t("请输入用户名和密码"))
            return
            
        success, msg = self.user_service.register(username, password)
        if success:
            QMessageBox.information(self, t("成功"), t("注册成功，请直接登录"))
        else:
            QMessageBox.warning(self, t("注册失败"), t(msg))
            
    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
            
    # 支持拖动窗口
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

class MainWindow(QMainWindow):
    logout_signal = pyqtSignal()
    lang_change_signal = pyqtSignal(str)

    def __init__(self, username="admin"):
        super().__init__()
        self.username = username
        self.setWindowTitle(t('颗粒含量测量'))
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.setAttribute(Qt.WA_TranslucentBackground) # 透明背景
        self.resize(1280, 800)
        # self.setStyleSheet("background-color: #F0F2F5; color: #1C1C1E;") # 移至 CentralWidget

        # Services
        self.seg_service = SegmentationService()
        self.ana_service = AnalysisService()
        self.hist_service = HistoryService()
        
        # State
        self.original = None
        self.segmented = None
        self.stats = None
        self.weight_path = None
        self.current_image_path = None
        self._segment_run_id = 0
        self._current_segment_run_id = None
        self._last_saved_signature = None
        self._last_saved_dir = None

        # Load Default Model
        default_model = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained', 'unet.pth')
        if os.path.exists(default_model):
            self.seg_service.set_model(default_model)
            self.weight_path = default_model
        
        self.init_ui()
        
        if self.weight_path:
             self.update_model_info(default_model)
        else:
             self.update_model_info(None)

    def init_ui(self):
        # Main Layout
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        
        central_widget.setStyleSheet("""
            QWidget#CentralWidget {
                background-color: #F0F2F5;
                color: #1C1C1E;
                border-radius: 20px;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
        """)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar with Glass effect
        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.85);
                border-right: 1px solid rgba(0, 0, 0, 0.05);
                border-top-left-radius: 20px;
                border-bottom-left-radius: 20px;
            }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(15)

        # Mac Buttons Area
        mac_layout = QHBoxLayout()
        mac_layout.setAlignment(Qt.AlignLeft)
        mac_layout.setSpacing(8)
        
        self.btn_close = MacButtonStyle("#FF5F57")
        self.btn_minimize = MacButtonStyle("#FFBD2E")
        self.btn_maximize = MacButtonStyle("#28C840")
        
        self.btn_close.clicked.connect(self.close)
        self.btn_minimize.clicked.connect(self.showMinimized)
        self.btn_maximize.clicked.connect(self.toggle_maximize)

        mac_layout.addWidget(self.btn_close)
        mac_layout.addWidget(self.btn_minimize)
        mac_layout.addWidget(self.btn_maximize)
        sidebar_layout.addLayout(mac_layout)
        
        sidebar_layout.addSpacing(20)

        # School Logo & Name (Removed Logo)
        logo_layout = QHBoxLayout()
        logo_layout.setAlignment(Qt.AlignCenter)
        
        logo_text = QLabel(t("颗粒含量测量"))
        _lf = QFont("Segoe UI", 16, QFont.Bold)
        if get_lang().startswith('en'):
            _lf = QFont("Segoe UI", 15, QFont.Bold)
        logo_text.setFont(_lf)
        logo_text.setStyleSheet("color: #1C1C1E; background: transparent; border: none;")
        logo_text.setAlignment(Qt.AlignCenter)
        logo_text.setWordWrap(True)
        logo_layout.addWidget(logo_text)
        sidebar_layout.addLayout(logo_layout)

        sidebar_layout.addSpacing(20)

        # New Chat Button (Load Image)
        btn_load = CrystalButton(t("+ 加载新图像"))
        btn_load.clicked.connect(self.on_load)
        sidebar_layout.addWidget(btn_load)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: rgba(0, 0, 0, 0.05); margin: 15px 0;")
        sidebar_layout.addWidget(line)

        # Menu Items
        self.btn_segment = CrystalButton(t("⚡ 执行分割"), color_tint="blue")
        self.btn_segment.clicked.connect(self.on_segment)
        sidebar_layout.addWidget(self.btn_segment)

        self.btn_save = CrystalButton(t("💾 保存结果"))
        self.btn_save.clicked.connect(self.on_save)
        sidebar_layout.addWidget(self.btn_save)

        self.btn_weight = CrystalButton(t("🔧 更换权重"))
        self.btn_weight.clicked.connect(self.on_weight)
        sidebar_layout.addWidget(self.btn_weight)

        sidebar_layout.addStretch()
        
        # History Panel Toggle
        self.btn_history = CrystalButton(t("📜 历史记录"))
        self.btn_history.setFont(QFont("Segoe UI", 9))
        self.btn_history.setFixedHeight(40)
        self.btn_history.clicked.connect(self.toggle_history)
        sidebar_layout.addWidget(self.btn_history)
        
        # Status Label (Moved from bottom StatusBar)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #8E8E93; font-size: 12px; margin-top: 5px; background: transparent;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        sidebar_layout.addWidget(self.status_label)
        
        main_layout.addWidget(sidebar)

        # Right Side Area (Header + Content)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Header Area
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(20, 20, 30, 0)
        header_layout.setAlignment(Qt.AlignRight)
        header_layout.setSpacing(10)

        # User Info
        user_label = QLabel(f"{t('当前用户')}: {self.username}")
        user_label.setStyleSheet("color: #636366; font-size: 14px; margin-right: 10px; background: transparent;")
        header_layout.addWidget(user_label)

        self.btn_switch_account = CrystalButton(t("更换账号"))
        self.btn_switch_account.setFixedSize(160, 40)
        self.btn_switch_account.clicked.connect(self.on_logout)
        
        self.btn_logout = CrystalButton(t("退出登录"), danger=True)
        self.btn_logout.setFixedSize(160, 40)
        self.btn_logout.clicked.connect(self.on_logout)

        header_layout.addWidget(self.btn_switch_account)
        header_layout.addWidget(self.btn_logout)

        right_layout.addWidget(header_container)

        # Content Area
        self.content_stack = QStackedWidget()
        
        # 1. Main Analysis View
        analysis_view = QWidget()
        analysis_layout = QVBoxLayout(analysis_view)
        analysis_layout.setContentsMargins(30, 30, 30, 30)
        analysis_layout.setSpacing(25)

        # Image Display Area (Glass)
        self.image_area = GlassCard()
        # 在图像显示模块背景居中放置半透明 logo（不影响前景内容）
        _logo_mid = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images', 'logo.png')
        _logo_mid = _logo_mid.replace('\\', '/')
        if os.path.exists(_logo_mid):
            _bg_logo = QLabel(self.image_area)
            _bg_logo.setObjectName("ImageAreaBgLogo")
            _bg_logo.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            _bg_logo.setAlignment(Qt.AlignCenter)
            _pix = QPixmap(_logo_mid)
            _bg_logo.setPixmap(_pix)
            _bg_logo.setGeometry(0, 0, self.image_area.width(), self.image_area.height())
            _bg_logo.lower()  # 置于背景层
            _opacity = QGraphicsOpacityEffect(_bg_logo)
            _opacity.setOpacity(0.25)  # 透明度可按需调整(0~1)
            _bg_logo.setGraphicsEffect(_opacity)
            self._bg_logo = _bg_logo
            # 监听尺寸变化，保持居中与覆盖区域
            self.image_area.installEventFilter(self)
        img_layout = QHBoxLayout(self.image_area)
        img_layout.setContentsMargins(20, 20, 20, 20)
        
        self.orig_label = QLabel(t("请加载图像"))
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setStyleSheet("color: #8E8E93; font-size: 16px; border: none; background: transparent;")
        
        self.seg_label = QLabel(t("等待分割..."))
        self.seg_label.setAlignment(Qt.AlignCenter)
        self.seg_label.setStyleSheet("color: #8E8E93; font-size: 16px; border: none; background: transparent;")

        img_layout.addWidget(self.orig_label, 1)
        
        # Divider between images
        img_sep = QFrame()
        img_sep.setFrameShape(QFrame.VLine)
        img_sep.setStyleSheet("background-color: rgba(0, 0, 0, 0.1);")
        img_layout.addWidget(img_sep)
        
        img_layout.addWidget(self.seg_label, 1)

        analysis_layout.addWidget(self.image_area, 3)

        # Stats Area (Glass Cards)
        stats_container = QWidget()
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setSpacing(20)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        
        self.stat_cards = []
        labels = [t("颗粒面积"), t("轨迹面积"), t("颗粒占比")]
        for label in labels:
            card = GlassCard()
            card.setFixedHeight(120)
            card_layout = QVBoxLayout(card)
            card_layout.setAlignment(Qt.AlignCenter)
            
            lbl_title = QLabel(label)
            lbl_title.setStyleSheet("color: #636366; font-size: 14px; font-weight: 500; border: none; background: transparent;")
            lbl_title.setAlignment(Qt.AlignCenter)
            
            lbl_value = QLabel("-")
            lbl_value.setStyleSheet("color: #000000; font-size: 28px; font-weight: bold; border: none; background: transparent;")
            lbl_value.setAlignment(Qt.AlignCenter)
            
            card_layout.addWidget(lbl_title)
            card_layout.addWidget(lbl_value)
            stats_layout.addWidget(card)
            self.stat_cards.append(lbl_value)

        analysis_layout.addWidget(stats_container, 1)
        
        self.content_stack.addWidget(analysis_view)

        # 2. History View
        history_view = QWidget()
        history_layout = QVBoxLayout(history_view)
        history_layout.setContentsMargins(30, 30, 30, 30)
        history_layout.setSpacing(20)
        
        hist_header = QHBoxLayout()
        btn_back = CrystalButton(t("← 返回分析"))
        btn_back.setFixedWidth(140)
        btn_back.clicked.connect(self.show_analysis)
        hist_header.addWidget(btn_back)
        
        hist_title = QLabel(t("历史记录"))
        hist_title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        hist_header.addWidget(hist_title)
        hist_header.addStretch()
        
        btn_clear = CrystalButton(t("清空记录"), danger=True)
        btn_clear.clicked.connect(self.clear_history)
        hist_header.addWidget(btn_clear)
        
        btn_export = CrystalButton(t("导出 Excel"), color_tint="blue")
        btn_export.clicked.connect(self.export_history)
        hist_header.addWidget(btn_export)
        
        history_layout.addLayout(hist_header)
        
        # Glass Table
        self.hist_table = QTableWidget()
        self.hist_table.setColumnCount(8)
        self.hist_table.setHorizontalHeaderLabels([t("时间"), t("用户"), t("图像名称"), t("颗粒数量"), t("颗粒面积"), t("轨迹面积"), t("颗粒占比"), t("权重模型")])
        self.hist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hist_table.setAlternatingRowColors(True)
        self.hist_table.setShowGrid(False)
        self.hist_table.setStyleSheet("""
            QTableWidget {
                background-color: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.8);
                border-radius: 12px;
                gridline-color: transparent;
                selection-background-color: rgba(0, 122, 255, 0.1);
                selection-color: black;
            }
            QHeaderView::section {
                background-color: rgba(255, 255, 255, 0.9);
                padding: 12px;
                border: none;
                border-bottom: 1px solid rgba(0, 0, 0, 0.05);
                font-weight: bold;
                color: #1C1C1E;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid rgba(0, 0, 0, 0.03);
            }
        """)
        # Add shadow to table
        table_shadow = QGraphicsDropShadowEffect(self.hist_table)
        table_shadow.setBlurRadius(15)
        table_shadow.setColor(QColor(0, 0, 0, 15))
        table_shadow.setOffset(0, 4)
        self.hist_table.setGraphicsEffect(table_shadow)
        
        history_layout.addWidget(self.hist_table)
        
        self.content_stack.addWidget(history_view)

        right_layout.addWidget(self.content_stack)

        # Footer Area
        footer_container = QWidget()
        footer_container.setObjectName("FooterContainer")
        footer_container.setFixedHeight(100) # Further decreased height
        footer_container.setAttribute(Qt.WA_StyledBackground, True)
        footer_container.setStyleSheet("""
            QWidget#FooterContainer {
                background-color: rgba(255, 255, 255, 0.85);
                border-bottom-right-radius: 20px;
            }
        """)
        
        footer_layout = QHBoxLayout(footer_container)
        footer_layout.setContentsMargins(40, 5, 40, 5) # Adjusted margins
        footer_layout.setSpacing(20)
        
        # Footer Text (Left Side, Multi-line)
        footer_info = QLabel(
            "南京工程学院 机械工程学院\n"
            "School of Mechanical Engineering, Nanjing Institute of Technology\n"
            "© 2026 All Rights Reserved"
        )
        footer_info.setFont(QFont("Segoe UI", 10)) # Increased font size
        footer_info.setStyleSheet("color: #48484A; background: transparent; line-height: 1.5;")
        footer_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        footer_layout.addWidget(footer_info)

        # (Removed) Language toggle in main UI per requirement

        footer_layout.addStretch()
        
        # Footer Logo (Right Side, Larger)
        footer_logo = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images', 'logo.png')
        if os.path.exists(logo_path):
             pix = QPixmap(logo_path)
             footer_logo.setPixmap(pix.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)) # Larger logo
             footer_logo.setStyleSheet("background: transparent;")
        
        footer_layout.addWidget(footer_logo)
        
        right_layout.addWidget(footer_container)

        main_layout.addWidget(right_widget)

        # Removed QStatusBar


    def show_message(self, message, timeout=0):
        self.status_label.setText(message)
        # 简单实现，暂不支持自动清除timeout，如果需要可以使用QTimer
        if timeout > 0:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(timeout, lambda: self.status_label.setText(""))

    def update_model_info(self, path):
        if path:
            name = os.path.basename(path)
            self.show_message(f"{t('已加载模型')}: {name}", 5000)
        else:
            self.show_message(t('未加载模型'), 5000)

    def toggle_history(self):
        self.refresh_history()
        self.content_stack.setCurrentIndex(1)

    def show_analysis(self):
        self.content_stack.setCurrentIndex(0)

    def refresh_history(self):
        records = self.hist_service.get_history(self.username)
        self.hist_table.setRowCount(len(records))
        for i, rec in enumerate(records):
            self.hist_table.setItem(i, 0, QTableWidgetItem(rec.get('time', '-')))
            self.hist_table.setItem(i, 1, QTableWidgetItem(rec.get('username', 'unknown')))
            self.hist_table.setItem(i, 2, QTableWidgetItem(rec.get('image_name', 'unknown')))
            self.hist_table.setItem(i, 3, QTableWidgetItem(str(rec.get('particle_count', '-'))))
            self.hist_table.setItem(i, 4, QTableWidgetItem(str(rec.get('particle_area', '-'))))
            self.hist_table.setItem(i, 5, QTableWidgetItem(str(rec.get('track_area', '-'))))
            ratio = rec.get('particle_ratio', 0)
            self.hist_table.setItem(i, 6, QTableWidgetItem(f"{ratio*100:.2f}%"))
            weight_path = rec.get('weight_path')
            weight_name = rec.get('weight_name') or (os.path.basename(weight_path) if weight_path else '-')
            weight_item = QTableWidgetItem(weight_name)
            if weight_path:
                weight_item.setToolTip(weight_path)
            self.hist_table.setItem(i, 7, weight_item)

    def export_history(self):
        try:
            import pandas as pd
        except ImportError:
            self.show_message(t("缺少 pandas 库，无法导出 Excel。请安装: pip install pandas openpyxl"), 5000)
            return

        records = self.hist_service.get_history(self.username)
        if not records:
            self.show_message(t("没有可导出的历史记录"), 3000)
            return

        # 获取历史记录的基础目录 (runs 目录)
        runs_dir = self.hist_service.runs_dir
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

        # 生成导出文件名：历史记录_时间.xlsx
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{t('历史记录')}_{timestamp}.xlsx"
        filepath = os.path.join(runs_dir, filename)

        try:
            # 准备数据
            data_list = []
            for rec in records:
                weight_path = rec.get('weight_path')
                weight_name = rec.get('weight_name') or (os.path.basename(weight_path) if weight_path else '-')
                data_list.append({
                    t('时间'): rec.get('time', '-'),
                    t('用户名'): rec.get('username', 'unknown'),
                    t('图像名称'): rec.get('image_name', 'unknown'),
                    t('颗粒数量'): rec.get('particle_count', '-'),
                    t('颗粒面积'): rec.get('particle_area', '-'),
                    t('轨迹面积'): rec.get('track_area', '-'),
                    t('颗粒占比'): f"{rec.get('particle_ratio', 0)*100:.2f}%",
                    t('权重模型'): weight_name,
                    t('文件夹路径'): rec.get('dir', '-')
                })
            
            # 创建 DataFrame 并保存为 Excel
            df = pd.DataFrame(data_list)
            df.to_excel(filepath, index=False, engine='openpyxl')
            
            self.show_message(f"{t('历史记录已导出到')}: {filepath}", 5000)
            
        except Exception as e:
            self.show_message(f"{t('导出失败')}: {str(e)}", 5000)

    def clear_history(self):
        reply = QMessageBox.question(self, t('确认清空'), 
                                   t("确定要清空您的历史记录吗？此操作不可恢复。"),
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if self.hist_service.clear_history(self.username):
                self.refresh_history()
                self.show_message(t("历史记录已清空"), 3000)
            else:
                self.show_message(t("清空失败"), 3000)

    def update_stats(self, stats):
        if not stats:
            return
        # 更新三个卡片：颗粒面积，轨迹面积，颗粒占比
        self.stat_cards[0].setText(str(stats['particle_area']))
        self.stat_cards[1].setText(str(stats['track_area']))
        self.stat_cards[2].setText(f"{stats['particle_ratio']*100:.2f}%")

    def eventFilter(self, obj, event):
        if obj is getattr(self, 'image_area', None) and event.type() == QEvent.Resize:
            bg = getattr(self, '_bg_logo', None)
            if bg:
                bg.setGeometry(0, 0, obj.width(), obj.height())
        return super().eventFilter(obj, event)

    def on_weight(self):
        path, _ = QFileDialog.getOpenFileName(self, t('选择权重文件'), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'unet_vgg16'), t('PyTorch 权重 (*.pth *.pt)'))
        if path:
            self.seg_service.set_model(path)
            self.weight_path = path
            self.update_model_info(path)

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, t('选择图像'), os.path.dirname(os.path.abspath(__file__)), 'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if path:
            self.current_image_path = path
            img = QImage(path)
            if img.isNull():
                return
            img = img.convertToFormat(QImage.Format_RGB888)
            w, h = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.array(ptr, dtype=np.uint8).reshape(h, w, 3)
            self.original = arr
            pix = QPixmap.fromImage(img)
            self.orig_label.setPixmap(pix.scaled(self.orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.show_message(f"{t('已加载图像')}: {os.path.basename(path)}", 5000)
            
            # Reset results
            self.seg_label.setText(t("等待分割..."))
            self.seg_label.setPixmap(QPixmap())
            for card in self.stat_cards:
                card.setText("-")
            self.segmented = None
            self.stats = None
            self._current_segment_run_id = None

    def on_segment(self):
        if self.original is None:
            self.show_message(t('请先加载图像'), 3000)
            return
        if self.seg_service.model is None:
            self.show_message(t('未加载模型权重，请先更换权重'), 3000)
            return
            
        self.show_message(t('正在分割...'), 0)
        QApplication.processEvents()
        
        try:
            mask = self.seg_service.segment_mask(self.original)
            rgb = self.seg_service.segment_rgb(self.original)
            stats = self.ana_service.compute_stats(mask)

            self.segmented = rgb
            self.stats = stats

            qi = np_to_qimage(rgb)
            self.seg_label.setPixmap(QPixmap.fromImage(qi).scaled(self.seg_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.update_stats(stats)
            self._segment_run_id += 1
            self._current_segment_run_id = self._segment_run_id

            image_name = os.path.basename(self.current_image_path) if self.current_image_path else "unknown"
            signature = (self._current_segment_run_id, self.current_image_path, self.weight_path, image_name)
            try:
                out = self.hist_service.save_result(
                    self.original,
                    self.segmented,
                    self.stats,
                    self.weight_path,
                    image_name,
                    self.username
                )
                self._last_saved_signature = signature
                self._last_saved_dir = out
                self.refresh_history()
                self.show_message(f"{t('结果已保存到')}: {out}", 5000)
            except Exception as e:
                self.show_message(f"{t('分割完成')}，{t('保存失败')}: {str(e)}", 5000)
        except Exception as e:
            self.segmented = None
            self.stats = None
            self._current_segment_run_id = None
            self.show_message(f"{t('分割失败')}: {str(e)}", 5000)

    def on_save(self):
        image_name = os.path.basename(self.current_image_path) if self.current_image_path else "unknown"
        signature = (self._current_segment_run_id, self.current_image_path, self.weight_path, image_name)
        if getattr(self, "_last_saved_signature", None) == signature and getattr(self, "_last_saved_dir", None):
            self.show_message(f"{t('结果已保存到')}: {self._last_saved_dir}", 5000)
            return
        if self.segmented is None or self.original is None or self.stats is None or self._current_segment_run_id is None:
            self.show_message(t("请先执行分割"), 3000)
            return

        out = self.hist_service.save_result(
            self.original,
            self.segmented,
            self.stats,
            self.weight_path,
            image_name,
            self.username
        )
        self._last_saved_signature = signature
        self._last_saved_dir = out
        self.refresh_history()
        self.show_message(f"{t('结果已保存到')}: {out}", 5000)

    def on_logout(self):
        self.logout_signal.emit()
        
    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # 支持拖动窗口
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def on_change_lang(self, lang):
        self.lang_change_signal.emit(lang)


class AppController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setFont(QFont("Segoe UI", 9))
        
        self.login_window = LoginWindow()
        self.main_window = None
        
        self.login_window.login_success.connect(self.show_main)
        self.login_window.lang_change_signal.connect(self.change_lang_login)
        self.login_window.show()

    def show_main(self, username):
        self.login_window.close()
        self.main_window = MainWindow(username)
        self.main_window.logout_signal.connect(self.logout)
        self.main_window.lang_change_signal.connect(self.change_lang_main)
        self.main_window.show()

    def logout(self):
        if self.main_window:
            self.main_window.close()
            self.main_window = None
        self.login_window.username.clear()
        self.login_window.password.clear()
        self.login_window.showNormal()

    def change_lang_login(self, lang):
        set_lang(lang)
        # Recreate login window with new language
        if self.login_window:
            self.login_window.close()
        self.login_window = LoginWindow()
        self.login_window.login_success.connect(self.show_main)
        self.login_window.lang_change_signal.connect(self.change_lang_login)
        self.login_window.show()

    def change_lang_main(self, lang):
        set_lang(lang)
        # Recreate main window preserving username
        if self.main_window:
            username = self.main_window.username
            self.main_window.close()
            self.main_window = MainWindow(username)
            self.main_window.logout_signal.connect(self.logout)
            self.main_window.lang_change_signal.connect(self.change_lang_main)
            self.main_window.show()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    controller = AppController()
    controller.run()
