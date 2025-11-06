import sys
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.path}")
try:
    import cv2
    print(f"OpenCV版本: {cv2.__version__}")
except ImportError:
    print("无法导入OpenCV (cv2)")

try:
    import numpy
    print(f"NumPy版本: {numpy.__version__}")
except ImportError:
    print("无法导入NumPy")

try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
except ImportError:
    print("无法导入PyTorch")