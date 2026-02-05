# 数据集配置
DATASET_PATH = "lung_split_dataset"  # 数据集相对路径
IMAGE_SUFFIX = "tif"  # 图像文件后缀（不带点号，方便程序处理）
LABEL_SUFFIX = "tif"  # 标签文件后缀（不带点号，方便程序处理）

# 训练配置
EPOCHS = 20  # 训练轮数
BATCH_SIZE = 4  # 批次大小
LEARNING_RATE = 1e-4  # 学习率
NUM_CLASSES = 3  # 类别数量（根据医学分割任务的标签数量设置）