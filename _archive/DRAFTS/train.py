import os
from ultralytics import YOLO

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "TRUE"

def main():
    # 加载预训练模型
    model = YOLO('yolo11n.pt')
    
    # 训练模型
    results = model.train(
        data=r'D:\ANU\ONCMAPPING\ONC-MAPPING\Code\ONC_Mapping_Stone-2\data.yaml',      # 数据配置文件路径
        epochs=100,                 # 训练轮数
        imgsz=512,                  # 图像尺寸
        batch=16,                   # 批次大小
        device='cpu',               # 使用CPU（如果没有GPU）
        project='runs/train',       # 项目目录
        name='ONCMAPPING',      # 实验名称
    )

if __name__ == "__main__":
    main()