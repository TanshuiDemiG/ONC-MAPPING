import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse

def main():
    # 设置参数
    parser = argparse.ArgumentParser(description='YOLO石头检测推理脚本')
    parser.add_argument('--weights', type=str, 
                       default=r'D:\ANU\ONCMAPPING\ONC-MAPPING\runs\train\ONCMAPPING\weights\best.pt',
                       help='权重文件路径')
    parser.add_argument('--source', type=str, 
                       default=r'D:\ANU\ONCMAPPING\ONC-MAPPING\Code\ONC_Mapping_Stone-2\test\images',
                       help='输入图像路径或文件夹')
    parser.add_argument('--output', type=str, 
                       default=r'D:\ANU\ONCMAPPING\ONC-MAPPING\Code\results',
                       help='输出结果保存路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--imgsz', type=int, default=640, help='推理图像尺寸')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备 (cpu 或 0,1,2,3)')
    parser.add_argument('--save-txt', action='store_true', help='保存检测结果为txt文件')
    parser.add_argument('--save-conf', action='store_true', help='在txt文件中保存置信度')
    
    args = parser.parse_args()
    
    # 加载训练好的模型
    print(f"加载模型权重: {args.weights}")
    model = YOLO(args.weights)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行推理
    print(f"开始推理...")
    print(f"输入源: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"置信度阈值: {args.conf}")
    print(f"设备: {args.device}")
    
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True
    )
    
    print(f"推理完成！结果保存在: {args.output}")
    
    # 打印检测统计信息
    total_detections = 0
    for result in results:
        if result.boxes is not None:
            total_detections += len(result.boxes)
    
    print(f"总共检测到 {total_detections} 个石头目标")

def predict_single_image(image_path, weights_path=None, conf=0.25, save_result=True):
    """
    对单张图像进行预测的便捷函数
    
    Args:
        image_path: 图像路径
        weights_path: 权重文件路径，默认使用best.pt
        conf: 置信度阈值
        save_result: 是否保存结果图像
    
    Returns:
        检测结果
    """
    if weights_path is None:
        weights_path = r'D:\ANU\ONCMAPPING\ONC-MAPPING\runs\train\ONCMAPPING\weights\best.pt'
    
    # 加载模型
    model = YOLO(weights_path)
    
    # 进行预测
    results = model.predict(
        source=image_path,
        conf=conf,
        save=save_result,
        device='cpu'
    )
    
    return results

if __name__ == "__main__":
    main()