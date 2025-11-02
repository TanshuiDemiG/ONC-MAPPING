import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import json
import argparse

def visualize_predictions(image_path, weights_path, output_dir=None, conf=0.25):
    """
    可视化单张图像的预测结果
    
    Args:
        image_path: 输入图像路径
        weights_path: 模型权重路径
        output_dir: 输出目录
        conf: 置信度阈值
    """
    # 路径与文件存在性检查
    image_path = str(Path(image_path).expanduser())
    weights_path = str(Path(weights_path).expanduser())
    if not Path(image_path).exists():
        print(f"图像文件不存在，请检查路径: {Path(image_path).resolve()}")
        return None
    if not Path(weights_path).exists():
        print(f"模型权重不存在，请检查路径: {Path(weights_path).resolve()}")
        return None

    # 加载模型
    model = YOLO(weights_path)
    
    # 读取图像
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像文件，请检查格式或权限: {Path(image_path).resolve()}")
            return None
    except Exception as e:
        print(f"读取图像时发生错误: {e}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 进行预测
    results = model.predict(source=image_path, conf=conf, device='cpu')
    
    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 显示原图
    axes[0].imshow(image_rgb)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示检测结果
    result_image = image_rgb.copy()
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for i, (box, conf_score) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box.astype(int)
            
            # 添加置信度标签
            # label = f'Stone: {conf_score:.3f}'
            # label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 根据置信度设置颜色
            if conf_score >= 0.8:
                box_color = (0, 255, 0)  # 绿色 - 高置信度
                text_bg_color = (0, 255, 0)
            elif conf_score >= 0.5:
                box_color = (0, 255, 255)  # 黄色 - 中等置信度
                text_bg_color = (0, 255, 255)
            else:
                box_color = (0, 165, 255)  # 橙色 - 低置信度
                text_bg_color = (0, 165, 255)
            
            # 绘制边界框（使用置信度颜色）
            cv2.rectangle(result_image, (x1, y1), (x2, y2), box_color, 1)
            
            # # 绘制标签背景
            # cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
            #              (x1 + label_size[0] + 5, y1), text_bg_color, -1)
            
            # # 添加标签文字
            # cv2.putText(result_image, label, (x1 + 2, y1 - 5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    axes[1].imshow(result_image)
    detection_count = len(results[0].boxes) if results[0].boxes is not None else 0
    axes[1].set_title(f'检测结果 (检测到 {detection_count} 个石头)\n颜色说明: 绿色(≥0.8) 黄色(≥0.5) 橙色(<0.5)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        plt.savefig(output_path / f'{image_name}_visualization.png', dpi=300, bbox_inches='tight')
        print(f"可视化结果保存到: {output_path / f'{image_name}_visualization.png'}")
    
    plt.show()
    
    return results

def batch_analyze(image_folder, weights_path, output_dir, conf=0.25):
    """
    批量分析图像文件夹中的所有图像
    
    Args:
        image_folder: 图像文件夹路径
        weights_path: 模型权重路径
        output_dir: 输出目录
        conf: 置信度阈值
    """
    # 加载模型
    model = YOLO(weights_path)
    
    # 获取所有图像文件
    image_folder = Path(image_folder).expanduser()
    if not image_folder.exists():
        print(f"图像文件夹不存在，请检查路径: {image_folder.resolve()}")
        return
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f'*{ext}')))
        image_files.extend(list(image_folder.glob(f'*{ext.upper()}')))
    
    if len(image_files) == 0:
        print(f"在文件夹中未找到图像文件: {image_folder.resolve()} (支持扩展名: {image_extensions})")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 分析结果统计
    analysis_results = {
        'total_images': len(image_files),
        'total_detections': 0,
        'images_with_detections': 0,
        'detection_details': []
    }
    
    # 批量处理
    for i, image_file in enumerate(image_files):
        print(f"处理 {i+1}/{len(image_files)}: {image_file.name}")
        
        # 进行预测
        results = model.predict(source=str(image_file), conf=conf, device='cpu')
        
        # 统计检测结果
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        analysis_results['total_detections'] += num_detections
        
        if num_detections > 0:
            analysis_results['images_with_detections'] += 1
        
        # 保存详细信息
        image_info = {
            'image_name': image_file.name,
            'detections': num_detections,
            'confidences': []
        }
        
        if results[0].boxes is not None:
            confidences = results[0].boxes.conf.cpu().numpy().tolist()
            image_info['confidences'] = confidences
        
        analysis_results['detection_details'].append(image_info)
    
    # 保存分析结果
    with open(output_path / 'analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    # 生成统计报告
    generate_analysis_report(analysis_results, output_path)
    
    print(f"\n分析完成！")
    print(f"总图像数: {analysis_results['total_images']}")
    print(f"总检测数: {analysis_results['total_detections']}")
    print(f"有检测结果的图像数: {analysis_results['images_with_detections']}")
    print(f"检测率: {analysis_results['images_with_detections']/analysis_results['total_images']*100:.1f}%")

def generate_analysis_report(analysis_results, output_path):
    """生成分析报告图表"""
    # 创建统计图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 检测数量分布
    detection_counts = [item['detections'] for item in analysis_results['detection_details']]
    axes[0, 0].hist(detection_counts, bins=max(10, max(detection_counts)+1), alpha=0.7, color='skyblue')
    axes[0, 0].set_title('每张图像检测数量分布')
    axes[0, 0].set_xlabel('检测数量')
    axes[0, 0].set_ylabel('图像数量')
    
    # 2. 置信度分布
    all_confidences = []
    for item in analysis_results['detection_details']:
        all_confidences.extend(item['confidences'])
    
    if all_confidences:
        axes[0, 1].hist(all_confidences, bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('检测置信度分布')
        axes[0, 1].set_xlabel('置信度')
        axes[0, 1].set_ylabel('检测数量')
    
    # 3. 检测率饼图
    has_detection = analysis_results['images_with_detections']
    no_detection = analysis_results['total_images'] - has_detection
    
    axes[1, 0].pie([has_detection, no_detection], 
                   labels=['有检测', '无检测'], 
                   autopct='%1.1f%%',
                   colors=['lightcoral', 'lightblue'])
    axes[1, 0].set_title('图像检测率')
    
    # 4. 统计信息文本
    axes[1, 1].axis('off')
    stats_text = f"""
    分析统计报告
    
    总图像数: {analysis_results['total_images']}
    总检测数: {analysis_results['total_detections']}
    有检测结果的图像: {analysis_results['images_with_detections']}
    检测率: {analysis_results['images_with_detections']/analysis_results['total_images']*100:.1f}%
    
    平均每张图像检测数: {analysis_results['total_detections']/analysis_results['total_images']:.2f}
    """
    
    if all_confidences:
        stats_text += f"""
    平均置信度: {np.mean(all_confidences):.3f}
    最高置信度: {np.max(all_confidences):.3f}
    最低置信度: {np.min(all_confidences):.3f}
        """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path / 'analysis_report.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='YOLO检测结果可视化和分析工具')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                       help='运行模式: single(单张图像) 或 batch(批量分析)')
    parser.add_argument('--image', type=str, help='单张图像路径 (single模式)')
    parser.add_argument('--folder', type=str, help='图像文件夹路径 (batch模式)')
    parser.add_argument('--weights', type=str, 
                       default=r'D:\ANU\ONCMAPPING\ONC-MAPPING\runs\train\ONCMAPPING\weights\best.pt',
                       help='模型权重路径')
    parser.add_argument('--output', type=str, 
                       default=r'D:\ANU\ONCMAPPING\ONC-MAPPING\Code\visualization_results',
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.image:
            print("单张图像模式需要指定 --image 参数\n示例: python visualize.py --mode single --image D:/ANU/ONCMAPPING/Urambi2025/ACT2025_RGB_75mm_ortho__Urambi_Clip.tif")
            return
        visualize_predictions(args.image, args.weights, args.output, args.conf)
    
    elif args.mode == 'batch':
        if not args.folder:
            print("批量分析模式需要指定 --folder 参数\n示例: python visualize.py --mode batch --folder D:/ANU/ONCMAPPING/Urambi2025/")
            return
        batch_analyze(args.folder, args.weights, args.output, args.conf)

if __name__ == "__main__":
    main()