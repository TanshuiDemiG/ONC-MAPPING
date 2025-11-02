# ONC石头检测项目

本项目使用YOLO模型进行石头目标检测，包含训练、推理和可视化功能。

## 文件说明

- `train.py` - 模型训练脚本
- `predict.py` - 推理预测脚本
- `visualize.py` - 结果可视化和分析脚本
- `ONC_Mapping_Stone-2/` - 训练数据集
- `../runs/train/ONCMAPPING/weights/` - 训练好的模型权重

## 使用方法

### 1. 模型训练

```bash
python train.py
```

训练完成后，权重文件会保存在 `../runs/train/ONCMAPPING/weights/` 目录下：
- `best.pt` - 验证集上表现最好的权重
- `last.pt` - 最后一个epoch的权重

### 2. 推理预测

#### 基本用法
```bash
python predict.py
```

#### 自定义参数
```bash
python predict.py --weights ../runs/train/ONCMAPPING/weights/best.pt --source test_images/ --output results/ --conf 0.3
```

#### 参数说明
- `--weights`: 权重文件路径
- `--source`: 输入图像路径或文件夹
- `--output`: 输出结果保存路径
- `--conf`: 置信度阈值 (默认: 0.25)
- `--iou`: NMS IoU阈值 (默认: 0.45)
- `--device`: 推理设备 (默认: cpu)
- `--save-txt`: 保存检测结果为txt文件
- `--save-conf`: 在txt文件中保存置信度

### 3. 结果可视化

#### 单张图像可视化
```bash
python visualize.py --mode single --image path/to/image.jpg
```
```bash
python visualize.py --mode single --image D:/ANU/ONCMAPPING/ONC-MAPPING/Code/Usage_Script/img/trial1.jpg
```


#### 批量分析
```bash
python visualize.py --mode batch --folder path/to/images/
```

#### 参数说明
- `--mode`: 运行模式 (single/batch)
- `--image`: 单张图像路径
- `--folder`: 图像文件夹路径
- `--weights`: 模型权重路径
- `--output`: 输出目录
- `--conf`: 置信度阈值

## 输出结果

### 推理结果
- 带有检测框的图像
- 检测结果txt文件（如果启用）
- 检测统计信息

### 可视化结果
- 原图与检测结果对比图
- 检测统计报告
- 置信度分布图
- 检测数量分布图

## 模型信息

- 基础模型: YOLOv8n
- 训练数据: ONC_Mapping_Stone-2 数据集
- 检测类别: 石头 (stone)
- 图像尺寸: 512x512

## 依赖库

```bash
pip install ultralytics opencv-python matplotlib numpy pathlib
```

## 注意事项

1. 确保已安装所需的依赖库
2. 如果有GPU，可以将 `device='cpu'` 改为 `device=0` 以使用GPU加速
3. 根据实际需求调整置信度阈值
4. 大批量处理时建议使用GPU以提高速度


