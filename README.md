# 遥感图像岩石检测处理工具

这个工具用于处理 TIFF 格式的遥感图像，主要功能包括图像增强和岩石区域分割。

## 功能特点

- 支持多波段 TIFF 图像处理
- 使用 CLAHE 进行自适应直方图均衡化
- Unsharp Masking 图像锐化
- 基于 NDVI 的岩石区域分割
- Canny 边缘检测
- 保留地理参考信息

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

基本用法：

```bash
python rock_detection.py input.tif output.tif
```

使用自定义参数：

```bash
python rock_detection.py input.tif output.tif \
    --clahe_clip 2.0 \
    --clahe_tile 8 \
    --sharpen_amount 1.5 \
    --ndvi_threshold 0.2 \
    --canny_low 100 \
    --canny_high 200
```

## 参数说明

- `input.tif`：输入的 TIFF 图像文件路径
- `output.tif`：输出的 TIFF 图像文件路径
- `--clahe_clip`：CLAHE 对比度限制参数（默认：2.0）
- `--clahe_tile`：CLAHE 网格大小（默认：8）
- `--sharpen_amount`：图像锐化强度（默认：1.5）
- `--ndvi_threshold`：NDVI 分割阈值（默认：0.2）
- `--canny_low`：Canny 边缘检测低阈值（默认：100）
- `--canny_high`：Canny 边缘检测高阈值（默认：200）

## 输出文件

程序会生成两个输出文件：
1. `enhanced_output.tif`：增强后的图像
2. `segmentation_output.tif`：岩石分割结果掩码

## 注意事项

- 确保输入图像包含近红外波段（默认为第4波段）和红色波段（默认为第1波段）
- 根据实际图像特征调整参数以获得最佳效果
- 处理大型图像时可能需要较大的内存空间
