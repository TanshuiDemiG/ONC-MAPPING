# ONC Stone Pipeline 使用说明

## 1. 适用范围

本说明对应以下两个文件：

- `d:\ANU\ONCMAPPING\ONC-MAPPING\COLAB_ED\stone_pipeline_colab.ipynb`
- `d:\ANU\ONCMAPPING\ONC-MAPPING\COLAB_ED\stone_pipeline_colab.py`

该流程主要用于在 Google Colab 中完成以下任务：

1. 对正射影像进行石块检测
2. 基于植被图构建全网格评分
3. 用 canopy 面直接裁切石块面
4. 只保留 canopy 裁切后石块范围内的评分结果

当前 notebook 默认采用的是“混合工作流”：

- 先做 rock detection
- 再做 full-grid habitat scoring
- 最终输出限制在 canopy 裁切后的 rock extent 内

## 2. 推荐目录结构

notebook 中默认使用以下 Google Drive 目录：

```text
/content/drive/MyDrive/ONCMAPPING/DEPLOY
├─ ORIGINAL_IMG
├─ CANOPY_IMAGE
├─ VEGE_MAP
├─ Model_Weights
└─ OUT
```

对应配置如下：

```python
DRIVE_ROOT = '/content/drive/MyDrive/ONCMAPPING/DEPLOY'
ORTHO = f'{DRIVE_ROOT}/ORIGINAL_IMG'
CANOPY = f'{DRIVE_ROOT}/CANOPY_IMAGE'
VEG = f'{DRIVE_ROOT}/VEGE_MAP'
OUT_DIR = f'{DRIVE_ROOT}/OUT'
```

建议准备的数据如下：

- `ORIGINAL_IMG`：待检测的正射影像，支持 `.tif`、`.tiff`、`.img`、`.jp2`、`.vrt`
- `CANOPY_IMAGE`：canopy 面数据，当前脚本要求为 `.shp`；仅 `full` 和 `habitat_only` 必需
- `VEGE_MAP`：植被图栅格，支持 `.tif`、`.tiff`、`.img`、`.jp2`、`.vrt`；仅 `full` 和 `habitat_only` 必需
- `Model_Weights`：检测模型，支持 `.pt` 或 `.onnx`
- `OUT`：输出目录，脚本会自动创建每次运行的子目录

补充说明：

- 脚本启动时会检查 `DRIVE_ROOT` 下是否存在 `ORIGINAL_IMG` 和 `VEGE_MAP`
- 如果这两个目录缺失，会自动创建空目录
- 仅自动建目录，不会自动放入数据文件；目录为空时，后续仍会提示找不到支持格式的输入文件

## 3. Notebook 运行步骤

按 notebook 中单元顺序依次执行：

1. 配置用户参数
2. 安装依赖
3. 挂载 Google Drive
4. 检查 PyTorch / CUDA / 模型路径
5. 导入 `stone_pipeline_colab.py`
6. 执行主流程

如果路径和文件准备正确，最后一个单元会自动：

- 读取 `ORTHO` 下全部影像
- 自动解析 canopy、vegetation、model 路径
- 对每张影像分别运行一次 `run_pipeline(...)`
- 输出所有生成文件的路径

## 4. 输入文件选择规则

脚本内部使用 `resolve_input_path()` 和 `resolve_input_paths()` 自动选择输入：

- `ORTHO`：如果给的是目录，会递归读取目录下所有支持格式影像
- `CANOPY`：如果给的是目录，会递归查找第一个 `.shp`
- `VEG`：如果给的是目录，会递归查找第一个支持格式栅格
- `MODEL_PATH`：如果给的是目录，会递归查找第一个 `.pt` 或 `.onnx`

注意：

- `ORTHO` 是批处理入口，可能一次处理多张影像
- `CANOPY`、`VEG`、`MODEL_PATH` 在当前 notebook 中是全局共享的
- `CANOPY` 和 `VEG` 只会在 `RUN_MODE` 为 `full` 或 `habitat_only` 时被检查
- `ORIGINAL_IMG` 和 `VEGE_MAP` 若目录不存在，会先自动创建再继续执行
- 如果目录下有多个候选文件，脚本会按路径排序后选择第一个

## 5. 运行模式

`RUN_MODE` 支持 3 种：

### 5.1 `full`

默认模式，完整流程：

1. 检测石块
2. 输出 `rocks.shp`
3. 用植被图和石块结果计算评分
4. 输出裁切后的评分栅格和网格面

适合大多数正常使用场景。

### 5.2 `detection_only`

只进行石块检测，输出 `rocks.shp` 及相关结果，不生成 habitat score。

适合：

- 先单独检查检测质量
- 只需要石块分布，不需要评分
- 没有准备 `VEG` 或 `CANOPY`，但希望先跑检测

该模式下必须有：

- `ORTHO`
- `MODEL_PATH`

该模式下可以缺失：

- `VEG`
- `CANOPY`

### 5.3 `habitat_only`

跳过检测，直接使用已有石块面文件进行评分。

此时必须设置：

```python
RUN_MODE = 'habitat_only'
EXISTING_ROCKS = '已有 rocks.shp 的路径'
```

适合：

- 已经有历史检测结果
- 希望重复调整评分参数而不重新跑检测

该模式下必须有：

- `VEG`
- `CANOPY`
- `EXISTING_ROCKS`

## 6. 模型配置

notebook 默认提供 3 个模型入口：

```python
MODEL_CHOICE = 'roboflow1'
MODEL_PATHS = {
    'local_trained1': f'{DRIVE_ROOT}/Model_Weights/best.pt',
    'local_trained2': f'{DRIVE_ROOT}/Model_Weights/best_DY.pt',
    'roboflow1': f'{DRIVE_ROOT}/Model_Weights/onc_drone2/9/weights.onnx',
}
MODEL_PATH = MODEL_PATHS[MODEL_CHOICE]
MODEL_BACKEND = 'auto'
```

说明：

- `MODEL_BACKEND = 'auto'` 时，脚本会根据文件后缀自动判断使用 `pt` 还是 `onnx`
- 对于 `.onnx` 模型，脚本会尝试自动读取模型输入尺寸
- 如果 ONNX 模型要求的输入尺寸与 `MODEL_IMGSZ` 不一致，脚本会自动覆盖为模型实际尺寸

## 7. 检测参数说明

### 7.1 切片与检测阈值

```python
TILE_SIZE = 512
OVERLAP = 128
CONF = 0.25
IOU_NMS = 0.35
MAX_TILES = None
MODEL_IMGSZ = 640
TARGET_CLASS_NAMES = ['rock']
```

含义：

- `TILE_SIZE`：切片大小，正射影像会按此尺寸分块推理
- `OVERLAP`：切片重叠像素，降低边缘漏检
- `CONF`：模型置信度阈值
- `IOU_NMS`：跨切片 NMS 的 IoU 阈值
- `MAX_TILES`：限制最大处理切片数，`None` 表示不限制
- `MODEL_IMGSZ`：YOLO 推理尺寸，主要对 `.pt` 模型有效
- `TARGET_CLASS_NAMES`：只保留指定类别，当前默认仅保留 `rock`

建议：

- 漏检较多时，可适当降低 `CONF`
- 重复框较多时，可适当降低 `IOU_NMS`
- 大图测试阶段可先设置 `MAX_TILES = 10` 做快速检查

### 7.2 绿色区域过滤

```python
GREEN_FILTER = False
GREEN_THRESHOLD = 0.35
GREEN_MARGIN = 12.0
```

作用：

- 当 `GREEN_FILTER = True` 时，会对候选框内像素做绿色主导判定
- 若框内绿色主导比例过高，则过滤该检测结果

适合：

- 误检大量发生在明显植被区域时

不建议一开始就开启，建议先比较开启前后的检测差异。

## 8. 尺寸分箱功能

```python
SIZE_BINS_ENABLED = False
SIZE_BINS = '10,40,100'
SIZE_METRIC = 'max_side_cm'
MANUAL_CM_PER_PIXEL = None
WRITE_SIZE_BIN_SHAPEFILES = True
HABITAT_SIZE_BIN = ''
```

功能说明：

- 开启后，脚本会根据检测框像素尺寸换算实际尺寸
- 再按阈值把石块分配到不同 size class

默认 `SIZE_BINS = '10,40,100'` 时，会形成：

- `0-10`
- `10-40`
- `40-100`
- `>100`

关键点：

- `SIZE_METRIC` 默认是 `max_side_cm`，即框长边的厘米值
- 若影像 CRS 可推导像素尺寸，脚本会自动换算 `cm/pixel`
- 若 CRS 不可用，可手动设置 `MANUAL_CM_PER_PIXEL`
- `WRITE_SIZE_BIN_SHAPEFILES = True` 时，会额外输出每个尺寸类别对应的 shapefile
- `HABITAT_SIZE_BIN` 可指定仅用某一尺寸类别参与 habitat scoring

例如：

```python
SIZE_BINS_ENABLED = True
SIZE_BINS = '10,40,100'
HABITAT_SIZE_BIN = '40-100'
```

表示：

- 检测结果会先分箱
- habitat scoring 只使用 `40-100` 这一类石块

## 9. 评分参数说明

notebook 当前暴露的评分参数如下：

```python
BLOCK_SIZE = '1'
SCORE_SCALING = 'absolute'
VEGETATION_WEIGHT = 0.7
ROCK_WEIGHT = 0.3
ROCK_PERCENTILE = 95.0
ROCK_CAP = None
ROCK_ASSIGNMENT = 'centroid'
```

含义：

- `BLOCK_SIZE`：评分网格大小，`'1'` 表示按植被图原始像素为单元；也可写成 `'3'` 或 `'3x5'`
- `SCORE_SCALING`：
  - `absolute`：直接输出原始加权分数
  - `minmax`：对正分数再做最小-最大归一化
- `VEGETATION_WEIGHT`：植被得分权重
- `ROCK_WEIGHT`：石块得分权重
- `ROCK_PERCENTILE`：当 `ROCK_CAP = None` 时，用该分位数自动确定 rock count 的归一化上限
- `ROCK_CAP`：手动指定 rock score 归一化上限
- `ROCK_ASSIGNMENT`：
  - `centroid`：按石块中心点落入哪个网格计数
  - `intersects`：按石块与网格是否相交计数

当前 notebook 推荐保持：

- `SCORE_SCALING = 'absolute'`
- `ROCK_ASSIGNMENT = 'centroid'`

## 10. 特殊点统计

```python
SPECIAL_POINTS = [
    ('P1', -35.398418, 149.048551),
    ('P2', -35.399366, 149.048177),
    ('P3', -35.400982, 149.048679),
]
SPECIAL_RADIUS_M = 11.3
```

脚本会在每次运行结束后，围绕这些经纬度点建立缓冲区，并统计：

- 缓冲范围内石块总数
- 各尺寸类别数量（如果启用了 size bins）

这些统计会写入运行目录下的 `SUMMARY.md`。

如果不需要此功能，可将：

```python
SPECIAL_POINTS = []
```

## 11. Smoke Test 与运行命名

```python
SMOKE_TEST = False
RUN_NAME = ''
```

说明：

- `SMOKE_TEST = True` 且 `MAX_TILES is None` 时，脚本会自动只处理 1 个 tile，便于验证流程
- `RUN_NAME` 为空时，输出目录会自动使用时间戳命名
- 若设置 `RUN_NAME = 'test1'`，则输出目录会使用该名称

## 12. 实际输出内容

每张影像都会在以下目录生成独立结果：

```text
OUT/<image_name>/<run_name 或时间戳>/
```

典型输出包括：

- `run_config.json`：本次运行参数快照
- `rocks.shp`：石块检测结果
- `rocks__<size_bin>.shp`：各尺寸类别的石块结果，只有启用 size bins 且允许写出时才生成
- `rock_overlay_score.tif`：最终评分栅格，单波段浮点数
- `rock_overlay_rgb.tif`：评分伪彩色渲染图，RGBA
- `rock_scored_cells.shp`：裁切到 rock extent 后的评分网格面
- `SUMMARY.md`：本次运行摘要

注意：

- 输出结果只保留在 canopy 裁切后的 rock extent 范围内
- 范围外像素在输出评分栅格中会被写为 0
- `rock_scored_cells.shp` 保存的是评分后的网格单元，不是平滑后的 habitat zone 面

## 13. 当前 notebook 的默认处理逻辑

按 notebook 当前配置，最后一个单元实际做的是：

1. 递归读取 `ORTHO` 下所有正射影像
2. 解析一个 canopy shapefile
3. 解析一个 vegetation raster
4. 解析一个模型权重文件
5. 对每张影像执行 `run_pipeline(...)`
6. 在 `OUT` 下分别生成独立运行目录

其中 `run_pipeline(...)` 的核心逻辑为：

- `full` / `detection_only`：先做石块检测
- 将检测结果保存为 `rocks.shp`
- `full` / `habitat_only`：读取植被图、canopy 和 rock 数据进行评分
- 先把 canopy 从 rock polygon 中差分扣除
- 再对全网格评分，但最终只保留 canopy-cut rock extent 内的输出

## 14. 推荐使用流程

### 场景 A：第一次完整跑通

建议配置：

```python
RUN_MODE = 'full'
SMOKE_TEST = True
MAX_TILES = None
```

先确认：

- 路径能否正确找到输入文件
- 模型是否可正常加载
- 输出目录是否能正常生成

确认无误后，再改为：

```python
SMOKE_TEST = False
```

### 场景 B：只检查检测效果

建议配置：

```python
RUN_MODE = 'detection_only'
```

这样可以先重点查看：

- `rocks.shp`
- 是否存在明显漏检或误检
- size bins 是否合理

在这个模式下，如果暂时没有植被图或 canopy 文件，也可以直接运行。

### 场景 C：固定石块结果，反复调评分

建议配置：

```python
RUN_MODE = 'habitat_only'
EXISTING_ROCKS = '某次输出的 rocks.shp'
```

这样可以反复调整：

- `BLOCK_SIZE`
- `VEGETATION_WEIGHT`
- `ROCK_WEIGHT`
- `ROCK_PERCENTILE`
- `ROCK_CAP`
- `ROCK_ASSIGNMENT`

而无需重复执行检测。

## 15. 常见问题

### 15.1 报错“找不到脚本文件”

请确认以下文件已复制到 Google Drive 对应位置：

```text
/content/drive/MyDrive/ONCMAPPING/DEPLOY/stone_pipeline_colab.py
```

因为 notebook 会直接导入：

```python
from stone_pipeline_colab import log, resolve_input_path, resolve_input_paths, run_pipeline
```

### 15.2 报错找不到输入文件

请检查：

- `ORTHO`、`CANOPY`、`VEG`、`MODEL_PATH` 是否指向正确目录或文件
- 目录下是否存在支持格式的文件
- canopy 当前是否确实为 `.shp`

### 15.3 ONNX 模型尺寸不匹配

这是脚本已经考虑的情况。若能正确读取 ONNX 输入尺寸，会自动覆盖 `MODEL_IMGSZ`。通常不需要手动处理。

### 15.4 size bins 无法换算厘米

请检查：

- 影像 CRS 是否存在
- CRS 是否能推导线性单位

若不能，请手动设置：

```python
MANUAL_CM_PER_PIXEL = 具体数值
```

### 15.5 为什么输出区域比整张 vegetation 图小

这是当前流程的设计结果，不是错误。脚本会：

1. 用 canopy 直接裁切 rock polygons
2. 对全网格评分
3. 只保留 canopy-cut rock extent 范围内的输出

因此最终评分结果不会覆盖整张 vegetation 图。

## 16. 建议后续补充

当前这份说明主要面向 notebook 使用。后续如果需要，可以继续补充两部分内容：

- 命令行版本 `stone_pipeline_colab.py` 的 CLI 使用说明
- 各输出字段的详细字段表说明，例如 `rocks.shp` 和 `rock_scored_cells.shp` 的属性字段解释
