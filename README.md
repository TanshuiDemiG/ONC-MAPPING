# YOLO26 Training

## Branch Description

This branch is code for training YOLO26 object detection model.

Data are split into three folders - train, valid and test.

Training results are stored in runs/detect folder. Results includes best and last weights, diagrams showing training process and sample images of ground truth and prediction for visual comparison.

## Usage

- ensure your environment satisfies requirements.txt, and if not, simply use `pip install -r requirements.txt`.
- set hyperparameters and augmentations in `yolov26.py` file.
- use `python yolov26.py` to start training.
- check results in `runs/detect`.

## Directory Structure

- `runs/detect/`：training results
- `test/`：test images and labels
- `train/`：train images and labels
- `valid/`：validation images and labels
- `data.yaml`：information of the dataset which the model needs
- `yolo26n.pt`: nano size YOLO26 object detection model initial weight
- `yolo26s.pt`: small size YOLO26 object detection model initial weight
- `yolov26.py`: training code
