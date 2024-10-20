import torch
import sys

sys.path.insert(0, '/mnt/d/poseture_estimated_by_yolo/SittingPostureDetection/yolov5')

device = torch.device('cpu')
model = torch.load('data/inference_models/small640.pt', map_location=device)['model'].float()
torch.onnx.export(model, torch.zeros((1, 3, 640, 640)), 'small640.onnx', opset_version=12)
