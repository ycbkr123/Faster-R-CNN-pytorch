# 라이브러리 호출
import random
import pandas as pd
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import torchmetrics
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm
from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_recall_fscore_support

# 오류문구 무시
import warnings
warnings.filterwarnings(action='ignore')

# GPU사용 선언
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 필요한 하이퍼파라미터 설정
CFG = {
    'NUM_CLASS':6,
    'IMG_SIZE':640,
    'EPOCHS':1,
    'LR':3e-4,
    'BATCH_SIZE':8,
    'SEED':42
}

# 데이터 전처리
class CustomDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.imgs = sorted(glob.glob(root+'/*'+'/*'))
        self.label_root = self.root.replace("images", "labels")
        self.boxes = sorted(glob.glob(self.label_root+'/*'+'/*.txt'))
    
    # yolo 좌표를 rcnn좌표로 변환
    def yolo_to_pascal_voc(self, x_center, y_center, width, height, image_width, image_height):
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)

        # [0,1]범위 초과하지 못하도록 설정
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(x_max, image_width)
        y_max = min(y_max, image_height)

        return [x_min, y_min, x_max, y_max]

    #boxe 좌표 설정 
    def parse_boxes(self, box_path, image_size):
        with open(box_path, 'r') as file:
            lines = file.readlines()

        boxes = []
        labels = []

        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = int(values[0])
            x_center, y_center, width, height = values[1], values[2], values[3], values[4]

            img_width, img_height = image_size
            box = self.yolo_to_pascal_voc(x_center, y_center, width, height, img_width, img_height)

            boxes.append(box)
            labels.append(class_id)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        # img_path = self.imgs[idx]
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        # 빈 텐서로 초기화
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)

        
        box_path = self.boxes[idx]
        boxes, labels = self.parse_boxes(box_path, (width, height))
        labels += 1  # Background = 0

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes.numpy(), labels=labels.numpy())
            img = transformed["image"]
            # Transform 적용 후 boxes와 labels 업데이트
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32) if transformed["bboxes"] else boxes
            labels = torch.tensor(transformed["labels"], dtype=torch.int64) if transformed["labels"] else labels

        # 항상 img, boxes, labels 반환
        return img, boxes, labels
        
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    images, targets_boxes, targets_labels = tuple(zip(*batch))
    images = torch.stack(images, 0)
    targets = []
    
    for i in range(len(targets_boxes)):
        target = {
            "boxes": targets_boxes[i],
            "labels": targets_labels[i]
        }
        targets.append(target)

    return images, targets

def get_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Dataset 생성
test_dataset = CustomDataset('/home/user/cvat/yolov5/dataset3/images/val', train=False, transforms=get_transforms())

# DataLoader 생성
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn)

# model 생성
def build_model(num_classes=CFG['NUM_CLASS']+1):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path="best_model.pt"):
    # 모델 구조 재정의
    model = build_model()
    # 저장된 state_dict 로드
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

# 모델 로딩
loaded_model = load_model("best_model.pt")


# 테스트함수
def evaluate_model(model, data_loader, device, csv_file_path):
    model.eval()  # 모델을 평가 모드로 설정
    metric = MeanAveragePrecision(class_metrics=True)  # mAP 계산을 위한 메트릭 초기화

    results_data = {
        "Class ID": [],
        "AP": []
    }

    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)  # Model predictions

            preds = []
            target = []
            
            for i, output in enumerate(outputs):
                # 예측된 결과를 torchmetrics가 요구하는 형태로 변환
                pred = {
                    "boxes": output['boxes'].cpu(),
                    "labels": output['labels'].cpu(),
                    "scores": output['scores'].cpu()
                }
                preds.append(pred)

                # 실제 타겟 데이터를 torchmetrics가 요구하는 형태로 변환
                tgt = {
                    "boxes": targets[i]['boxes'].cpu(),
                    "labels": targets[i]['labels'].cpu()
                }
                target.append(tgt)
            
            metric.update(preds=preds, target=target)
                
    # 최종 mAP 값 계산
    metric_result = metric.compute()
    print(f"mAP: {metric_result['map']:.4f}")

    # Iterate over the per-class AP results
    if isinstance(metric_result['map_per_class'], dict):
        for class_id, ap in metric_result['map_per_class'].items():
            print(f"Class {class_id} AP: {ap:.4f}")
            results_data["Class ID"].append(class_id)
            results_data["AP"].append(ap.item())
    else:  # If it's a tensor, we need to handle it differently
        for class_id, ap in enumerate(metric_result['map_per_class'].tolist()):
            print(f"Class {class_id} AP: {ap:.4f}")
            results_data["Class ID"].append(class_id)
            results_data["AP"].append(ap)

    # mAP 추가
    results_data["Class ID"].append("mAP")
    results_data["AP"].append(metric_result['map'].item())

    # 결과 데이터를 DataFrame으로 변환하고 CSV 파일로 저장
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(csv_file_path, index=False)
    print(f"Results saved to {csv_file_path}")

    return metric_result

# 모델 평가 실행
mAP_result = evaluate_model(loaded_model, test_loader, device, "results.csv")