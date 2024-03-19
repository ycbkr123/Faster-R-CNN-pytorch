# Faster R-CNN
![image](https://github.com/ycbkr123/Faster-R-CNN-pytorch/assets/73626645/8c2cb0d9-126a-4b20-891f-0f3da5492db1)

후보 영역 추출을 위해 사용되는 Selective search 알고리즘은 CPU 상에서 동작하고 이로 인해 네트워크에서 병목현상이 발생하게 된다. 
Faster R-CNN은 이러한 문제를 해결하고자 후보 영역 추출 작업을 수행하는 네트워크인 Region Proposal Network(RPN)를 도입한다. 
RPN은 region proposals를 보다 정교하게 추출하기 위해 다양한 크기와 가로세로비를 가지는 bounding box인 Anchor box를 도입한다. 
Faster R-CNN 모델을 간략하게 보면 RPN과 Fast R-CNN 모델이 합쳐졌다고 볼 수 있습니다. 
RPN에서 region proposals를 추출하고 이를 Fast R-CNN 네트워크에 전달하여 객체의 class와 위치를 예측합니다. 
이를 통해 모델의 전체 과정이 GPU 상에서 동작하여 병목 현상이 발생하지 않으며, end-to-end로 네트워크를 학습시키는 것이 가능해집니다. 

 # 동작 순서
* 원본 이미지를 pre-trained된 CNN 모델에 입력하여 feature map을 얻습니다.
* feature map은 RPN에 전달되어 적절한 region proposals을 산출합니다.
* region proposals와 1) 과정에서 얻은 feature map을 통해 RoI pooling을 수행하여 고정된 크기의 feature map을 얻습니다. 
* Fast R-CNN 모델에 고정된 크기의 feature map을 입력하여 Classification과 Bounding box regression을 수행합니다.

  **RPN(Region Proposal Networks)**
 ![image](https://github.com/ycbkr123/Faster-R-CNN-pytorch/assets/73626645/da690084-8d49-4df0-a1a0-b37255867ff4)

RPN(Region Proposal Networks)은 크기에 상관없이 이미지 전체를 입력받는다. 그다음 영역 추정 경계 박스를 반환한다. 
각 경계 박스는 객체가 있는지 여부를 점수로 나타낸다. 이런 RPN을 합성곱 네트워크로 처리하도록 만들었다. RPN과 Fast R-CNN 객체 탐지기는 피처 맵을 공유한다

# CODE
* train.py코드는 torchvision에서 제공한 fasterrcnn_resnet50_fpn 모델로 학습한 것으로, 다른 코드들과는 다르게 yolo 형식의 좌표를 pascal VOC형식으로 변환 함수가 있다.
* test.py코드는 train.py를 통해 얻은 best_model.pt파일을 사용하여 mAP, 각 Class별 AP값을 구한다. 또한 result.csv 파일 형식으로 결과가 저장되도록 설계하였다.
