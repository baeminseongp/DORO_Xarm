import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
from ox_crop import imgcut,get_image_files

# 이미지 증강
trans = transforms.Compose([
    transforms.Resize((100, 100)),
    # transforms.RandomHorizontalFlip(),  # 랜덤하게 이미지를 좌우 반전합니다.
    # transforms.RandomRotation(15),  # 이미지를 최대 15도까지 랜덤하게 회전시킵니다.
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 색상을 랜덤하게 변경합니다.
    transforms.ToTensor(), # 배열 -> Tensor로 변환합니다.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 픽셀 값의 범위를 0~1로 조절합니다.
    ])

class CustomDataset(Dataset):

    def __init__(self, image_files, transform=None):
        # 이미지 파일들을 정렬하여 저장합니다.
        self.image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.transform = transform

    def __len__(self):
        # 데이터셋의 총 샘플 수를 반환합니다.
        return len(self.image_files)

    def __getitem__(self, idx):
        # 주어진 인덱스에 해당하는 샘플을 반환합니다.
        img_path = self.image_files[idx]
        # 이미지를 RGB 형식으로 열어서 반환합니다.
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            # 이미지에 전처리(transform)를 적용합니다.
            image = self.transform(image)
        return image


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 합성곱 신경망 모델을 정의합니다.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=9680, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=50, out_features=2, bias=True)
        
    def forward(self, x):
        # 데이터를 모델에 통과시켜 예측값을 반환합니다.
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1,9680)

        x = self.fc1(x)
        x = self.relu1_fc1(x)

        x = self.fc2(x)

        return x


def main():
    # 모델과 이미지 파일 경로를 초기화합니다.
    DEVICE = torch.device('cpu')
    model = CNN().to(DEVICE)
    imgpath ="/home/min/pytorch-ox/test_image/20240413_004119.jpg"
    imgcut(imgpath) # 주석 추가: 이미지를 잘라내는 함수 호출
    inferdir = "/home/min/pytorch-ox/inferance"

    # 이미지 파일들을 불러와 데이터셋을 생성합니다.
    image_files = get_image_files(inferdir)
    infer_dataset = CustomDataset(image_files, transform=trans)

    # 미리 학습된 모델을 불러와서 예측을 수행합니다.
    model2 = torch.load("modelcnn.pt", map_location=torch.device('cpu'))
    model2.eval()  
    
    predictions = []
    
    # 각 이미지에 대해 모델을 통해 예측을 수행합니다.
    for idx in range(1, 10):  # 파일 이름이 1부터 9까지 있으므로 반복 범위를 1부터 9까지로 설정합니다.
        img = infer_dataset[idx - 1]  # 파일 이름이 1부터 시작하므로 인덱스를 1씩 줄입니다.
        out = model2(img.unsqueeze(0))
        _, predict = torch.max(out.data, 1)
        predictions.append(predict.item())
        
    print("Predictions:", predictions)


if __name__ == '__main__':
    main()
