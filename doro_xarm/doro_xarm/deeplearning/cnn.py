import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn

DEVICE = torch.device('cpu')  # 기기 설정: CPU 사용
BATCH_SIZE = 10  # 배치 크기 설정
EPOCHS = 30  # 에포크 설정



# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=9680, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=50, out_features=3, bias=True)

    def forward(self, x):
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

    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss(self, loss):
        self.loss = loss
    
    def train_dataset(self, train_loader, log_interval, epoch): 
                                                                        # 모델을 학습 모드로 설정
        self.train()
        for batch_idx, (image, label) in enumerate(train_loader):                         # 미니 배치 단위로 학습
            image = image.to(self.device)                                                # 이미지를 기기로 이동
            label = label.to(self.device)                                                # 레이블을 기기로 이동
            self.optimizer.zero_grad()                                                   # 그래디언트 초기화
            output = self(image)                                                         # 모델에 이미지 전달하여 예측
            loss = self.loss(output, label)                                              # 손실 계산
            loss.backward()                                                              # 역전파
            self.optimizer.step()                                                        # 가중치 업데이트

            # if batch_idx % log_interval == 0:                                            # 현재 진행 중인 학습 상황 출력   
            #     print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            #         epoch, batch_idx * len(image), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))
    
    def evaluate(self, testloader):
        self.eval()                                                                      # 모델을 평가 모드로 설정
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for image, label in testloader:
                image = image.to(self.device)                                            # 이미지를 기기로 이동
                label = label.to(self.device)                                            # 레이블을 기기로 이동
                output = self(image)                                                     # 모델에 이미지 전달하여 예측
                print('label: ', label, 'predict', torch.argmax(output, dim=-1))
                test_loss += self.loss(output, label).item()                              # 손실 누적
                prediction = output.max(1, keepdim = True)[1]                            # 가장 높은 확률을 가진 클래스 선택
                correct += prediction.eq(label.view_as(prediction)).sum().item()         # 정확한 예측 수 누적

        # 손실 평균 및 정확도 계산
        test_loss /= (len(testloader.dataset) / BATCH_SIZE)
        test_accuracy = 100. * correct / len(testloader.dataset)
        return test_loss, test_accuracy