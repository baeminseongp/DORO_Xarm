from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn

DEVICE = torch.device('cpu')

BATCH_SIZE = 10
EPOCHS = 30

trans = transforms.Compose([
    transforms.Resize((100, 100)),  
    transforms.RandomHorizontalFlip(),  # 랜덤하게 이미지를 좌우 반전합니다.
    transforms.RandomRotation(15),  # 이미지를 최대 15도까지 랜덤하게 회전시킵니다.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 색상을 랜덤하게 변경합니다.
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.ImageFolder(root = "/home/zxro/DeepLearning/dataset",
                                            transform = trans)
testset = torchvision.datasets.ImageFolder(root = "/home/zxro/DeepLearning/testset", transform = trans)

#trainset.__getitem__(1)
#print(len(trainset))

classes = trainset.classes
print(classes)

trainloader = DataLoader(trainset,
                        batch_size = BATCH_SIZE,
                        shuffle = True)
testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False)

for (X_train, y_train) in trainloader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

model = CNN().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
criterion = nn.CrossEntropyLoss()

print(model)

def train(model, trainloader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(trainloader):
        image = image.to(DEVICE) # 이미지를 GPU로 이동
        label = label.to(DEVICE) # 레이블을 GPU로 이동
        optimizer.zero_grad() # 그래디언트 초기화
        output = model(image) # 모델에 이미지 전달하여 예측, 모델 함수는 Net함수로 train의 main code
        loss = criterion(output, label) # 손실 계산
        loss.backward() # 역전파, 이전 텐서와 현재 텐서의 차이, 즉 gradient를 반환
        optimizer.step() # optimizer에게 loss function를 효율적으로 최소화 할 수 있게 파라미터 수정 위탁

        # 현재 진행 중인 학습 상황 출력
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} \tTrain Loss: {:.6f}".format(
                EPOCHS,  
                loss.item()))
            
def evaluate(model, testloader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in testloader:
            image = image.to(DEVICE) # 이미지를 GPU로 이동
            label = label.to(DEVICE) # 레이블을 GPU로 이동
            output = model(image) # 모델에 이미지 전달하여 예측
            test_loss += criterion(output, label).item() # 손실 누적
            prediction = output.max(1, keepdim = True)[1] # 가장 높은 확률을 가진 클래스 선택
            correct += prediction.eq(label.view_as(prediction)).sum().item() # 정확한 예측 수 누적
    
    # 손실 평균 및 정확도 계산
    test_loss /= (len(testloader.dataset) / BATCH_SIZE)
    test_accuracy = 100. * correct / len(testloader.dataset)
    return test_loss, test_accuracy


for epoch in range(1, EPOCHS + 1):
    train(model, trainloader, optimizer, log_interval = 200)
    test_loss, test_accuracy = evaluate(model, testloader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))

torch.save(model, 'modelcnn.pt')