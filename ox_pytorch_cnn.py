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
# from ox_crop import imgcut


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

imgdir = "/home/min/pytorch-ox/test-image.jpg"
inferdir = "/home/min/pytorch-ox/inferance"

def get_image_files(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_files.append(os.path.join(root, file))
    return image_files

def imgcut(input_dir):
# 이미지 불러오기
    img = Image.open(input_dir)
    # 이미지 회전하기
    img = img.rotate(270)

    imgresized = img.resize((400,400))

    # (left, top, right, bottom) 영역을 지정하여 자르기
    left =[111,180,250,110,176,250,102,176,253]
    top = [112,180,250]
    img1 = imgresized.crop((left[0],top[0],left[0]+35,top[0]+35)) #image1
    img2 = imgresized.crop((left[1],top[0],left[1]+35,top[0]+35)) #image2
    img3 = imgresized.crop((left[2],top[0],left[2]+35,top[0]+35)) #image3
    img4 = imgresized.crop((left[3],top[1],left[3]+30,top[1]+30)) #image4
    img5 = imgresized.crop((left[4],top[1]-4,left[4]+40,top[1]+36)) #image5
    img6 = imgresized.crop((left[5],top[1]-4,left[5]+40,top[1]+36)) #image6
    img7 = imgresized.crop((left[6],top[2]-5,left[6]+40,top[2]+35)) #image7
    img8 = imgresized.crop((left[7],top[2]-5,left[7]+42,top[2]+35)) #image8
    img9 = imgresized.crop((left[8],top[2]-5,left[8]+42,top[2]+35)) #image9

    # img9.show()
    # 이미지 저장하기(jpeg)
    img1.save('/home/min/pytorch-ox/inferance/1.jpg')
    img2.save('/home/min/pytorch-ox/inferance/2.jpg')
    img3.save('/home/min/pytorch-ox/inferance/3.jpg')
    img4.save('/home/min/pytorch-ox/inferance/4.jpg')
    img5.save('/home/min/pytorch-ox/inferance/5.jpg')
    img6.save('/home/min/pytorch-ox/inferance/6.jpg')
    img7.save('/home/min/pytorch-ox/inferance/7.jpg')
    img8.save('/home/min/pytorch-ox/inferance/8.jpg')
    img9.save('/home/min/pytorch-ox/inferance/9.jpg')
    

BATCH_SIZE = 10
EPOCHS = 30

trans = transforms.Compose([
    transforms.Resize((100, 100)),  
    # transforms.RandomHorizontalFlip(),  # 랜덤하게 이미지를 좌우 반전합니다.
    # transforms.RandomRotation(15),  # 이미지를 최대 15도까지 랜덤하게 회전시킵니다.
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 색상을 랜덤하게 변경합니다.
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CustomDataset(Dataset):
    def __init__(self, image_files, transform=trans):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')  # 이미지를 열고 RGB로 변환합니다.
        if self.transform:
            image = self.transform(image)
        return image

# 이미지 파일들을 리스트로 가져옵니다.
image_files = get_image_files(inferdir)

# 사용자 정의 데이터셋 객체를 생성합니다.
infer_dataset = CustomDataset(image_files, transform=trans)

# DataLoader를 이용하여 데이터를 불러옵니다.
# infer_loader = DataLoader(infer_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 추론 코드 이후 부분은 그대로 유지됩니다.

# testset = torchvision.datasets.ImageFolder(root = "/home/min/pytorch-ox/inferance", transform = trans)
    
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

model2 = torch.load("modelcnn.pt", map_location=torch.device('cpu'))
model2.eval()

figure = plt.figure(figsize=(10, 5))
cols, rows = 5, 2
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(infer_dataset), size=(1,)).item()
    img = infer_dataset[sample_idx]
    img = img.unsqueeze(0)  # 이미지 데이터를 GPU로 보내지 않습니다.
    figure.add_subplot(rows, cols, i)
    out = model2(img.cpu())  # 모델에 입력 데이터를 CPU로 보냅니다.
    _, predict = torch.max(out.data, 1)
    img = img.squeeze()  # 이미지 차원 축소
    img = img.numpy()  # 텐서를 넘파이 배열로 변환
    img = np.transpose(img, (1, 2, 0))  # 이미지 차원 재배열
    plt.title("actual: {}".format(predict.item()))
    plt.axis("off")
    plt.imshow(img)
plt.savefig('output.png', facecolor='w')
plt.show()
