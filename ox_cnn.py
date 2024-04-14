from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

EPOCHS = 10

trans = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])

testset = torchvision.datasets.ImageFolder(root = "/home/min/deep-learning-prac/pytorch-cnn/test_dataset", transform = trans)

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

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

print(model)

model2 = torch.load("modelcnn1.pt", map_location=torch.device('cpu'))
model2.eval()
sample_idx = 0
img, label = testset[sample_idx]
#image = img.view(-1,9680)
#image = img.reshape(100, 100)
#image = transforms.ToPILImage()(img)
image_pil = transforms.ToPILImage()(img)
image_tensor = transforms.ToTensor()(image_pil)
out = model2(image_tensor.unsqueeze(0))
#out = model2(image)
_,predict = torch.max(out.data, 1)
plt.title("label: {} actual: {}".format(label,predict.item()))
plt.axis("off")
plt.imshow(img.squeeze(), cmap="gray")
plt.savefig('output.png',facecolor = 'w')
plt.show()
