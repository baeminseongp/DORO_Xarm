# DORO_Xarm

1) device = cpu
2) code의 대략적인 설명
 카메라에서 사진을 받아온다
-> oxcrop코드에서 9등분하여 inferance파일에 저장
-> inferance파일에서 받아와 customDataset 제작
-> 이후 추론 진행
3) 학습 신경망(CNN)
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
