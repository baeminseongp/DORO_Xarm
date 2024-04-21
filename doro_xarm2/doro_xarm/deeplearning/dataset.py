from PIL import Image
from torch.utils.data import Dataset

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
        image = Image.open(img_path)
        if self.transform:
            # 이미지에 전처리(transform)를 적용합니다.
            image = self.transform(image)
        return image

