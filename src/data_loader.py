import os
import cv2
from torch.utils.data import Dataset
import albumentations as A

class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform or A.Compose([
            A.Resize(224,224), A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast()
        ])
        self.samples = []
        for lbl in sorted(os.listdir(root_dir)):
            cls_path = os.path.join(root_dir, lbl)
            for img in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, img), lbl))
        self.label2idx = {l:i for i,l in enumerate(sorted({lbl for _,lbl in self.samples}))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, lbl = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = img.transpose(2,0,1) / 255.0
        return img.astype('float32'), self.label2idx[lbl]
