import os
from typing import Tuple, Sequence, Callable
from PIL import Image
import pandas as pd
import numpy as np
import csv

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

def make_labels(dir: os.PathLike = 'data') -> None:
    categories = [category for category in os.listdir(dir)
                           if os.path.isdir(os.path.join(dir, category))]
    labels_df = pd.DataFrame(columns=['category', 'label'], index=None)

    i = 0
    for label, category in enumerate(categories):
        for _ in range(len(os.listdir(os.path.join(dir, category)))):
            labels_df.loc[i] = [category, label]
            i += 1

    print(labels_df.head())
    print(labels_df.shape)
    labels_df.to_csv('data/labels.csv')

    

class FaceDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = [row[1], int(row[2])]
        
        self.image_ids = list(self.labels.keys())

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        category = self.labels.get(image_id)[0]
        print(category)

        start_points = {
            'jeungen': 0, 
            'leeyeoreum': len(os.listdir(os.path.join(self.dir, 'jeungen'))),
            }

        if category == 'jeungen':
            image = Image.open(
                os.path.join(
                    self.dir, category, f'train_{image_id}.jpg')).convert('RGB')
        elif category == 'leeyeoreum':
            image = Image.open(
                os.path.join(
                    self.dir, 
                    category,
                    f"train_{image_id - start_points['leeyeoreum']}.jpg"
                    )).convert('RGB')
        target = np.array(self.labels.get(image_id)[1]).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)


if __name__ == '__main__':
    make_labels()

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    face = FaceDataset('data', 'data/labels.csv', transforms)
    print(face[0])