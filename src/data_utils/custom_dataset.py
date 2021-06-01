import os
from typing import Tuple, Sequence, Callable, List
from PIL import Image
import pandas as pd
import numpy as np
import csv

from sklearn.preprocessing import OneHotEncoder

from torch import Tensor
from torchvision import transforms

from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder


def make_labels(dir: os.PathLike = 'data') -> None:
    categories = [category for category in os.listdir(dir)
                           if os.path.isdir(os.path.join(dir, category))]
    cat_data = [data for data in categories
                     for _ in range(len(os.listdir(os.path.join(dir, data))))]
    temp_df = pd.DataFrame(cat_data, columns=['category'])

    temp_cat = temp_df[['category']]
    onehot_encoder = OneHotEncoder()
    label_encoded = onehot_encoder.fit_transform(temp_cat).toarray()

    columns = [cat_data]
    print(columns[:10])
    labels = [label_encoded[:, i] for i in range(len(categories))]
    columns.extend(labels)
    print(columns)

    final_data = list(zip(*columns))
    print(final_data[:10])
    
    final_df = pd.DataFrame(final_data, columns=['category', '0', '1'])
    final_df.to_csv('data/labels.csv')


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
                self.labels[int(row[0])] = [row[1], row[2:]]
        
        self.image_ids = list(self.labels.keys())

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        category = self.labels.get(image_id)[0]

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


# Testing ImageFolder()
class FaceDataset_test(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.dataset = ImageFolder(root=self.dir, transform=self.transforms, target_transform=None)

        return self.dataset

    def __getitem__(self, index:int) -> Tuple[Tensor]:
        pass

    def __iter__(self):
        pass

    def __len__(self) -> int:
        pass



def split_dataset(dataset_size: int, split_rate: float, num_seed: int = 42) -> Tuple[List]: 
    indices = list(range(dataset_size))
    split_indices = int(np.floor(split_rate * dataset_size))

    np.random.seed(num_seed)
    np.random.shuffle(indices)
    test_indices, train_indices = indices[:split_indices], indices[split_indices:]
    print('\nlen(train_indices) =', len(train_indices), ', len(test_indices) =', len(test_indices), '\n')

    return train_indices, test_indices

# test
def split_dataset_test(dataset, dataset_size, split_rate) -> Tuple[List]:
    train_size = int((1 - split_rate) * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


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
    print(face[978])