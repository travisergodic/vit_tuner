import os

import torch
from PIL import Image
from tensordict import TensorDict
from torch.utils.data import Dataset

from src.registry import DATASET


@DATASET.register("single_task")
class SingleTaskDataset(Dataset):
    def __init__(self, df, filename_col, y_col, image_dir, image_transform=None):
        super().__init__()
        self.df = df
        self.filename_col = filename_col
        self.y_col = y_col
        self.image_dir = image_dir
        self.image_transform = image_transform 

    def __getitem__(self, index):
        label = int(self.df.loc[index, self.y_col[0]])

        image_path = os.path.join(self.image_dir, self.df.loc[index, self.filename_col])
        image = Image.open(image_path).convert("RGB")
        
        if self.image_transform is not None:
            image = self.image_transform(image)

        return {"data": image, "target": label, "name": os.path.basename(image_path)}

    def __len__(self):
        return len(self.df)
    

@DATASET.register("multi_task")
class MultiTaskDataset(Dataset):
    def __init__(self, df, filename_col, y_col, image_dir, image_transform=None):
        super().__init__()
        self.df = df
        self.filename_col = filename_col
        self.y_col = y_col
        self.image_dir = image_dir
        self.image_transform = image_transform 

    def __getitem__(self, index):
        label = self.df.loc[index, self.y_col].astype("int64").to_dict() 

        image_path = os.path.join(self.image_dir, self.df.loc[index, self.filename_col])
        image = Image.open(image_path).convert("L").convert("RGB")
        
        if self.image_transform is not None:
            image = self.image_transform(image)

        return {"data": image, "target": label, "name": os.path.basename(image_path)}

    def __len__(self):
        return len(self.df)
    

def multitask_collate_fn(batch):
    data = torch.stack([ele["data"] for ele in batch])
    names = [item["name"] for item in batch]
    tasks = batch[0]["target"].keys()
    targets=TensorDict(
        {task: torch.tensor([ele["target"][task] for ele in batch]) for task in tasks}, batch_size=len(names)
    )
    return {"data": data, "target": targets, "name": names}