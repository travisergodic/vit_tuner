import os

import numpy as np 
from PIL import Image
from torch.utils.data import Dataset


class DataframeDataset(Dataset):
    def __init__(self, df, filename_col, y_cols, image_dir, image_transform=None):
        super().__init__()
        self.df = df
        self.filename_col = filename_col
        self.y_cols = y_cols
        self.image_dir = image_dir
        self.image_transform = image_transform 

    def __getitem__(self, index):
        if len(self.y_cols) == 1:
            label = int(self.df.loc[index, self.y_cols[0]])

        else:
            label = self.df.loc[index, self.y_cols].astype("int64").values 

        image_path = os.path.join(self.image_dir, self.df.loc[index, self.filename_col])
        image = Image.open(image_path).convert("L").convert("RGB")
        
        if self.image_transform is not None:
            image = self.image_transform(image)

        return {"data": image, "label": label, "name": os.path.basename(image_path)}

    def __len__(self):
        return len(self.df)