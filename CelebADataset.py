from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_df, transform=None):
        self.img_dir = img_dir
        self.attr_df = attr_df
        self.transform = transform
        self.filenames = attr_df.index.values  # aligned with attributes

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        attrs = self.attr_df.loc[img_name].values.astype("float32")

        if self.transform:
            image = self.transform(image)

        return image, attrs