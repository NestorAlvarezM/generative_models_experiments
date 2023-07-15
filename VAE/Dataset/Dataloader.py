import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

batch_size=9
training_data=CustomImageDataset("celeb_A/img_align_celeba")
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
train_images = next(iter(train_dataloader))
batch_array = np.transpose(train_images, (0, 2, 3, 1))
fig, axes = plt.subplots(3, 3)
for i, ax in enumerate(axes.flat):
    ax.imshow(batch_array[i])
    ax.axis('off')
#TODO: show on the plot info like img size
plt.tight_layout()
plt.show()
