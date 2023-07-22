import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import visualize_dataloader

class CustomImageDataset(Dataset):
    """
    Custom Image Dataset

     Parameters:
         img_dir (str): The path to the directory containing the images.

     Attributes:
         img_dir (str): The path to the directory containing the images.
         img_names (list of str): A list of image filenames in the dataset.
         transform (torchvision.transforms.Compose): A composition of image transformations
             to be applied to the images loaded from the dataset.

     Returns:
         image: Image on the format defined by the transformations.
     """
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transform

    def __len__(self):
        """this gives the len of the dataset, is required by the pytorch dataloader"""
        return len(self.img_names)

    def __getitem__(self, idx):
        """this gives a specific example of index idx, is requeried by the dataloader class"""
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

if __name__=='__main__':
    # run main to test that images are being loaded correctly


    """this transformation si the same as the original but without 
    To.Tensor that takes the numpy array and transforms it into a tensor
    and without the Normalize because this transformation are importante to the model
    but we can`t visualize them"""

    transform=transforms.Compose([
        transforms.CenterCrop(224),
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print("transform",transform)
    """to show 9 images"""
    batch_size=9
    """The image dataset is asociated with the folder of out images and is decided which transformation to apply"""
    training_data=CustomImageDataset("celeb_A/img_align_celeba")
    """Data transformation is actualized to one that is visible"""
    training_data.transform=transform
    """Now the dataloader defines how many images we will be sampling"""
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    """we get and example, the object train_dataloader is an iterable so we get one element"""
    train_images = next(iter(train_dataloader))

    """given this elements we change the channels on the image from (batch_size,channels,width,height) to
    (batch_size,width,height) it is mostly a convention that pytorch uses the images on the former way but common
    libraries like cv2 o matplotlib recognizes imagen on the former way"""
    batch_array = np.transpose(train_images, (0, 2, 3, 1))
    """now we simply plot the images"""
    visualize_dataloader(batch_array)

