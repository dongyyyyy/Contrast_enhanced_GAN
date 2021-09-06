import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def read_dataset(self):
        none_enhance_image_list = []
        enhance_image_list = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            image_list = os.listdir(signals_path)
            image_list.sort()
            current_path = dataset_folder
            # print('current_path ==> ', current_path)
            enhance_path = current_path.split('/')
            enhance_path[-3] = 'enhanced_png'
            enhance_path = '/'.join(enhance_path)+'/'
            for filename in image_list:
                none_enhance_image_list.append(dataset_folder+filename)
                enhance_image_list.append(enhance_path+filename)
        return none_enhance_image_list, enhance_image_list

    def __init__(self, dataset_list,transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.dataset_list = dataset_list
        self.none_enhanced_image_list, self.enhanced_image_list = self.read_dataset()

    def __getitem__(self, index):
        none_enhanced_image = self.transform(Image.open(self.none_enhanced_image_list[index % len(self.none_enhanced_image_list)]).convert('L'))

        enhanced_image = self.transform(Image.open(self.enhanced_image_list[index % len(self.enhanced_image_list)]).convert('L'))

        # print('none_enhanced_image shape = ',none_enhanced_image.shape)
        return {'n_enhanced_image': none_enhanced_image, 'enhanced_image': enhanced_image}

    def __len__(self):
        return max(len(self.none_enhanced_image_list), len(self.enhanced_image_list))