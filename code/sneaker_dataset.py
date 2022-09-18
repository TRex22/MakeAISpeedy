import os
from PIL import Image
from torchvision.io import read_image

import torch
import torchvision

# https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec # VaporWave
# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
class SneakerDataset:
  def __init__(self, base_path, target_class='train', transforms=None, data_split=0.8):
    self.base_path = base_path
    self.train_path = f'{self.base_path}/train/'
    self.test_path = f'{self.base_path}/test/'

    self.transforms = transforms

    self.classes = os.listdir(self.train_path)

    self.train_image_paths = []
    self.test_image_paths = []

    self.target_class = target_class

    for klass in self.classes:
      train_files = os.listdir(f'{self.train_path}/{klass}')
      test_files = os.listdir(f'{self.test_path}/{klass}')

      for idx in range(len(train_files)):
        self.train_image_paths +=  [f'{klass}/{train_files[idx]}']

      for idx in range(len(test_files)):
        self.test_image_paths += [f'{klass}/{test_files[idx]}']

    self.split = data_split # 80% for val, as a general rule

    self.test_count = len(self.test_image_paths)
    test_start_idex = int(self.test_count - (self.test_count * (1.0 - self.split)))

    if self.target_class == 'train':
      self.image_paths = self.train_image_paths
      self.target_path = 'train'
    # elif self.target_class == 'val':
    #   self.image_paths = self.test_image_paths[0:test_start_idex]
    #   self.target_path = 'test'
    elif self.target_class == 'test':
      # self.image_paths = self.test_image_paths[test_start_idex:-1]
      self.image_paths = self.test_image_paths
      self.target_path = 'test'

    self.count = len(self.image_paths)

  def __len__(self):
    return self.count

  def __getitem__(self, idx):
    label = 1 # 'nike'
    # str_label = 'nike'

    if self.image_paths[idx].__contains__('nike'):
      label = 1 # 'nike'
      # str_label = 'nike'
    elif self.image_paths[idx].__contains__('adidas'):
      label = 2 # 'adidas'
      # str_label = 'adidas'
    elif self.image_paths[idx].__contains__('converse'):
      label = 3 # 'converse'
      # str_label = 'converse'

    img_path = f'{self.base_path}/{self.target_path}/{self.image_paths[idx]}'

    # read_image(img_path) #.float() # .permute(0, 2, 1).float() # .permute(0, 2, 1)
    # image = read_image(img_path) # TODO: Device and better read

    # PIL version
    image = Image.open(img_path)
    # image = torchvision.transforms.functional.to_tensor(image)

    if self.transforms:
      # image = image * 1./255.
      transformed_image = self.transforms(image)

    del image
    return transformed_image, torch.tensor(label, dtype=torch.long)
