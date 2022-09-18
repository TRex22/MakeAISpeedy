# Based on my work from: https://github.com/TRex22/deeplabv3-custom-training
# But extended to more domains

import os
import gc
import time
import json
import numpy as np
from pathlib import Path

import torch
from torch import optim
from torch import nn
import torchvision
from torch.utils.data import DataLoader

import tqdm

import torchvision.transforms as T
from torchvision import models
from torchinfo import summary

from torch import autocast

# from collections import namedtuple

import sys
sys.path.insert(1, '../references/segmentation/')

from sneaker_dataset import SneakerDataset

class TestModel2(nn.Module):
  def __init__(self, config):
    super(TestModel2, self).__init__()

    self.config = config

    # Grad-CAM interface
    self.target_layer = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.target_layers = [self.target_layer]

    # self.in_features = nn.Linear(122880, 10)

    num_classes = self.config['num_classes']

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(config["input_size"][0], 32, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      self.target_layer,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      nn.Flatten(),
      # nn.Dropout(p=0.2),
      # self.in_features,
      # nn.Dropout(p=0.2)
      # nn.Linear(10, config[''])
      nn.Linear(460800, 10),
      nn.Linear(10, num_classes)
    )

    # self.cnn_stack = compute_output_layers(config, self.cnn_stack, output_layer_node_size=10)

    # fc.in_features
    # fc = namedtuple('fc', 'in_features')
    # fc.in_features = self.in_features
    # self.fc = fc

  def forward(self, x):
    logits = self.cnn_stack(x)

    # if self.config["normalise_output"]:
    #   logits = F.normalize(logits, dim = 0)

    return logits

# Reference Code
# import presets
# import utils
# from coco_utils import get_coco
# import transforms as T

# import deeplabv3_metrics

SMOOTH = 1e-4 #1e-6 # Beware Float16!

################################################################################
# Helper Methods                                                               #
################################################################################
def create_folder(path):
  Path(path).mkdir(parents=True, exist_ok=True)

def check_if_file_exists(filepath):
  return Path(filepath).is_file()

def compute_base_save_path(config):
  path = f'{config["save_path"]}/{config["selected_model"]}_{int(time.time())}/'
  create_folder(path)
  return path

# You can use the built-in python logger but
# Id like to keep existing functionality
def log(text, config):
  print(text)

  if config["log_to_file"]:
    base_path = config["base_path"]
    log_path = f'{base_path}/console_output.txt'

    with open(log_path, 'a') as f:
      f.write(f'{text}\n')

def clear_gpu():
  # If you need to purge memory
  gc.collect() # Force the Training data to be unloaded. Loading data takes ~10 secs
  # time.sleep(15) # 30
  # torch.cuda.empty_cache() # Will nuke the model in-memory
  torch.cuda.synchronize() # Force the unload before the next step

def fetch_device(config):
  print(f'Cuda available? {torch.cuda.is_available()}')
  dev = torch.device('cpu')
  summary_dev = 'cpu'

  if config["summary_device_name"] == "cpu":
    return [dev, summary_dev]

  if torch.cuda.is_available():
    dev = torch.device('cuda')
    summary_dev = 'cuda'

  return [dev, summary_dev]

# Will either open the config path or get config from model checkpoint
def open_config(path):
  try:
    # Remove comments first
    raw_json = ""
    with open(path) as f:
      for line in f:
        line = line.partition('//')[0]
        line = line.rstrip()
        raw_json += f"{line}\n"

    config = json.loads(raw_json)
    epoch = 0
    model_path = None
  except:
    checkpoint = torch.load(path)
    config = checkpoint['args']
    epoch = checkpoint['epoch'] + 1
    model_path = path

  config["save_path"] = f'{config["save_path"]}/{config["dataset"]}'
  create_folder(config["save_path"])

  return [config, epoch, model_path]

def create_folder(path):
  Path(path).mkdir(parents=True, exist_ok=True)

def save_csv(file_path, csv_data):
  with open(file_path, 'a') as f:
    f.write(f'{csv_data}\n')

def save_json(file_path, data):
  active_file = open(file_path, 'w', encoding='utf-8')
  json.dump(data, active_file)
  active_file.write("\n")
  active_file.close()

# https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1)
  return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
def show_batch(dl):
  """Plot images grid of single batch"""
  for images, labels in dl:
    fig,ax = plt.subplots(figsize = (16,12))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
    break

################################################################################
# Image Transforms                                                             #
################################################################################
def standard_image_transforms(config):
  mean = (0.485, 0.456, 0.406) # Taken from COCO reference
  std = (0.229, 0.224, 0.225)

  # https://pytorch.org/vision/stable/transforms.html
  transforms_arr = T.Compose(
    [
      T.RandomCrop(200), # 480 # 513 # 520
      # .CenterCrop(10)
      T.PILToTensor(),
      T.Resize([config["input_size"][1], config["input_size"][2]]),
      # T.Grayscale(), # if bw
      T.ConvertImageDtype(torch.float),
      # T.Normalize(mean=mean, std=std), # Broken
    ]
  )

  return transforms_arr

# https://discuss.pytorch.org/t/normalising-images-in-cityscapes-using-mean-and-std-of-imagenet/120556
def cityscapes_transforms():
  mean = (0.485, 0.456, 0.406) # Taken from COCO reference
  std = (0.229, 0.224, 0.225)

  # https://stackoverflow.com/questions/49356743/how-to-train-tensorflows-deeplab-model-on-cityscapes
  transforms_arr = T.Compose(
    [
      T.RandomCrop(460), # 480 # 513 # 520
      T.PILToTensor(),
      T.ConvertImageDtype(torch.float),
      T.Normalize(mean=mean, std=std),
    ]
  )

  return transforms_arr

def rescale_pixels(image, scale=1./255.):
  return image * scale

def undo_rescale_pixels(image, scale=1./255.):
  return image * (1/scale)

################################################################################
# Dataset Loading                                                              #
################################################################################
def cityscapes_collate(batch):
  images, targets = list(zip(*batch))

  # images = np.array([(rescale_pixels(i.numpy())) for i in images]) # Cant rescale if want to maintain compatibility
  images = np.array([(i.numpy()) for i in images])
  targets = np.array([(t.numpy()) for t in targets])

  return torch.from_numpy(images), torch.from_numpy(targets)

def load_dataset(config, target_class, category_list=None, batch_size=1):
  root = config["dataset_path"]

  # TODO: Make this configurable again
  # dataset = SneakerDataset(root, split=target_class, transforms=cityscapes_transforms())
  dataset = SneakerDataset(root, target_class=target_class, transforms=standard_image_transforms(config))

  sample_size = len(dataset) * config["sample_percentage"]
  if target_class == 'train':
    if sample_size < batch_size:
      sample_size = len(dataset)

    subset_idex = list(range(int(sample_size))) # TODO: Unload others
    subset = torch.utils.data.Subset(dataset, subset_idex)

    if config["dataset"] == "cityscapes":
      dataloader = DataLoader(subset, batch_size=batch_size, shuffle=config["shuffle"], drop_last=config["drop_last"], collate_fn=cityscapes_collate, num_workers=config["train_num_workers"], pin_memory=config["pin_memory"])
    else:
      dataloader = DataLoader(subset, batch_size=batch_size, shuffle=config["shuffle"], drop_last=config["drop_last"], num_workers=config["train_num_workers"], pin_memory=config["pin_memory"])
  else:
    if config["dataset"] == "cityscapes":
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=config["shuffle"], drop_last=config["drop_last"], collate_fn=cityscapes_collate, num_workers=config["val_num_workers"], pin_memory=config["pin_memory"])
    else:
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=config["shuffle"], drop_last=config["drop_last"], num_workers=config["val_num_workers"], pin_memory=config["pin_memory"])

  # print(f'Number of data points for {target_class}: {len(dataloader)}')
  return [dataset, dataloader]

def fetch_category_list(config):
  if config["dataset"] == "COCO16":
    return  [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16] # New list of categories
  elif config["dataset"] == "COCO21":
    return [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72] # Original List
  elif config["dataset"] == "cityscapes" or config["dataset"] == "fromgames":
    # return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, -1] # Cityscapes
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33] # Cityscapes

# COCO Dataset
# train_image_path = '/data/data/coco/data_raw/train2017'
# val_image_path = '/data/data/coco/data_raw/val2017'
# train_annotation_path = '/data/data/coco/zips/annotations/annotations/instances_train2017.json'
# val_annotation_path = '/data/data/coco/zips/annotations/annotations/instances_val2017.json'

# train_dataset = torchvision.datasets.CocoDetection(train_image_path, train_annotation_path)
# val_dataset = torchvision.datasets.CocoDetection(val_image_path, val_annotation_path)
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, persistent_workers=False)
def load_coco(root, target_class, category_list=None):
  # Using reference code
  # See Readme.md for new category list

  if category_list is None:
    category_list = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72] # Default
    # category_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21] # New

  if target_class == 'train':
    transforms = presets.SegmentationPresetTrain(base_size=520, crop_size=480)
  else:
    transforms = presets.SegmentationPresetEval(base_size=520)

  return get_coco(root, target_class, transforms, category_list=category_list)

################################################################################
# Model Functions                                                              #
################################################################################
# Used to load pre-trained models or return None (So we can construct the new ones)
def fetch(config):
  # return models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=21)

  model = config["selected_model"]

  if model == 'resnet18':
    model =  models.resnet18()
  elif model == 'resnet34':
    model =  models.resnet34()
  elif model == 'resnet50':
    model =  models.resnet50()
  elif model == 'resnet101':
    model =  models.resnet101()
  elif model == 'resnet152':
    model =  models.resnet152()
  elif model == 'simple_cnn':
    return TestModel2(config)
  else:
    model = None

  if model == None:
    return None

  num_classes = config['num_classes']
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, num_classes)

  # torchvision.models.segmentation.fcn_resnet50(pretrained: bool = False, progress: bool = True, num_classes: int = 21, aux_loss: Optional[bool] = None, **kwargs: Any)
  return model

def xavier_uniform_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    gain = nn.init.calculate_gain('relu')
    nn.init.xavier_uniform_(layer.weight, gain=gain)

def initialise_model(dev, config, pretrained=False, num_classes=21, randomise=True):
  if config["selected_model"] == 'deeplabv3_resnet101':
    model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, num_classes=num_classes)
  elif config["selected_model"] == 'deeplabv3_resnet50':
    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
  else:
    raise RuntimeError("Invalid model selected.")

  # Randomise weights
  if randomise:
    model.apply(xavier_uniform_init)

  model = model.to(dev)

  # Reference code uses SGD
  # https://www.programmersought.com/article/22245145270/
  if config["opt_function"] == 'ADAM':
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=config["betas"], eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    print('ADAM Optimizer is selected!')
  # if config["opt_function"] == 'RMSprop':
    # https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
  else:
    opt = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    print('SGD Optimizer is selected!')

  return [model, opt]

def select_optimizer(config, model):
  if config["opt_function"] == 'ADAM':
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=config["betas"], eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    print('ADAM Optimizer is selected!')
  # if config["opt_function"] == 'RMSprop':
    # https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
  else:
    opt = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    print('SGD Optimizer is selected!')

  return opt

def load(model, device, path, opt=None, show_stats=True):
  # Load model weights
  # Training crashed when lr dropped to complex numbers

  # TODO: Load optimizer
  print(f'Loading model from: {path}')
  checkpoint = torch.load(path)
  epoch = checkpoint['epoch']
  model.load_state_dict(checkpoint['model'], strict=False)

  if opt:
    opt.load_state_dict(checkpoint['optimizer'])

  model.to(device)

  print(f'Model loaded into {device}!')

  if show_stats:
    model_stats = summary(model, device=device)

  return model, opt, epoch

# Built to be compatible with reference code
def save(model, opt, scaler, lr_scheduler, epoch, config, save_path):
  checkpoint = {
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "scaler": scaler.state_dict(),
    "epoch": epoch,
    "config": config,
  }

  if lr_scheduler is not None:
    checkpoint["lr_scheduler"] = lr_scheduler.state_dict()

  torch.save(checkpoint, os.path.join(save_path, f"model_{epoch}.pth"))

################################################################################
# Loss Functions                                                               #
################################################################################
# Based on: https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
def compute_iou1(output, target):
  intersection = torch.logical_and(output, target)
  union = torch.logical_or(output, target)
  iou_score = torch.sum(intersection) / torch.sum(union)
  # print(f'IoU is {iou_score}')

  return iou_score

# Based on: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy/script
def compute_iou2(outputs: torch.Tensor, labels: torch.Tensor):
  # You can comment out this line if you are passing tensors of equal shape
  # But if you are passing output from UNet or something it will most probably
  # be with the BATCH x 1 x H x W shape
  outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

  intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
  union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

  iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

  thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

  return thresholded.mean() #thresholded  # Or thresholded.mean() if you are interested in average across the batch

# TODO: Convert to Tensor
# Based on: https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/utils.py
def compute_iou3(output, target, num_classes):
  pred = np.asarray(output.cpu(), dtype=np.uint8).copy()
  mask = np.asarray(target.cpu(), dtype=np.uint8).copy()
  # 255 -> 0
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_classes, range=(1, num_classes))
  (area_pred, _) = np.histogram(pred, bins=num_classes, range=(1, num_classes))
  (area_mask, _) = np.histogram(mask, bins=num_classes, range=(1, num_classes))
  area_union = area_pred + area_mask - area_inter

  # return (area_inter, area_union)
  return ((area_inter + SMOOTH) / (area_union + SMOOTH)).mean()

# TODO: IOU SKLearn

# https://towardsdatascience.com/choosing-and-customizing-loss-functions-for-image-processing-a0e4bf665b0a
# https://stackoverflow.com/questions/47084179/how-to-calculate-multi-class-dice-coefficient-for-multiclass-image-segmentation
# Dice Co-Efficient
def dice_coef(y_true, y_pred, epsilon=SMOOTH): # 1e-6 wont work for float16
  # Altered Sorensen–Dice coefficient with epsilon for smoothing.
  y_true_flatten = y_true.to(torch.bool)
  y_pred_flatten = y_pred.to(torch.bool)

  if not torch.sum(y_true_flatten) + torch.sum(y_pred_flatten):
    return 1.0

  return (2. * torch.sum(y_true_flatten * y_pred_flatten)) / (torch.sum(y_true_flatten) + torch.sum(y_pred_flatten) + epsilon)

################################################################################
# Train and Eval                                                               #
################################################################################
def simple_eval(config, model, device, loss_fn, eval_dataloader):
  model.eval()

  pbar = tqdm.tqdm(total=len(eval_dataloader))

  sum_of_loss = 0.0
  sum_of_accuracy = 0.0
  number_of_iterations = 0

  for xb, yb in eval_dataloader:
    input = xb.to(device)
    prediction = model(input)

    # output = prediction.argmax(1)
    # output = prediction['out']

    target = yb.to(device)
    loss = loss_fn(prediction, target)

    acc = accuracy(prediction, target)

    sum_of_loss += loss
    sum_of_accuracy += acc
    number_of_iterations += 1

    pbar.update(1)

    # print(torch.cuda.memory_summary())

    del input
    del prediction
    del target

  final_loss = sum_of_loss / number_of_iterations
  final_accuracy = sum_of_accuracy / number_of_iterations
  return [final_loss, final_accuracy]

def simple_train(config, model, device, loss_fn, opt, scaler, train_dataloader):
  model.train()

  pbar = tqdm.tqdm(total=len(train_dataloader))
  sum_of_loss = 0.0
  sum_of_accuracy = 0.0
  number_of_iterations = 0

  for xb, yb in train_dataloader:
    input = xb.to(device)
    target = yb.to(device)
    number_of_iterations += 1

    if opt is not None:
      with autocast(config["summary_device_name"], enabled=config["mixed_precision"], cache_enabled=config["cache_enabled"]):
        prediction = model(input)

        # output = prediction['out']

        loss = loss_fn(prediction, target)
        acc = accuracy(prediction, target)

      # backprop
      # loss.backward()
      scaler.scale(loss).backward()
      if config["clip_grad_norm"]:
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(opt)

        # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
        # You may use the same value for max_norm here as you would without gradient scaling.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_max_norm"])
        # torch.nn.utils.clip_grad_norm_(model.parameters())

      scaler.step(opt)
      scaler.update()
      # opt.step()

      for param in model.parameters(): # Optimisation to save n operations
        param.grad = None
    else:
      with autocast(config["summary_device_name"], enabled=config["mixed_precision"], cache_enabled=config["cache_enabled"]):
        prediction = model(xb)
        loss = loss_func(prediction.flatten(), yb) # TODO: Automate for two outputs
        acc = accuracy(prediction, target)

    sum_of_loss += loss
    sum_of_accuracy += acc

    del input
    del prediction
    del target

  final_loss = sum_of_loss / number_of_iterations
  final_accuracy = sum_of_accuracy / number_of_iterations
  return [final_loss, final_accuracy, model]

def simple_loop(config, model, device, loss_fn, opt, scaler, train_dataloader, test_dataloader):
  pbar = tqdm.tqdm(total=config["epochs"])

  train_loss = 0.0
  val_loss = 0.0

  save_csv(f'{config["base_path"]}/train_data.csv', 'epoch,train_loss,train_accuracy')
  save_csv(f'{config["base_path"]}/test_data.csv', 'epoch,test_loss,test_accuracy')

  for epoch in range(config["epochs"]):
    # TODO:
    # with torch.cuda.amp.autocast(enabled=True, cache_enabled=True): # TODO: cache_enabled

    test_loss, test_accuracy = simple_eval(config, model, device, loss_fn, test_dataloader)
    train_loss, train_accuracy, model = simple_train(config, model, device, loss_fn, opt, scaler, train_dataloader)
    # val_loss = simple_eval(config, model, device, loss_fn, val_dataloader)

    # Save Looped Data
    save_csv(f'{config["base_path"]}/train_data.csv', f'{epoch},{train_loss},{train_accuracy}')
    save_csv(f'{config["base_path"]}/test_data.csv', f'{epoch},{test_loss},{test_accuracy}')

    pbar.update(1)
    std_out = f'Epoch {epoch} train loss: {train_loss}, train_acc: {train_accuracy}, test loss: {test_loss}, test_acc: {test_accuracy}'
    pbar.write(std_out)
    log(std_out, config)

  test_loss, test_accuracy = simple_eval(config, model, device, loss_fn, test_dataloader)
  pbar.write(f'Final Test loss: {test_loss}')
  return [model, train_loss, train_accuracy, test_loss, test_accuracy]
