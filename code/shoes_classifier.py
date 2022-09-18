import gc
import sys
import time
import tqdm

import numpy as np

import torch
from torch import nn

from torchinfo import summary

import custom_utils
from custom_utils import log

from PIL import Image

config = {
  "save_identifier": 'mixed-precision amp',
  "selected_model": 'resnet18', #'resnet34', #'resnet18', # 'resnet50', # 'resnet101', # 'resnet152', # 'simple_cnn',
  "epochs": 100, # 5
  "lr": 0.001,
  "epsilon": 1e-03,
  "betas": [0.9, 0.999],
  "momentum": 0.9,
  "train_batch_size": 100,
  "stats_batch_size": 100,
  "input_size": [3, 240, 240], #[3, 120, 120], #[3, 240, 240],
  "summary_device_name": "cuda", #"cpu", #"cuda",
  "cache_enabled": True,
  "mixed_precision": True,
  "clip_grad_norm": False,
  "cuda": True,
  "dataset_path": "/home/trex22/development/AIExpo2022Poster/data/shoes",
  "sample_percentage": 1.0,
  "dataset": 'shoes',
  "shuffle": True,
  "drop_last": False,
  "train_num_workers": 1, # TODO: Persistent workers
  "val_num_workers": 1,
  "pin_memory": True,
  "opt_function": "ADAM", # "SGD", "ADAM"
  "amsgrad": False,
  "weight_decay": 1e-03,
  "save_path": '/data/trained_models/ai_expo_2022/shoes/performance_experiments/',
  "summary_save_path": '/data/trained_models/ai_expo_2022/shoes/performance_experiments/',
  "log_to_file": True,
  "num_classes": 4,
  "simple_optimisations": False,
  # "early_stop": False
  # "bw": True
}

################################################################################
# Optimisations                                                                #
################################################################################
# https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
if config["simple_optimisations"]:
  torch.backends.cudnn.benchmark = True # Initial training steps will be slower
  torch.autograd.set_detect_anomaly(False)
  torch.autograd.profiler.profile(False)
  torch.autograd.profiler.emit_nvtx(False)
################################################################################

################################################################################
# Main Thread                                                                  #
################################################################################

# TODO: Add logging
if __name__ == '__main__':
  config["base_path"] = custom_utils.compute_base_save_path(config)
  log('Custom train shoes classifier ...', config)

  custom_utils.save_json(f'{config["base_path"]}/config.json', config)

  final_train_data_path = f'{config["save_path"]}/final_train_data.csv'
  if not custom_utils.check_if_file_exists(final_train_data_path):
    custom_utils.save_csv(final_train_data_path, 'save_identifier,train_loss,train_accuracy,memory_usage,total_time,memory_used_by_model')

  final_test_data_path = f'{config["save_path"]}/final_test_data.csv'
  if not custom_utils.check_if_file_exists(final_test_data_path):
    custom_utils.save_csv(final_test_data_path, 'save_identifier,test_loss,test_accuracy,memory_usage,total_time,memory_used_by_model')

  start_time = time.time()

  if config['summary_device_name'] == 'cuda':
    custom_utils.clear_gpu()
    config["cuda"] = True
  else:
    config["cuda"] = False

  config['save_identifier'] = f'{config["save_identifier"]}-{config["selected_model"]}'

  dev, summary_dev = custom_utils.fetch_device(config)

  if config['summary_device_name'] == 'cuda':
    cuda_memory_before = torch.cuda.memory_allocated(dev)

  # Load Model
  # TODO: RNG Weights
  model = custom_utils.fetch(config)

  model_stats = summary(model, input_size=(config["stats_batch_size"], config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=summary_dev, verbose=0)
  log(model_stats, config)

  # Estimate total memory usage
  estimated_total_size_of_model = float(f'{model_stats}'.split("\n")[-2].split(" ")[-1])

  if config["train_num_workers"] > 0:
    estimated_total_memory_usage = estimated_total_size_of_model * config["train_num_workers"]
  else:
    estimated_total_memory_usage = estimated_total_size_of_model

  log(f"Estimated total memory usage: {estimated_total_memory_usage} MB", config)
  memory_usage = estimated_total_memory_usage

  # Get Data
  log("Get Data ...", config)
  outer_batch_size = config["train_batch_size"]
  train_dataset, train_dataloader = custom_utils.load_dataset(config, 'train', batch_size=outer_batch_size)
  # val_dataset, val_dataloader = custom_utils.load_dataset(config, 'val', batch_size=outer_batch_size)
  test_dataset, test_dataloader = custom_utils.load_dataset(config, 'test', batch_size=outer_batch_size)

  classes = train_dataset.classes

  log(f"train dataset count: {len(train_dataset)}", config)
  # log(f"val dataset count: {len(val_dataset)}", config)
  log(f"test dataset count: {len(test_dataset)}", config)
  log(f"{classes}", config)

  # Check Image Dimensions
  # log(f'Image Dim: {np.array(train_dataset[0][0]).shape}', config)
  log(f'Image Dim: {train_dataset[0][0].shape}', config)

  opt = custom_utils.select_optimizer(config, model)
  scaler = torch.cuda.amp.GradScaler(enabled=config["mixed_precision"])

  # loss_fn = torch.nn.NLLLoss() # Multi-Class
  loss_fn = torch.nn.CrossEntropyLoss() # binary classifier
  # loss_fn = torch.nn.MSELoss() # Regression

  # Train Model
  log("Train and test model (Simple Loop) ...", config)
  model, train_loss, train_accuracy, test_loss, test_accuracy = custom_utils.simple_loop(config, model, dev, loss_fn, opt, scaler, train_dataloader, test_dataloader)

  if config['summary_device_name'] == 'cuda':
    cuda_memory_after = torch.cuda.memory_allocated(dev)

  # Sanity Check
  log("Sanity Check ...", config)
  model.eval()

  for idx in range(len(test_dataset)):
    if test_dataset[idx][1] == 1: # Nike
      nike_input_image = torch.stack([test_dataset[idx][0]]).to(dev)
    elif test_dataset[idx][1] == 2: # Adidas
      adidas_input_image = torch.stack([test_dataset[idx][0]]).to(dev)
    elif test_dataset[idx][1] == 3: # Converse
      converse_input_image = torch.stack([test_dataset[idx][0]]).to(dev)

  # Nike
  prediction = model(nike_input_image).argmax(1)
  log(f'Prediction for Nike example: {prediction.detach()} (should be: 1)', config)

  # Adidas
  prediction = model(adidas_input_image).argmax(1)
  log(f'Prediction for Adidas example: {prediction.detach()} (should be: 2)', config)

  # Converse
  prediction = model(converse_input_image).argmax(1)
  log(f'Prediction for Converse example: {prediction.detach()} (should be: 3)', config)

  # Save Final Model
  if config['summary_device_name'] == 'cuda':
    memory_used_by_model = cuda_memory_after - cuda_memory_before
  else:
    memory_used_by_model = -1.0

  custom_utils.save(model, opt, scaler, None, config["epochs"], config, config["base_path"])

  log('===================================================================\n\n', config)
  total_time = time.time() - start_time
  log(f'Total time: {total_time} secs.', config)

  # Save final results
  custom_utils.save_csv(final_train_data_path, f'{config["save_identifier"]},{train_loss.detach()},{train_accuracy},{memory_usage},{total_time},{memory_used_by_model}')
  custom_utils.save_csv(final_test_data_path, f'{config["save_identifier"]},{test_loss.detach()},{test_accuracy},{memory_usage},{total_time},{memory_used_by_model}')

