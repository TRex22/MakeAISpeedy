import gc
import sys
import time
import tqdm
import csv

import numpy as np
import pandas as pd

import torch
from torch import nn

import seaborn as sns
import matplotlib.pyplot as plt

print('Generate Graphs...')
base_savepath = "/data/trained_models/ai_expo_2022/shoes/"

sns.set_theme(style="whitegrid")

f, ax = plt.subplots(figsize=(20, 10))
# sns.set(rc={"figure.figsize":(8, 4)}) #width=8, height=4
sns.set(rc={"figure.figsize":(20, 10)}) #width=8, height=4

# sns.set(rc={'axes.facecolor':'lightblue', 'figure.facecolor':'lightblue'})
# sns.set(rc={'axes.facecolor':'darkblue', 'figure.facecolor':'darkblue'})

raw_train_data = "save_identifier,train_loss,train_accuracy,memory_usage,total_time\n\
Baseline-cuda-resnet18,0.33460697531700134,0.8736363649368286,4707.5,228.13301062583923\n\
cuda-resnet34,0.194105863571167,0.9261363744735718,7064.63,331.89176988601685\n\
cpu-resnet50,0.10863842070102692,0.96875,20709.95,10230.312728643417\n\
cpu-resnet18,0.13631969690322876,0.9637500643730164,4707.5,2955.2129168510437\n\
cpu-resnet34,0.33543860912323,0.8609089851379395,7064.63,5211.019391298294\n\
cpu-resnet101,0.3859284222126007,0.8434090614318848,30186.24,16397.04039120674\n\
mini_model-simple_cnn,0.7280412912368774,0.6757954359054565,4511.32,223.59772372245789"

# train_dict = pd.read_csv(f'{base_savepath}/models_experiments/final_train_data.csv')

raw_test_data = "save_identifier,test_loss,test_accuracy,memory_usage,total_time\n\
Baseline-cuda-resnet18,1.6680350303649902,0.6521428823471069,4707.5,228.13301062583923\n\
cuda-resnet34,1.8691956996917725,0.40857142210006714,7064.63,331.89176988601685\n\
cpu-resnet50,1.7658429145812988,0.47357141971588135,20709.95,10230.312728643417\n\
cpu-resnet18,1.061974048614502,0.6871428489685059,4707.5,2955.2129168510437\n\
cpu-resnet34,2.0758094787597656,0.34214285016059875,7064.63,5211.019391298294\n\
cpu-resnet101,1.0979480743408203,0.5957143306732178,30186.24,16397.04039120674\n\
mini_model-simple_cnn,1.2427204847335815,0.5357142686843872,4511.32,223.59772372245789"

# test_dict = pd.read_csv(f'{base_savepath}/models_experiments/final_test_data.csv')
result_dict = pd.read_csv('/data/trained_models/ai_expo_2022/shoes/models_experiments_final_data.csv')

# Plot the total
# sns.set_color_codes("pastel")
sns.set_color_codes("dark")
# breakpoint()

# print('Generate Experiment 1 Times...')
# ax = sns.barplot(x="Model Name", y="Total Time (Seconds)", data=result_dict, color="b", palette = "flare")
# ax.set(title='Amount of time each model took to train', xlabel='Models')
# ax.bar_label(ax.containers[0])
# sns.barplot(x="save_identifier", y="total_time", data=result_dict, label="Total", color="b")


# print('Generate Experiment 1 Memory...')
# # ax = sns.relplot(x="Model Name", y="Memory Usage (MB)", data=result_dict, color="b", palette = "flare", hue="Model Name", size="Memory Usage (MB)", sizes=(40, 400), alpha=.5)
# # ax.set(title='Amount of Memory Used Per Model', xlabel='Models')
# # ax.set(xlabel=None)
# # ax.bar_label(ax.containers[0])

# ax = sns.barplot(x="Memory Usage (MB)", y="Model Name", data=result_dict, color="b", palette="flare")
# ax.set(title='Amount of Memory Used Per Model', xlabel='Memory Usage (MB)')
# ax.bar_label(ax.containers[0])
# sns.despine(left=True, bottom=True)

# plt.savefig(f'{base_savepath}/model_experiments_memory.png')


print('Generate Experiment 1 Train...')
result_dict = pd.read_csv('/data/trained_models/ai_expo_2022/shoes/models_experiments_final_data_train.csv')
# ax = sns.relplot(x="Model Name", y="Memory Usage (MB)", data=result_dict, color="b", palette = "flare", hue="Model Name", size="Memory Usage (MB)", sizes=(40, 400), alpha=.5)
# ax.set(title='Amount of Memory Used Per Model', xlabel='Models')
# ax.set(xlabel=None)
# ax.bar_label(ax.containers[0])

# ax = sns.barplot(x="Model Name", y="Test Loss", data=result_dict, color="b", palette="flare")

# palette="dark"
g = sns.catplot(
    data=result_dict, kind="bar",
    x="Model Name", y="accuracy", hue="Hue",
     palette="flare", alpha=.5, height=20
)

ax = g.facet_axis(0, 0)
g.set(title='Amount of Memory Used Per Model', xlabel='Models')
# ax.bar_label(ax.containers[0])
# sns.despine(left=True, bottom=True)

# https://stackoverflow.com/questions/55586912/seaborn-catplot-set-values-over-the-bars
for c in ax.containers:
  labels = [f'{v.get_height()}' for v in c]
  ax.bar_label(c, labels=labels, label_type='edge')

plt.savefig(f'{base_savepath}/model_experiments_accuracy.png')

