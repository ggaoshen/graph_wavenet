import os
import ssl

from sklearn.discriminant_analysis import StandardScaler

from graphwavenet import GraphWaveNet
from model import Model
import util as util
from util import masked_mse

ssl._create_default_https_context = ssl._create_unverified_context


from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

dataset = ChickenpoxDatasetLoader().get_dataset()
iterator = iter(dataset)
print("ChickenpoxDatasetLoader", next(iterator))
print("ChickenpoxDatasetLoader", next(iterator))

# from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
# dataset = EnglandCovidDatasetLoader().get_dataset()
# print("EnglandCovidDatasetLoader", next(iter(dataset)))

# from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader
# dataset = TwitterTennisDatasetLoader().get_dataset()
# print("TwitterTennisDatasetLoader", next(iter(dataset)))


# Run Chickenpox Dataset

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# training
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch_geometric_temporal.signal import temporal_signal_split

train_ratio = 0.8
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)
offset = int(dataset.snapshot_count * train_ratio)  # starting index for test set

in_dim = dataset[0].num_node_features  # 8-period lagged inputs as node features
out_dim = 1
num_nodes = dataset[0].num_nodes  # 20
timesteps_to_predict = 10  # 10, 20, 40 week forecast horizon
epochs = 50
lrate = 0.0001
wdecay = 0.001
save_path = "store/checkpoint"

# Enable extensions
util.extensions_enabled = True

model = Model(
    num_nodes=num_nodes,
    in_dim=in_dim,
    out_dim=out_dim,
    out_horizon=timesteps_to_predict,
    lrate=lrate,
    wdecay=wdecay,
    device=device,
    edge_index=dataset.edge_index,
    edge_weight=dataset.edge_weight,
)

if not os.path.exists(save_path):
    os.makedirs(save_path)

print("start training...", flush=True)
his_loss = []
val_time = []
train_time = []
best_epoch = 0


def prepare_n_period_y(dataset):
    res = []
    for data in dataset:
        res.append(data.y)
    res = torch.stack(res, dim=0)

    return res


y_all = prepare_n_period_y(dataset)
training_curve_dict = {"epoch_train_loss": []}

for epoch in tqdm(range(epochs)):
    train_loss = []
    t1 = time.time()

    for i, snapshot in enumerate(train_dataset):
        x_train = snapshot.x.reshape(-1, num_nodes, in_dim).to(device)
        if timesteps_to_predict == 1:
            y_train = snapshot.y.to(device)
        else:
            y_train = y_all[i : i + timesteps_to_predict, :].to(
                device
            )  # get y label over forecast horizon

        loss = model.train(x_train, y_train)

        train_loss.append(loss)

    t2 = time.time()
    train_time.append(t2 - t1)
    mtrainloss = np.mean(train_loss)
    training_curve_dict["epoch_train_loss"].append(mtrainloss)
    print(f"training loss: {mtrainloss}")


print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))

# eval
loss = 0
for i, snapshot in enumerate(test_dataset):
    if i + timesteps_to_predict > test_dataset.snapshot_count:
        break

    x_test = snapshot.x.reshape(-1, num_nodes, in_dim).to(device)

    if timesteps_to_predict == 1:
        y_test = snapshot.y.to(device)
    else:
        y_test = y_all[offset + i : offset + i + timesteps_to_predict, :].to(device)

    with torch.no_grad():
        # pred = model(x_test, snapshot.edge_index, snapshot.edge_attr).squeeze()
        loss += model.eval(x_test, y_test)

loss = loss / (i + 1)
print("Test MSE Loss: {:.4f}".format(loss))

# store training and testing loss
pd.DataFrame(training_curve_dict).to_csv(
    "training_curve.csv"
)  # store training and testing loss
