import os
import ssl
from graphwavenet import GraphWaveNet

from util import masked_mse, temporal_dataset_split

ssl._create_default_https_context = ssl._create_unverified_context



# Temporal Datasets

from torch_geometric_temporal.dataset import METRLADatasetLoader
loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
iterator = iter(dataset)
print("METRLA dataset from original Graph Wavenet paper: \n", next(iterator))


from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
dataset = ChickenpoxDatasetLoader().get_dataset(lags=8) # consistent with chickenpox paper
iterator = iter(dataset)
print("Chickenpox dataset: \n", next(iterator))


# Run Chickenpox Dataset

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# training
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = "store/checkpoint"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Get data

# from torch_geometric_temporal.signal import temporal_signal_split
train_ratio, val_ratio = 0.8, 0.1
# train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)
train_dataset, val_dataset, test_dataset = temporal_dataset_split(dataset, train_split = train_ratio, validation_split = val_ratio)
val_offset = int(dataset.snapshot_count * train_ratio) # starting index for valid set
test_offset = val_offset+int(dataset.snapshot_count * val_ratio) # starting index for valid set

def prepare_n_period_y(dataset):

    res = []
    for data in dataset:
        res.append(data.y)
    res = torch.stack(res, dim=0)

    return res

y_all = prepare_n_period_y(dataset)


# Model inputs 
in_dim = dataset[0].num_node_features # 8 treat lagged inputs as node features
out_dim = 1
num_nodes = dataset[0].num_nodes # 1068
timesteps_to_predict = 10 # 10, 20, 40 week forecast horizon
epochs = 200
lrate = 0.0001
wdecay = 0.001

# early stopping parameters
patience = 10
counter = 0
best_val_loss = float('inf') 


model = GraphWaveNet(
    num_nodes=num_nodes,
    in_channels=in_dim,
    out_channels=out_dim,
    out_timesteps=timesteps_to_predict,
).to(device)

# Training loop
print("start training...", flush=True)
his_loss = []
val_time = []
train_time = []
best_epoch = 0

optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=wdecay)

model.train()

training_curve_dict = {"epoch_train_loss": [], "epoch_valid_loss": []}

for epoch in tqdm(range(epochs)):
    train_loss = []
    t1 = time.time()

    for i, snapshot in enumerate(train_dataset):
        x_train = snapshot.x.reshape(-1, num_nodes, in_dim).to(device)
        if timesteps_to_predict == 1:
            y_train = snapshot.y.to(device)
        else:
            y_train = y_all[i : i + timesteps_to_predict,:].to(device)

        pred = model(x_train, snapshot.edge_index, snapshot.edge_attr).squeeze()
        loss = masked_mse(pred, y_train, 0.0) # mean squared error for loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss.append(loss.item())

    mtrain_loss = np.mean(train_loss)
    training_curve_dict['epoch_train_loss'].append(mtrain_loss)
    print(f"training loss: {mtrain_loss}")


    t2 = time.time()
    train_time.append(t2 - t1)
    valid_loss = []

    s1 = time.time()
    for i, snapshot in enumerate(val_dataset):
        x_val = snapshot.x.reshape(-1, num_nodes, in_dim).to(device)
        if timesteps_to_predict == 1:
            y_val = snapshot.y.to(device)
        else:
            y_val = y_all[val_offset + i : val_offset + i + timesteps_to_predict,:].to(device)

        with torch.no_grad():
            pred = model(x_val, snapshot.edge_index, snapshot.edge_attr).squeeze()
        loss = masked_mse(pred, y_val, 0.0).cpu().numpy()
        valid_loss.append(loss)

    s2 = time.time()
    # log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
    # print(log.format(i, (s2 - s1)))
    val_time.append(s2 - s1)

    mvalid_loss = np.mean(valid_loss)
    training_curve_dict['epoch_valid_loss'].append(mvalid_loss)

    his_loss.append(mvalid_loss)

    if np.argmin(his_loss) == len(his_loss) - 1:
        torch.save(
            model.state_dict(), save_path + "/epoch_" + str(i) + ".pth"
        )
        best_epoch = i

    log = (
        "Epoch: {:03d}, Train Loss: {:.4f}, "
        + "Valid Loss: {:.4f}, "
        + "Training Time: {:.4f}/epoch"
    )
    print(
        log.format(
            i,
            mtrain_loss,
            mvalid_loss,
            (t2 - t1),
        ),
        flush=True,
    )
print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


# eval
model.load_state_dict(
    torch.load(save_path + "/epoch_" + str(best_epoch) + ".pth")
)
model.eval()
loss = 0
for i, snapshot in enumerate(test_dataset[:-timesteps_to_predict]):

    x_test = snapshot.x.reshape(-1, num_nodes, in_dim).to(device)
    
    if timesteps_to_predict == 1:
        y_test = snapshot.y.to(device)
    else:
        y_test = y_all[test_offset+i : test_offset+i + timesteps_to_predict,:].to(device)
        

    with torch.no_grad():
        pred = model(x_test, snapshot.edge_index, snapshot.edge_attr).squeeze()
    loss += masked_mse(pred, y_test, 0.0) # mean squared error as loss

loss = loss / (i+1)
loss = loss.item()
print("Test MSE Loss: {:.4f}".format(loss))


# store training loss 
df = pd.DataFrame(training_curve_dict)
df.to_csv("training_curve.csv")


# plot training and validation loss
df.plot()
