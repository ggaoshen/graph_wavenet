import os
import ssl
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from model import Model

from util import masked_mse, temporal_dataset_split
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

ssl._create_default_https_context = ssl._create_unverified_context

# Get dataset
dataset = ChickenpoxDatasetLoader().get_dataset(
    lags=8
)  # 8 lags to be consistent with dataset paper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = "store/checkpoint"
if not os.path.exists(save_path):
    os.makedirs(save_path)


# Get data
train_ratio, val_ratio = 0.8, 0.1
train_dataset, val_dataset, test_dataset = temporal_dataset_split(
    dataset, train_split=train_ratio, validation_split=val_ratio
)
val_offset = int(dataset.snapshot_count * train_ratio)  # starting index for valid set
test_offset = val_offset + int(
    dataset.snapshot_count * val_ratio
)  # starting index for test set


# Post-processing for n-period forecast label
def prepare_n_period_y(dataset):
    res = []
    for data in dataset:
        res.append(data.y)
    res = torch.stack(res, dim=0)

    return res


y_all = prepare_n_period_y(dataset)


# Model inputs
in_dim = dataset[0].num_node_features  # 8 treat lagged inputs as node features
out_dim = 1
num_nodes = dataset[0].num_nodes  # 20
timesteps_to_predict = 10  # 10, 20, 40 week forecast horizon
epochs = 50
lrate = 0.0001
wdecay = 0.001

# early stopping parameters
patience = 10
counter = 0
best_val_loss = float("inf")


# Model definition
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


# Training loop
print("start training...", flush=True)
his_loss = []
val_time = []
train_time = []
best_epoch = 0

training_curve_dict = {"epoch_train_loss": [], "epoch_valid_loss": []}

for epoch in tqdm(range(epochs)):
    # train
    train_loss = []
    t1 = time.time()

    for i, snapshot in enumerate(train_dataset):
        x_train = snapshot.x.reshape(-1, num_nodes, in_dim).to(device)
        if timesteps_to_predict == 1:
            y_train = snapshot.y.to(device)
        else:
            y_train = y_all[i : i + timesteps_to_predict, :].to(device)

        loss = model.train(x_train, y_train)

        train_loss.append(loss)

    mtrain_loss = np.mean(train_loss)
    training_curve_dict["epoch_train_loss"].append(mtrain_loss)
    print(f"training loss: {mtrain_loss}")

    t2 = time.time()
    train_time.append(t2 - t1)

    # validation
    valid_loss = []

    s1 = time.time()
    for i, snapshot in enumerate(val_dataset):
        x_val = snapshot.x.reshape(-1, num_nodes, in_dim).to(device)
        if timesteps_to_predict == 1:
            y_val = snapshot.y.to(device)
        else:
            y_val = y_all[val_offset + i : val_offset + i + timesteps_to_predict, :].to(
                device
            )

        with torch.no_grad():
            loss = model.eval(x_val, y_val)
        valid_loss.append(loss)

    s2 = time.time()
    val_time.append(s2 - s1)

    mvalid_loss = np.mean(valid_loss)
    training_curve_dict["epoch_valid_loss"].append(mvalid_loss)

    his_loss.append(mvalid_loss)

    if np.argmin(his_loss) == len(his_loss) - 1:
        torch.save(model.gwnet.state_dict(), save_path + "/epoch_" + str(i) + ".pth")
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
model.gwnet.load_state_dict(torch.load(save_path + "/epoch_" + str(best_epoch) + ".pth"))
model.eval()
loss = 0
for i, snapshot in enumerate(test_dataset[:-timesteps_to_predict]):
    x_test = snapshot.x.reshape(-1, num_nodes, in_dim).to(device)

    if timesteps_to_predict == 1:
        y_test = snapshot.y.to(device)
    else:
        y_test = y_all[test_offset + i : test_offset + i + timesteps_to_predict, :].to(
            device
        )  # get y label over forecast horizon

    with torch.no_grad():
        loss += model.eval(x_test, y_test)

loss = loss / (i + 1)
loss = loss
print("Test MSE Loss: {:.4f}".format(loss))


# store training loss
df = pd.DataFrame(training_curve_dict)
df.to_csv("training_curve.csv")


# plot training and validation loss
df.plot()
