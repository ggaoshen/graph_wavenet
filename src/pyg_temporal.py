import ssl

from graphwavenet import GraphWaveNet
ssl._create_default_https_context = ssl._create_unverified_context



# # look for other datasets
from torch_geometric_temporal.dataset import METRLADatasetLoader
loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
iterator = iter(dataset)
print("METRLADatasetLoader", next(iterator))
print("METRLADatasetLoader", next(iterator))


from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
dataset = ChickenpoxDatasetLoader().get_dataset()
iterator = iter(dataset)
print("ChickenpoxDatasetLoader", next(iterator))
print("ChickenpoxDatasetLoader", next(iterator))

# from torch_geometric_temporal.dataset import PedalMeDatasetLoader
# dataset = PedalMeDatasetLoader().get_dataset()
# print("PedalMeDatasetLoader", next(iter(dataset)))


from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
dataset = WikiMathsDatasetLoader().get_dataset(lags=14)
iterator = iter(dataset)
print("WikiMathsDatasetLoader", next(iterator))
print("WikiMathsDatasetLoader", next(iterator))

# from torch_geometric_temporal.dataset import WindmillOutputSmallDatasetLoader
# dataset = WindmillOutputSmallDatasetLoader().get_dataset()
# print("WindmillOutputSmallDatasetLoader", next(iter(dataset)))

# from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
# dataset = EnglandCovidDatasetLoader().get_dataset()
# print("EnglandCovidDatasetLoader", next(iter(dataset)))


# from torch_geometric_temporal.dataset import MontevideoBusDatasetLoader
# dataset = MontevideoBusDatasetLoader().get_dataset()
# print("MontevideoBusDatasetLoader", next(iter(dataset)))


# from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader
# dataset = TwitterTennisDatasetLoader().get_dataset()
# print("TwitterTennisDatasetLoader", next(iter(dataset)))


from torch_geometric_temporal.signal import temporal_signal_split

# sample training loop
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)


import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

# training
from tqdm import tqdm

# model = RecurrentGCN(node_features=14, filters=32)
model = GraphWaveNet(
    num_nodes=1068,
    in_channels=14,
    out_channels=1,
    out_timesteps=1,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(50)):
    for time, snapshot in enumerate(train_dataset):
        x = snapshot.x.reshape(1, 1068, 14)
        y_hat = model(x, snapshot.edge_index, snapshot.edge_attr).squeeze()
        print(y_hat)
        cost = torch.mean((y_hat-snapshot.y)**2)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

# eval
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
