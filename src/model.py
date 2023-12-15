import torch
import torch.optim as optim
import util as util
from graphwavenet import GraphWaveNet


class Model:
    def __init__(
        self,
        num_nodes,
        in_dim,
        out_dim,
        out_horizon,
        lrate,
        wdecay,
        device,
        edge_index=None,
        edge_weight=None,
    ):
        self.gwnet = GraphWaveNet(num_nodes, in_dim, out_dim, out_horizon)
        self.gwnet.to(device)
        self.optimizer = optim.Adam(
            self.gwnet.parameters(), lr=lrate, weight_decay=wdecay
        )

        # As part of extensions, we enable learning rate decay and gradient clipping
        self.scheduler = None
        self.clip = None
        if util.extensions_enabled:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.99
            )
            self.clip = 5

        self.loss = util.masked_mse

        self.edge_index = [[], []]
        self.edge_weight = []

        # use adjacency matrix in the form of edge_index and edge_weight
        # assuming static graph
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_weight is not None:
            self.edge_weight = edge_weight

        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=device)
        self.edge_weight = torch.tensor(
            self.edge_weight, dtype=torch.float, device=device
        )

    def train(self, input, real_val):
        self.gwnet.train()
        self.optimizer.zero_grad()
        predict = self.gwnet(input, self.edge_index, self.edge_weight).squeeze()

        real = torch.unsqueeze(real_val, dim=1)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.gwnet.parameters(), self.clip)
        self.optimizer.step()

        if util.extensions_enabled:
            self.scheduler.step()
        return loss.item()

    def eval(self, input, real_val):
        self.gwnet.eval()
        predict = self.gwnet(input, self.edge_index, self.edge_weight)
        real = torch.unsqueeze(real_val, dim=1)
        loss = self.loss(predict, real, 0.0)
        return loss.item()
