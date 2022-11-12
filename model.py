import torch
from torch_geometric.nn import GATv2Conv, GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gcn1 = GCNConv(input_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        # keep_going = torch.relu(F.dropout(x, p=0.6, training=self.training))
        keep_going = torch.relu(self.gcn1(x, edge_index))
        keep_going =torch.relu( F.dropout(keep_going, p=0.6, training=self.training))
        keep_going = self.gcn2(keep_going, edge_index)
        return keep_going


class GAT(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.gat1 = GATv2Conv(input_size, hidden_size)
        self.gat2 = GATv2Conv(hidden_size, output_size)

    def forward(self, x, edge_index):
        keep_going = torch.relu(F.dropout(x, p=0.6, training=self.training))
        keep_going = self.gat1(x, edge_index)
        keep_going = F.dropout(keep_going, p=0.6, training=self.training)
        keep_going = self.gat2(keep_going, edge_index)
        return keep_going
