import torch
from model import GCN, GAT
from helper import train, test
from dataset import elliptical_data

model = GCN(165, 128, 1)
optim = torch.optim.Adam(model.parameters(), lr=0.2)
loss_fun = torch.nn.BCEWithLogitsLoss()
data = elliptical_data

train(model, 100, data, loss_fun, optim)

y_test_pred = test(model, data)

print(
    f"Test fraud Guess:{y_test_pred[data.val_idx].detach().sum().numpy()/len(data.y[data.val_idx])}"
)
