from model import GAT
from dataset import elliptical_data
from helper import train, test
import torch

data = elliptical_data
model = GAT(165, 128, 1)
loss_fun = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)

train(model, 100, data, loss_fun, optim)

y_test_pred = test(model, data)

print(
    f"Fraud detection:{y_test_pred[data.val_idx].detach().sum().numpy()/len(data.y[data.val_idx])}"
)
