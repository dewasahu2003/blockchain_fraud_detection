import torch
from torch_geometric.data import Data


def accuracy(y_pred: torch.Tensor, y_real: torch.Tensor, y_thresold=0.5):

    y_pred_label = (torch.sigmoid(y_pred) > y_real).float()
    correct_sum = (y_pred_label > y_thresold).sum().float()

    acc = correct_sum / y_real.shape[0]
    return acc


def train(model: torch.nn.Module, epochs, data: Data, loss_fun, optim):

    model.train()
    for epoch in range(epochs):

        output = model(data.x, data.edge_index)
        loss = loss_fun(output[data.train_idx], data.y[data.train_idx].unsqueeze(1))
        acc = accuracy(output[data.train_idx], data.y[data.train_idx].unsqueeze(1))

        loss.backward()

        optim.step()
        optim.zero_grad()

        val_loss = loss_fun(output[data.val_idx], data.y[data.val_idx].unsqueeze(1))
        val_acc = accuracy(output[data.val_idx], data.y[data.val_idx])

        print(
            f"epoch:{epoch} || loss:{loss} || acc:{acc} || val_loss:{val_loss} || val_acc:{val_acc}"
        )
    return model


def test(model: torch.nn.Module, data: Data):
    model.eval()
    output = model(data.x, data.edge_index)
    pred = (torch.sigmoid(output) > 0.5).unsqueeze(1)
    return pred
