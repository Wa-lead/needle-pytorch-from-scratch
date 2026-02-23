import sys

sys.path.append("./python")
import needle as ndl
import needle.nn as nn
import numpy as np
import os


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU()
    )


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    loss_func = nn.SoftmaxLoss()

    hit, total = 0, 0
    loss_all = 0

    if opt is not None:
        model.train()
        for idx, (X, y) in enumerate(dataloader):
            opt.reset_grad()
            output = model(X)
            loss = loss_func(output, y)
            loss_all += loss.numpy().item()
            loss.backward()
            opt.step()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    else:
        model.eval()
        for idx, (X, y) in enumerate(dataloader):
            output = model(X)
            loss = loss_func(output, y)
            loss_all += loss.numpy().item()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]

    acc = (total - hit) / total
    return acc, loss_all / (idx + 1)


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    train_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
    )
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    )

    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_acc, train_loss = epoch(train_dataloader, model, opt)

    test_acc, test_loss = epoch(test_dataloader, model)
    return train_acc, train_loss, test_acc, test_loss
