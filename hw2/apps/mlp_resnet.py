import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    main_block = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    return nn.Sequential(
        nn.Residual(main_block),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    layers = [nn.Flatten(), nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        layers.append(ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, 
                                    norm=norm, drop_prob=drop_prob))
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
        
    loss_func = nn.SoftmaxLoss()
    
    total, loss, err_num = 0, 0, 0
    for X, y in dataloader:
        logits = model(X)
        l = loss_func(logits, y)
        
        if opt is not None:
            opt.reset_grad()
            l.backward()
            opt.step()
        
        batch_size = y.shape[0]
        total += batch_size
        loss += float(l.numpy() * batch_size)
        
        y_pred = np.argmax(logits.numpy(), axis=1)
        err_num += (y_pred != y.numpy()).sum()
        
    avg_loss = float(loss / total)
    avg_err = float(err_num / total)
    return avg_err, avg_loss
    ### END YOUR SOLUTION


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
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        image_filename = os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        label_filename = os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_dataset = ndl.data.MNISTDataset(
        image_filename = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        label_filename = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size, True)
    
    import functools
    input_dim = functools.reduce(lambda a, b: a*b, train_dataset[0][0].shape)
    model = MLPResNet(input_dim, hidden_dim)
    
    opt = optimizer(model.parameters(), lr, weight_decay = weight_decay)
    
    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        
    # statictis calculated during training is different from plain evaluation.
    # train_err, train_loss = epoch(train_dataloader, model)
    
    test_err, test_loss = epoch(test_dataloader, model)
    return train_err, train_loss, test_err, test_loss # somehow here should return err_rate instead of accuracy
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
