import torch
import numpy as np
import pandas as pd

def mlp(preprocessed_data):
    """Train a MLP for multi-target regression."""

    # mask is used to mark which data points we are supposed to predict on
    Xtr, Ytr, Xval, Yval = preprocessed_data
    Ytr_mask = pd.notnull(Ytr).astype(int)
    Yval_mask = pd.notnull(Yval).astype(int)
    Yval_original = Yval

    # Normalize
    Xmeans, Xstds = Xtr.mean(), Xtr.std()
    Ymeans, Ystds = Ytr.mean(), Ytr.std()
    Xtr = (Xtr - Xmeans)/Xstds
    Ytr = (Ytr - Ymeans)/Ystds
    Xtr = Xtr.fillna(0)
    Ytr = Ytr.fillna(0)

    Xval = (Xval - Xmeans)/Xstds
    Yval = (Yval - Ymeans)/Ystds
    Xval = Xval.fillna(0)
    Yval = Yval.fillna(0)

    # Taken from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

    N, D_in = Xtr.values.shape
    _, D_out = Ytr.values.shape

    H = 10

    # Create random Tensors to hold inputs and outputs
    x = torch.FloatTensor(Xtr.values)
    y = torch.FloatTensor(Ytr.values)
    ymask = torch.ByteTensor(Ytr_mask.values)

    xval = torch.FloatTensor(Xval.values)
    yval = torch.FloatTensor(Yval.values)
    ymask_val = torch.ByteTensor(Yval_mask.values)

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    loss_fn = torch.nn.MSELoss()

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Mask y and ypred such that loss is zero for missing targets
        y_pred_for_loss = torch.where(ymask, y_pred, torch.zeros_like(y_pred))
        y_for_loss = torch.where(ymask, y, torch.zeros_like(y))

        # Compute and print loss.
        loss = loss_fn(y_pred_for_loss, y_for_loss)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        y_pred = model(xval)
        y_pred_for_loss = torch.where(ymask_val, y_pred, torch.zeros_like(y_pred))
        y_for_loss = torch.where(ymask_val, yval, torch.zeros_like(yval))
        lossval = loss_fn(y_pred_for_loss, y_for_loss)
        print('{epoch:03d}   train loss:{train:0.5f}  val loss:{val:0.5f}'.format(
            epoch=t, train=loss.item(), val=lossval.item()))


    # Eval
    y_pred = y_pred.detach().numpy()
    Ypred = y_pred
    Yval = Yval_original
    Ypred = {col: Ypred[:, i] for i, col in enumerate(Yval.columns)}
    Ypred = pd.DataFrame(Ypred, index=Yval.index)
    Ypred = Ypred*Ystds + Ymeans
    Ypred = Ypred.fillna(0)

    se = np.power((Ypred - Yval).values, 2)
    rmse = np.sqrt(np.nanmean(se))
    print(rmse)
    assert False

    return y_pred

