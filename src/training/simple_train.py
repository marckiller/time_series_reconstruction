import torch
from torch import nn
from typing import Callable
from tqdm import tqdm

def train_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: Callable,
    optimizer_class: Callable = torch.optim.Adam,
    optimizer_kwargs: dict = {"lr": 1e-3},
    epochs: int = 10,
    device: str = 'cpu'
):
    model = model.to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d} [train]"):
            X_seq = batch['X_seq'].to(device)
            X_static = batch['X_static'].to(device)
            y_seq = batch['y_seq'].to(device)

            optimizer.zero_grad()
            y_pred = model(X_seq, X_static)
            loss = loss_fn(y_pred, y_seq)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X_seq = batch['X_seq'].to(device)
                X_static = batch['X_static'].to(device)
                y_seq = batch['y_seq'].to(device)

                y_pred = model(X_seq, X_static)
                loss = loss_fn(y_pred, y_seq)
                val_losses.append(loss.item())

        avg_train = sum(train_losses) / len(train_losses)
        avg_val = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")