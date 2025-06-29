import matplotlib.pyplot as plt
import torch
import random

def plot_prediction_example(model, val_loader, device='cpu', example_idx=None):
    model.eval()
    model = model.to(device)

    batch = next(iter(val_loader))
    X_seq = batch['X_seq'].to(device)
    X_static = batch['X_static'].to(device)
    y_true = batch['y_seq'].to(device)

    if example_idx is None:
        example_idx = random.randint(0, X_seq.size(0) - 1)

    with torch.no_grad():
        y_pred = model(X_seq, X_static)

    y_true_np = y_true[example_idx].cpu().numpy()
    y_pred_np = y_pred[example_idx].cpu().numpy()
    x_seq_np = X_seq[example_idx].cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(x_seq_np, label='Input (X_seq)', linestyle='--', color='gray')
    plt.plot(y_true_np, label='True (y)', marker='o')
    plt.plot(y_pred_np, label='Predicted (y)', marker='x')
    plt.title(f"Example #{example_idx}")
    plt.xlabel("Timestep")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.show()