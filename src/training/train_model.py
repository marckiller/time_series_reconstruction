

import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.utils.dataset import build_dataset, MaskedTimeSeriesDataset
from src.models.conv_reconstruction_model import ConvReconstructionModel
from src.utils.loss.masked_loss_functions import composite_loss

def train_model(df, config):
    X_index, X_ts, y = build_dataset(df, ['X_index', 'X_ts'], ['y'], {
        'X_index': ['index_ts_low_high_norm'],
        'X_ts': ['ts_low_high_norm'],
        'y': ['ts_low_high_norm']
    })

    (X_static,) = build_dataset(df, inputs=['X_static'], outputs=[], columns_map={
        'X_static': [
            'corr_30', 'corr_60',
            'open_pos', 'close_pos', 'body_to_range', 'direction',
            'index_open_pos', 'index_close_pos', 'index_body_to_range', 'index_direction'
        ]
    }, expand_sequence_columns=False)

    train_idx, val_idx = train_test_split(list(range(len(X_index))), test_size=0.2, random_state=42)

    def split_tensor(tensor, indices):
        return tensor[indices]

    mask_cfg = config["data"].get("mask_config", {})

    dataset_train = MaskedTimeSeriesDataset(
        split_tensor(X_index, train_idx),
        split_tensor(X_ts, train_idx),
        split_tensor(X_static, train_idx),
        split_tensor(y, train_idx),
        mask_config=mask_cfg
    )
    dataset_val = MaskedTimeSeriesDataset(
        split_tensor(X_index, val_idx),
        split_tensor(X_ts, val_idx),
        split_tensor(X_static, val_idx),
        split_tensor(y, val_idx),
        mask_config=mask_cfg
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=16)

    model = ConvReconstructionModel(
        seq_len=60,
        static_dim=dataset_train[0]['X_static'].shape[-1]
    ).to(device)

    if config["training"].get("mode") == "finetune":
        path = config["training"].get("pretrained_model_path")
        if path and os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded pretrained model from {path}")
        else:
            raise ValueError("Pretrained model path is missing or invalid.")

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    scheduler_type = config["training"].get("scheduler", "none")

    if scheduler_type == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, threshold=1e-4, verbose=True
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    model_path = config["training"]["model_save_path"]
    patience = config["training"]["early_stopping_patience"]
    loss_weights = config["loss"]["weights"]

    best_val_loss = float('inf')
    wait = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_vals = torch.zeros(8).to(device)
        for batch in train_loader:
            xb_idx = batch['X_index'].to(device)
            xb_idx_m = batch['X_index_mask'].to(device)
            xb_ts = batch['X_ts'].to(device)
            xb_ts_m = batch['X_ts_mask'].to(device)
            xb_static = batch['X_static'].to(device)
            xb_static_m = batch['X_static_mask'].to(device)
            yb = batch['y'].to(device)
            yb_mask = batch['y_mask'].to(device)

            optimizer.zero_grad()
            y_pred = model(xb_idx, xb_idx_m, xb_ts, xb_ts_m, xb_static, xb_static_m)
            eff_mask = (yb_mask == 1) & (xb_ts_m == 0)

            loss, loss_dict = composite_loss(y_pred, yb, eff_mask.float(), xb_ts_m, loss_weights)
            loss.backward()
            optimizer.step()

            components = [loss_dict[k] for k in ['mse', 'min_val', 'max_val', 'min_pos', 'max_pos', 'roughness', 'pull'] if k in loss_dict]
            train_vals += torch.tensor([loss.item(), *[c.item() for c in components]], device=device)

        model.eval()
        val_vals = torch.zeros(8).to(device)
        with torch.no_grad():
            for batch in val_loader:
                xb_idx = batch['X_index'].to(device)
                xb_idx_m = batch['X_index_mask'].to(device)
                xb_ts = batch['X_ts'].to(device)
                xb_ts_m = batch['X_ts_mask'].to(device)
                xb_static = batch['X_static'].to(device)
                xb_static_m = batch['X_static_mask'].to(device)
                yb = batch['y'].to(device)
                yb_mask = batch['y_mask'].to(device)

                y_pred = model(xb_idx, xb_idx_m, xb_ts, xb_ts_m, xb_static, xb_static_m)
                eff_mask = (yb_mask == 1) & (xb_ts_m == 0)
                loss, loss_dict = composite_loss(y_pred, yb, eff_mask.float(), xb_ts_m, loss_weights)
                components = [loss_dict[k] for k in ['mse', 'min_val', 'max_val', 'min_pos', 'max_pos', 'roughness', 'pull'] if k in loss_dict]
                val_vals += torch.tensor([loss.item(), *[c.item() for c in components]], device=device)

        train_vals /= len(train_loader)
        val_vals /= len(val_loader)

        print(f"Epoch {epoch+1}")
        print(f"  Train -> Total: {train_vals[0]:.4f}, MSE: {train_vals[1]:.4f}, MinV: {train_vals[2]:.4f}, MaxV: {train_vals[3]:.4f}, MinP: {train_vals[4]:.4f}, MaxP: {train_vals[5]:.4f}, Rough: {train_vals[6]:.4f}, Pull: {train_vals[7]:.4f}")
        print(f"  Val   -> Total: {val_vals[0]:.4f}, MSE: {val_vals[1]:.4f}, MinV: {val_vals[2]:.4f}, MaxV: {val_vals[3]:.4f}, MinP: {val_vals[4]:.4f}, MaxP: {val_vals[5]:.4f}, Rough: {val_vals[6]:.4f}, Pull: {val_vals[7]:.4f}")

        if scheduler:
            scheduler.step(val_vals[0].item())

        if epoch >= 10 and val_vals[0] < best_val_loss - 1e-4:
            best_val_loss = val_vals[0]
            wait = 0
            torch.save(model.state_dict(), model_path)
        else:
            if epoch >= 10:
                wait += 1
                if wait >= patience:
                    print("Early stopping.")
                    break