import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
import os
import argparse
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def visualize_weights(csv_path, output_img):
    import csv
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [(row[0], float(row[1])) for row in reader if row and row[0] != 'BIAS']
            
    bp_data = sorted([x for x in data if x[0].startswith('bp_')], key=lambda x: numerical_sort_key(x[0]))
    rp_data = sorted([x for x in data if x[0].startswith('rp_')], key=lambda x: numerical_sort_key(x[0]))
    other_data = [x for x in data if not (x[0].startswith('bp_') or x[0].startswith('rp_'))]
    
    all_data = bp_data + rp_data + other_data
    all_features = [x[0] for x in all_data]
    all_weights = [x[1] for x in all_data]
    bp_len = len(bp_data)
    rp_len = len(rp_data)
    
    fig = plt.figure(figsize=(14, 6))
    x = np.arange(len(all_features))
    
    plt.bar(x[:bp_len], all_weights[:bp_len], label='BP weights', color='blue', alpha=0.7)
    plt.bar(x[bp_len:bp_len+rp_len], all_weights[bp_len:bp_len+rp_len], label='RP weights', color='red', alpha=0.7)
    if other_data:
        plt.bar(x[bp_len+rp_len:], all_weights[bp_len+rp_len:], label='Other weights', color='green', alpha=0.7)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Linear Regression Weights for Features', fontsize=14)
    
    plt.xticks(x[::5], [all_features[i] for i in range(0, len(all_features), 5)], rotation=45)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {output_img}")

def evaluate_all(weights_file, out_dir, suffix=''):
    import csv
    print(f"Evaluating on all data (suffix: '{suffix}')...")
    with open(weights_file, 'r') as f:
        rows = list(csv.DictReader(f))
    bias = float(next(r['weight'] for r in rows if r['feature'] == 'BIAS'))
    weights = np.array([float(r['weight']) for r in rows if r['feature'] != 'BIAS'])

    file_path = 'bp_rp_lamost_normalized.h5'
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        bp_cols = sorted([k for k in keys if k.startswith('bp_')], key=numerical_sort_key)
        rp_cols = sorted([k for k in keys if k.startswith('rp_')], key=numerical_sort_key)
        feature_cols = bp_cols + rp_cols
        feature_cols.append('ebv')
        
        full_feh = f['feh'][:].astype(np.float64)
        valid_mask = np.isfinite(full_feh)
        feh_valid = full_feh[valid_mask]
        
        y_true = feh_valid
        
        print("Loading and normalizing all data for evaluation...")
        X_all = []
        for col_name in feature_cols:
             X_all.append(np.nan_to_num(f[col_name][:][valid_mask]))
        
        X_all = np.column_stack(X_all)
        
        X_to_norm = X_all[:, :-1]
        norms = np.linalg.norm(X_to_norm, axis=1, keepdims=True) + 1e-8
        X_all[:, :-1] = X_to_norm / norms
        
        y_pred = X_all.dot(weights) + bias
        
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Overall MSE on all data: {mse:.4f}")
    print(f"Overall MAE on all data: {mae:.4f}")
    print(f"Overall R2 Score on all data: {r2:.4f}\n")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    from matplotlib.colors import LogNorm
    h = ax.hist2d(y_true, y_pred, bins=100, cmap='viridis', norm=LogNorm())
    fig.colorbar(h[3], ax=ax, label='Counts')
    
    # Plot y=x line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y = x')
    
    ax.set_xlabel('True [Fe/H]')
    ax.set_ylabel('Predicted [Fe/H]')
    
    title_str = f'Overall Evaluation {suffix}\nMSE: {mse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}'
    plt.title(title_str, fontsize=11)
    plt.legend()
    
    plt.tight_layout()
    out_name = f'regression_scatter_all_data{suffix}.png'
    out_file = os.path.join(out_dir, out_name)
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved regression scatter plot to {out_file}.")

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

def main():
    parser = argparse.ArgumentParser(description="Train a linear regression model on stellar data.")
    p_add = parser.add_argument
    p_add('--device', type=str, default=None, help="Device to use ('cuda', 'cpu', 'mps')")
    p_add('--list-hardware', action='store_true', help="List available hardware and exit.")
    p_add('--no-tf32', action='store_true', help="Disable TF32 for Ampere+ GPUs.")
    p_add('--compile', action='store_true', help="Use torch.compile().")

    p_add('--run-name', type=str, default=None, help="Name of the run. Outputs will be saved to linear_reg_{run_name}/.")
    p_add('--seed', type=int, default=42, help="Random seed for reproducibility.")
    p_add('--data-split', type=str, default='random', choices=['random', 'low_temp'], help="Which dataset split to use. Defaults to 'random'.")
    p_add('--train-file', type=str, default=None, help="Path to train H5 file. Overrides --data-split.")
    p_add('--test-file', type=str, default=None, help="Path to test H5 file. Overrides --data-split.")
    p_add('--hidden-dim', type=int, default=2, help="Hidden dimension (currently unused).")
    p_add('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'exact'], help="Optimizer type: 'adam', 'sgd', or 'exact' (for OLS/Ridge exact solve).")
    p_add('--lr', type=float, default=1e-3, help="Initial learning rate.")
    p_add('--momentum', type=float, default=0.0, help="Momentum factor for SGD optimizer. Ignored if --optimizer=adam.")
    p_add('--weight-decay', type=float, default=0.0, help="Weight decay factor for L2 regularization.")
    p_add('--epochs', type=int, default=50, help="Number of training epochs.")
    p_add('--batch-size', type=int, default=30000, help="Batch size for training.")
    p_add('--lr-end-factor', type=float, default=1.0, help="Final learning rate multiplier (linear scheduler).")

    args = parser.parse_args()

    if args.run_name:
        out_dir = f"linear_reg_{args.run_name}"
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "."

    # --- Hardware Search / List ---
    if args.list_hardware:
        print("Searching for available hardware...")
        print(f"PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"\nCUDA Available: Yes ({count} devices)")
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                print(f"  [{i}] {props.name} (VRAM: {props.total_memory / 1024**3:.2f} GB, CC: {props.major}.{props.minor})")
        else:
            print("\nCUDA Available: No")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             print("\nMPS (Apple Silicon) Available: Yes")
        else:
             print("\nMPS (Apple Silicon) Available: No")

        print(f"\nCPU Threads: {torch.get_num_threads()}")
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_file = args.train_file
    test_file = args.test_file

    def find_data_split(split, root='.'):
        for dirpath, dirnames, _ in os.walk(root):
            for d in dirnames:
                if split.lower() in d.lower():
                    found_dir = os.path.join(dirpath, d)
                    t_f, te_f = None, None
                    try:
                        files = os.listdir(found_dir)
                    except OSError:
                        continue

                    for f in files:
                        if f.endswith('.h5'):
                            if 'train' in f.lower():
                                t_f = os.path.join(found_dir, f)
                            elif 'test' in f.lower():
                                te_f = os.path.join(found_dir, f)
                    
                    if t_f and te_f:
                        return t_f, te_f
        return None, None

    if train_file is None or test_file is None:
        train_file, test_file = find_data_split(args.data_split)

    if train_file is None or test_file is None or not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Error: Could not find training/test h5 files for split '{args.data_split}'.")
        return

    print("Loading data...")
    with h5py.File(train_file, 'r') as f_tr, h5py.File(test_file, 'r') as f_te:
        keys = list(f_tr.keys())
        bp_cols = sorted([k for k in keys if k.startswith('bp_')], key=numerical_sort_key)
        rp_cols = sorted([k for k in keys if k.startswith('rp_')], key=numerical_sort_key)
        feature_cols = bp_cols + rp_cols
        feature_cols.append('ebv')
        
        print("Extracting features (this may take a moment)...")
        train_data_list = []
        test_data_list = []
        for i, col_name in enumerate(feature_cols):
            if i % 20 == 0:
                print(f" Reading column {i}/{len(feature_cols)}...")
            train_data_list.append(f_tr[col_name][:])
            test_data_list.append(f_te[col_name][:])
            
        X_train = np.column_stack(train_data_list)
        X_test = np.column_stack(test_data_list)
        
        y_train = f_tr['feh'][:].astype(np.float32)
        y_test = f_te['feh'][:].astype(np.float32)

        print(f"Total Train: {len(y_train)}")
        print(f"Total Test:  {len(y_test)}")

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    y_train = np.nan_to_num(y_train)
    y_test = np.nan_to_num(y_test)
    
    print("Pre-normalizing data (L2)...")
    X_train_to_norm = X_train[:, :-1]
    X_test_to_norm = X_test[:, :-1]
    ebv_train = X_train[:, -1:]
    ebv_test = X_test[:, -1:]
    
    train_norms = np.linalg.norm(X_train_to_norm, axis=1, keepdims=True)
    X_train_normed = X_train_to_norm / (train_norms + 1e-8)
    X_train = np.hstack([X_train_normed, ebv_train])
    
    test_norms = np.linalg.norm(X_test_to_norm, axis=1, keepdims=True)
    X_test_normed = X_test_to_norm / (test_norms + 1e-8)
    X_test = np.hstack([X_test_normed, ebv_test])

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    if device.type == 'cuda':
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        if not args.no_tf32 and torch.cuda.get_device_capability(device)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
    print(f"Moving data to {device}...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1)

    model = LinearRegressionModel(input_dim=len(feature_cols)).to(device)
    
    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
    
    criterion = nn.MSELoss(reduction='none')

    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = None

    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    epochs = args.epochs

    from torch.optim.lr_scheduler import LinearLR
    if optimizer is not None:
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args.lr_end_factor,
            total_iters=epochs,
        )
    else:
        scheduler = None

    train_losses = []
    test_losses = []
    test_r2_scores = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        if args.optimizer == 'exact':
            with torch.no_grad():
                X_pad = torch.cat([X_train_t, torch.ones(X_train_t.size(0), 1, dtype=torch.float32, device=device)], dim=1)
                y_tr_view = y_train_t.view(-1, 1)
                
                Xy = X_pad.T @ y_tr_view
                XX = X_pad.T @ X_pad
                
                wd = args.weight_decay
                H_reg = wd * torch.eye(X_pad.size(1), device=device)
                H_reg[-1, -1] = 0.0 # No weight decay on bias
                
                H = XX + H_reg
                
                # small damping
                H += 1e-5 * torch.eye(H.size(0), device=device)
                
                try:
                    w_pad_new = torch.linalg.solve(H, Xy).view(-1)
                except Exception as e:
                    print(f"Warning: Exact solve failed {e}. Using pseudo-inverse.")
                    w_pad_new = (torch.linalg.pinv(H) @ Xy).view(-1)
                    
                model.fc.weight.data = w_pad_new[:-1].view(1, -1)
                model.fc.bias.data = w_pad_new[-1:]
        else:
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss_unreduced = criterion(out, batch_y)
                loss = loss_unreduced.mean()
                loss.backward()
                optimizer.step()

        if scheduler is not None:
            scheduler.step()

        model.eval()
        epoch_train_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                out = model(batch_x)
                loss_unreduced = criterion(out, batch_y)
                loss = loss_unreduced.mean()
                epoch_train_loss += loss.item() * batch_x.size(0)
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)

        epoch_test_loss = 0.0
        all_test_preds = []
        all_test_y = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                out = model(batch_x)
                loss_unreduced = criterion(out, batch_y)
                loss = loss_unreduced.mean()
                epoch_test_loss += loss.item() * batch_x.size(0)
                all_test_preds.append(out.cpu().numpy())
                all_test_y.append(batch_y.cpu().numpy())

        epoch_test_loss /= len(test_dataset)
        
        all_test_preds = np.vstack(all_test_preds)
        all_test_y = np.vstack(all_test_y)
        test_r2 = r2_score(all_test_y, all_test_preds)
        
        test_losses.append(epoch_test_loss)
        test_r2_scores.append(test_r2)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs}"
                f" | TrLoss (MSE) {epoch_train_loss:.4f}  TeLoss {epoch_test_loss:.4f}  Test R2 {test_r2:.4f}"
            )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    l1 = ax1.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue', linewidth=2)
    l2 = ax1.plot(range(1, epochs+1), test_losses, label='Test Loss', color='red', linewidth=2)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test R2 Score', fontsize=12, color='green')
    l3 = ax2.plot(range(1, epochs+1), test_r2_scores, label='Test R2', color='green', linewidth=2, linestyle=':')
    ax2.tick_params(axis='y', labelcolor='green')

    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=12, loc='center right')

    plt.title("Train/Test Loss & Test R2 Score (Linear Regression)", fontsize=14)
    
    fig.tight_layout()
    out_img = 'linear_reg.png'
    out_img = os.path.join(out_dir, out_img)
    fig.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Training complete. Loss curve saved to {out_img}")
    print(f"Final Test R2: {test_r2_scores[-1]:.4f}")

    print("Saving model weights...")
    original_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    weights = original_model.fc.weight.detach().cpu().numpy().flatten()
    bias = original_model.fc.bias.detach().cpu().item()
    
    out_csv = 'linear_reg_model_weights.csv'
    weights_img = 'linear_reg_weights_plot.png'
    
    out_csv = os.path.join(out_dir, out_csv)
    weights_img = os.path.join(out_dir, weights_img)

    with open(out_csv, 'w') as f:
        f.write("feature,weight\n")
        f.write(f"BIAS,{bias}\n")
        for name, w in zip(feature_cols, weights):
            f.write(f"{name},{w}\n")
    print(f"Latest weights saved to {out_csv} (Total features: {len(feature_cols)})")

    visualize_weights(out_csv, weights_img)
    evaluate_all(out_csv, out_dir, suffix='')

if __name__ == "__main__":
    main()
