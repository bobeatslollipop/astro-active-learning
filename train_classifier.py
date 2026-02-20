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

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x is already L2-normalized during preprocessing
        return self.fc(x)

def main():
    parser = argparse.ArgumentParser(description="Train a classifier on stellar data.")
    p_add = parser.add_argument
    p_add('--device', type=str, default=None, help="Device to use ('cuda', 'cpu', 'mps')")
    p_add('--list-hardware', action='store_true', help="List available hardware and exit.")
    p_add('--no-tf32', action='store_true', help="Disable TF32 for Ampere+ GPUs.")
    p_add('--compile', action='store_true', help="Use torch.compile().")
    p_add('--use-ebv', action='store_true', help="Include 'ebv' column as a feature.")
    args = parser.parse_args()

    # --- Hardware Search / List ---
    if args.list_hardware:
        print("Searching for available hardware...")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"\nCUDA Available: Yes ({count} devices)")
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                print(f"  [{i}] {props.name} (VRAM: {props.total_memory / 1024**3:.2f} GB, CC: {props.major}.{props.minor})")
        else:
            print("\nCUDA Available: No")

        # Check MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             print("\nMPS (Apple Silicon) Available: Yes")
        else:
             print("\nMPS (Apple Silicon) Available: No")

        # CPU info
        print(f"\nCPU Threads: {torch.get_num_threads()}")
        return

    cfg = {'seed': 42, 'file_path': './bp_rp_lamost_normalized.h5', 'feh_threshold': -2.0,
           'train_frac': 0.8, 'mr_ratio': 2, 'hidden_dim': 2, 'optimizer': 'adam', 'lr': 1,
           'momentum': 0, 'weight_decay': 0, 'epochs': 500, 'batch_size': 30000, 'lr_end_factor': 1}

    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])

    # 2. Load data
    file_path = cfg['file_path']
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print("Loading data...")
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        bp_cols = sorted([k for k in keys if k.startswith('bp_')], key=numerical_sort_key)
        rp_cols = sorted([k for k in keys if k.startswith('rp_')], key=numerical_sort_key)
        feature_cols = bp_cols + rp_cols
        if args.use_ebv:
            feature_cols.append('ebv')
        
        full_feh = f['feh'][:].astype(np.float64)
        valid_mask = np.isfinite(full_feh)
        
        valid_idx = np.where(valid_mask)[0]
        valid_feh = full_feh[valid_mask]
        
        # Binary definition: MP < feh_threshold <= MR
        thr = cfg['feh_threshold']
        mp_idx = valid_idx[valid_feh < thr]
        mr_idx = valid_idx[valid_feh >= thr]

        np.random.shuffle(mp_idx)
        np.random.shuffle(mr_idx)

        N_mp = len(mp_idx)
        if N_mp == 0:
            print("No metal-poor data found!")
            return

        N_tr_mp = int(cfg['train_frac'] * N_mp)
        N_te_mp = N_mp - N_tr_mp

        ratio   = cfg['mr_ratio']
        N_tr_mr = ratio * N_tr_mp
        N_te_mr = ratio * N_te_mp
        
        print(f"Total metal-poor data count: {N_mp}")
        print(f"Train MP (0): {N_tr_mp}, Train MR (1): {N_tr_mr}  [Total Train: {N_tr_mp + N_tr_mr}]")
        print(f"Test MP (0): {N_te_mp}, Test MR (1): {N_te_mr}   [Total Test:  {N_te_mp + N_te_mr}]")
        
        if len(mr_idx) < N_tr_mr + N_te_mr:
            print(f"Warning: Not enough metal-rich data! Have {len(mr_idx)}, need {N_tr_mr + N_te_mr}.")
            print("Adjusting test metal-rich limit to match available sizes.")
            N_te_mr = len(mr_idx) - N_tr_mr
            if N_te_mr < 0:
                print("Error: Too little MR data even for training.")
                return
            print(f"New Test MR (1): {N_te_mr}")

        tr_mp_idx = mp_idx[:N_tr_mp]
        te_mp_idx = mp_idx[N_tr_mp:]
        
        tr_mr_idx = mr_idx[:N_tr_mr]
        te_mr_idx = mr_idx[N_tr_mr:N_tr_mr+N_te_mr]
        
        train_idx = np.concatenate([tr_mp_idx, tr_mr_idx])
        train_labels = np.concatenate([np.zeros(N_tr_mp), np.ones(N_tr_mr)])
        
        test_idx = np.concatenate([te_mp_idx, te_mr_idx])
        test_labels = np.concatenate([np.zeros(N_te_mp), np.ones(N_te_mr)])
        
        train_shuffle = np.random.permutation(len(train_idx))
        train_idx = train_idx[train_shuffle]
        train_labels = train_labels[train_shuffle]
        
        test_shuffle = np.random.permutation(len(test_idx))
        test_idx = test_idx[test_shuffle]
        test_labels = test_labels[test_shuffle]
        
        print("Extracting features (this may take a moment)...")
        train_data_list = []
        test_data_list = []
        for i, col_name in enumerate(feature_cols):
            if i % 20 == 0:
                print(f" Reading column {i}/{len(feature_cols)}...")
            full_col = f[col_name][:]
            train_data_list.append(full_col[train_idx])
            test_data_list.append(full_col[test_idx])
            
        X_train = np.column_stack(train_data_list)
        X_test = np.column_stack(test_data_list)
        
        y_train = train_labels
        y_test = test_labels

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    # --- Optimization: Pre-normalize Data ---
    # L2-normalize input features once here instead of in every forward pass
    print("Pre-normalizing data (L2)...")
    if args.use_ebv:
        # ebv is the last column, do not normalize it
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
    else:
        train_norms = np.linalg.norm(X_train, axis=1, keepdims=True)
        X_train = X_train / (train_norms + 1e-8)
        
        test_norms = np.linalg.norm(X_test, axis=1, keepdims=True)
        X_test = X_test / (test_norms + 1e-8)

    # --- Device Selection & Optimization ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    if device.type == 'cuda':
        # Enable CuDNN benchmark for fixed-size inputs (speedup)
        if torch.backends.cudnn.is_available():
            print("Enabling CuDNN Benchmark...")
            torch.backends.cudnn.benchmark = True
        
        # Enable TF32 on Ampere+ GPUs
        if not args.no_tf32 and torch.cuda.get_device_capability(device)[0] >= 8:
            print("Enabling TF32 for Ampere+ GPUs...")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
    # --- Optimization: Move ALL data to device immediately ---
    # Since dataset is small (~10MB), we put it all on VRAM/RAM once to avoid 
    # PCIe transfer overhead during training loop.
    print(f"Moving data to {device}...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1)

    model = LinearClassifier(input_dim=len(feature_cols)).to(device)
    
    # --- Optimization: torch.compile ---
    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
    
    criterion = nn.BCEWithLogitsLoss()

    if cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay'],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
        )

    from torch.utils.data import TensorDataset, DataLoader

    # Data is already on device, so num_workers must be 0
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg['batch_size'], shuffle=False)

    epochs = cfg['epochs']

    from torch.optim.lr_scheduler import LinearLR
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=cfg['lr_end_factor'],
        total_iters=epochs,
    )

    train_losses = []
    test_losses = []
    test_accs = []

    print("Starting training...")
    for epoch in range(epochs):
        # ---- Forward + backward pass ----
        model.train()
        for batch_x, batch_y in train_loader:
            # batch_x, batch_y are already on device
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # ---- Eval pass: train loss (post-epoch, comparable to test loss) ----
        model.eval()
        epoch_train_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                # batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                epoch_train_loss += criterion(out, batch_y).item() * batch_x.size(0)
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)

        # ---- Eval pass: test loss + accuracy + precision/recall ----
        epoch_test_loss, correct_test = 0.0, 0
        TP0 = FP0 = FN0 = TP1 = FP1 = FN1 = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                out = model(batch_x)
                epoch_test_loss += criterion(out, batch_y).item() * batch_x.size(0)
                preds = (out > 0.0).float()
                correct_test += (preds == batch_y).sum().item()
                
                y_0 = (batch_y == 0); p_0 = (preds == 0)
                y_1 = (batch_y == 1); p_1 = (preds == 1)
                
                TP0 += (p_0 & y_0).sum().item(); FP0 += (p_0 & y_1).sum().item(); FN0 += (p_1 & y_0).sum().item()
                TP1 += (p_1 & y_1).sum().item(); FP1 += (p_1 & y_0).sum().item(); FN1 += (p_0 & y_1).sum().item()

        epoch_test_loss /= len(test_dataset)
        test_acc = correct_test / len(test_dataset)
        test_losses.append(epoch_test_loss)
        test_accs.append(test_acc)

        if (epoch+1) % 5 == 0 or epoch == 0:
            prec0 = TP0 / (TP0 + FP0) if (TP0 + FP0) > 0 else float('nan')
            rec0  = TP0 / (TP0 + FN0) if (TP0 + FN0) > 0 else float('nan')
            prec1 = TP1 / (TP1 + FP1) if (TP1 + FP1) > 0 else float('nan')
            rec1  = TP1 / (TP1 + FN1) if (TP1 + FN1) > 0 else float('nan')
            print(
                f"Epoch {epoch+1:3d}/{epochs}"
                f" | TrLoss {epoch_train_loss:.4f}  TeLoss {epoch_test_loss:.4f}  Acc {test_acc:.4f}"
                f" | MP(0): P={prec0:.4f} R={rec0:.4f}"
                f" | MR(1): P={prec1:.4f} R={rec1:.4f}"
            )

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('BCE Loss', fontsize=12)
    l1 = ax1.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue', linewidth=2)
    l2 = ax1.plot(range(1, epochs+1), test_losses, label='Test Loss', color='red', linewidth=2)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', fontsize=12, color='green')
    l3 = ax2.plot(range(1, epochs+1), test_accs, label='Test Accuracy', color='green', linewidth=2, linestyle=':')
    ax2.tick_params(axis='y', labelcolor='green')

    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=12, loc='center right')

    plt.title("Train/Test Loss & Test Accuracy (Linear Classifier)", fontsize=14)
    
    fig.tight_layout()
    out_img = 'train_test_loss.png'
    if args.use_ebv:
        out_img = 'train_test_loss_ebv.png'
    fig.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Training complete. Loss curve saved to {out_img}")
    print(f"Final Test Accuracy: {test_accs[-1]:.4f}")

    # --- Save Weights ---
    print("Saving model weights...")
    weights = model.fc.weight.detach().cpu().numpy().flatten()
    bias = model.fc.bias.detach().cpu().item()
    
    out_csv = 'linear_model_weights.csv'
    if args.use_ebv:
        out_csv = 'linear_model_weights_ebv.csv'

    with open(out_csv, 'w') as f:
        f.write("feature,weight\n")
        f.write(f"BIAS,{bias}\n")
        for name, w in zip(feature_cols, weights):
            f.write(f"{name},{w}\n")
    print(f"Weights saved to {out_csv} (Total features: {len(feature_cols)})")

if __name__ == "__main__":
    main()
