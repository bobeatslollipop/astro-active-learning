import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import re
import os
import argparse

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def evaluate_all(model, device, out_dir):
    from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_recall_fscore_support
    print("Evaluating on all data...")

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
        
        y_true = (feh_valid >= -2.0).astype(int) 
        
        print("Loading and normalizing all data for evaluation...")
        X_all = []
        for col_name in feature_cols:
             X_all.append(np.nan_to_num(f[col_name][:][valid_mask]))
        
        X_all = np.column_stack(X_all)
        
        X_to_norm = X_all[:, :-1]
        norms = np.linalg.norm(X_to_norm, axis=1, keepdims=True) + 1e-8
        X_all[:, :-1] = X_to_norm / norms
        
        batch_size = 50000
        y_pred = np.zeros(len(y_true), dtype=int)
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_all), batch_size):
                batch_x = torch.tensor(X_all[i:i+batch_size], dtype=torch.float32, device=device)
                out = model(batch_x)
                preds = (out > 0.0).cpu().numpy().astype(int).flatten()
                y_pred[i:i+batch_size] = preds
                
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate precision and recall for both classes
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    
    print(f"Overall Accuracy on all data: {acc:.4%}")
    print(f"Class MP (0): Precision = {precision[0]:.4f}, Recall = {recall[0]:.4f}")
    print(f"Class MR (1): Precision = {precision[1]:.4f}, Recall = {recall[1]:.4f}\n")
    
    print("Confusion Matrix:")
    print("                 | Pred MP (0) | Pred MR (1)")
    print("--------------------------------------------")
    print(f"True MP (0)     | {cm[0, 0]:11d} | {cm[0, 1]:11d}")
    print(f"True MR (1)     | {cm[1, 0]:11d} | {cm[1, 1]:11d}")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['MP (Fe/H < -2)', 'MR (Fe/H >= -2)'])
    disp.plot(cmap='Blues', ax=ax, values_format=',') 
    from matplotlib.colors import LogNorm
    try:
        disp.im_.set_norm(LogNorm(vmin=max(cm.min(), 1), vmax=cm.max()))
    except Exception:
        pass

    title_str = f'Overall Evaluation\nAcc: {acc:.2%}  MP(P:{precision[0]:.3f}, R:{recall[0]:.3f}) MR(P:{precision[1]:.3f}, R:{recall[1]:.3f})'
    plt.title(title_str, fontsize=11)
    
    plt.tight_layout()
    out_file = os.path.join(out_dir, 'confusion_matrix_all_data.png')
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {out_file}.")

class TwoLayerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    parser = argparse.ArgumentParser(description="Train a two layer classifier on stellar data.")
    p_add = parser.add_argument
    p_add('--device', type=str, default=None, help="Device to use ('cuda', 'cpu', 'mps')")
    p_add('--list-hardware', action='store_true', help="List available hardware and exit.")
    p_add('--no-tf32', action='store_true', help="Disable TF32 for Ampere+ GPUs.")
    p_add('--compile', action='store_true', help="Use torch.compile().")

    p_add('--run-name', type=str, default=None, help="Name of the run. Outputs will be saved to twolayers_{run_name}/.")
    p_add('--seed', type=int, default=42, help="Random seed for reproducibility.")
    p_add('--data-split', type=str, default='random', choices=['random', 'low_temp'], help="Which dataset split to use. Defaults to 'random'.")
    p_add('--train-file', type=str, default=None, help="Path to train H5 file. Overrides --data-split.")
    p_add('--test-file', type=str, default=None, help="Path to test H5 file. Overrides --data-split.")
    p_add('--feh-threshold', type=float, default=-2.0, help="[Fe/H] threshold defining the boundary between MP and MR classes.")
    p_add('--hidden-size', type=int, default=16, help="Hidden dimension size for the two layer network.")
    p_add('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help="Optimizer type: 'adam' or 'sgd'.")
    p_add('--lr', type=float, default=1.0, help="Initial learning rate.")
    p_add('--momentum', type=float, default=0.0, help="Momentum factor for SGD optimizer. Ignored if --optimizer=adam.")
    p_add('--weight-decay', type=float, default=0.0, help="Weight decay factor for L2 regularization.")
    p_add('--epochs', type=int, default=50, help="Number of training epochs.")
    p_add('--batch-size', type=int, default=30000, help="Batch size for training.")
    p_add('--lr-end-factor', type=float, default=1.0, help="Final learning rate multiplier (linear scheduler).")
    p_add('--lambda-MP', type=float, default=1.0, help="Reweight factor for MP class. MP weight = lambda_MP / (1+lambda_MP), MR weight = 1/(1+lambda_MP).")
    args = parser.parse_args()

    if args.run_name:
        out_dir = f"twolayers_{args.run_name}"
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "."

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 2. Load data
    train_file = args.train_file
    test_file = args.test_file

    def find_data_split(split, root='.'):
        # 1. Look for a directory that contains the split name (e.g., 'random' or 'low_temp')
        found_dir = None
        for dirpath, dirnames, _ in os.walk(root):
            for d in dirnames:
                if split.lower() in d.lower():
                    found_dir = os.path.join(dirpath, d)
                    break
            if found_dir:
                break
        
        if not found_dir:
            return None, None
            
        # 2. In that directory, look for .h5 files containing 'train' and 'test'
        t_f, te_f = None, None
        for f in os.listdir(found_dir):
            if f.endswith('.h5'):
                if 'train' in f.lower():
                    t_f = os.path.join(found_dir, f)
                elif 'test' in f.lower():
                    te_f = os.path.join(found_dir, f)
        return t_f, te_f

    if train_file is None or test_file is None:
        train_file, test_file = find_data_split(args.data_split)

    if train_file is None or test_file is None or not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Error: Could not find training/test h5 files for split '{args.data_split}'.")
        print("Please ensure a subdirectory exists with the split name and contains H5 files with 'train' and 'test' in their names.")
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
        
        thr = args.feh_threshold
        y_train = (f_tr['feh'][:] >= thr).astype(np.float32)
        y_test = (f_te['feh'][:] >= thr).astype(np.float32)

        N_tr_mp = np.sum(y_train == 0)
        N_tr_mr = np.sum(y_train == 1)
        N_te_mp = np.sum(y_test == 0)
        N_te_mr = np.sum(y_test == 1)
        print(f"Train MP (0): {int(N_tr_mp)}, Train MR (1): {int(N_tr_mr)}  [Total Train: {len(y_train)}]")
        print(f"Test MP (0): {int(N_te_mp)}, Test MR (1): {int(N_te_mr)}   [Total Test:  {len(y_test)}]")

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    # --- Optimization: Pre-normalize Data ---
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

    # --- Device Selection & Optimization ---
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

    model = TwoLayerClassifier(input_dim=len(feature_cols), hidden_size=args.hidden_size).to(device)
    
    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
    
    w_mr = 1.0 / (1.0 + args.lambda_MP)
    w_mp = args.lambda_MP / (1.0 + args.lambda_MP)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    epochs = args.epochs

    from torch.optim.lr_scheduler import LinearLR
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args.lr_end_factor,
        total_iters=epochs,
    )

    train_losses = []
    test_losses = []
    test_accs = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss_unreduced = criterion(out, batch_y)
            w = w_mr * batch_y + w_mp * (1 - batch_y)
            loss = (loss_unreduced * w).mean()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        epoch_train_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                out = model(batch_x)
                loss_unreduced = criterion(out, batch_y)
                w = w_mr * batch_y + w_mp * (1 - batch_y)
                loss = (loss_unreduced * w).mean()
                epoch_train_loss += loss.item() * batch_x.size(0)
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)

        epoch_test_loss, correct_test = 0.0, 0
        TP0 = FP0 = FN0 = TP1 = FP1 = FN1 = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                out = model(batch_x)
                loss_unreduced = criterion(out, batch_y)
                w = w_mr * batch_y + w_mp * (1 - batch_y)
                loss = (loss_unreduced * w).mean()
                epoch_test_loss += loss.item() * batch_x.size(0)
                preds = (out > 0.0).float()
                correct_test += (preds == batch_y).sum().item()
                
                y_0, p_0 = (batch_y == 0), (preds == 0)
                y_1, p_1 = (batch_y == 1), (preds == 1)
                
                TP0 += (p_0 & y_0).sum().item()
                FP0 += (p_0 & y_1).sum().item()
                FN0 += (p_1 & y_0).sum().item()
                
                TP1 += (p_1 & y_1).sum().item()
                FP1 += (p_1 & y_0).sum().item()
                FN1 += (p_0 & y_1).sum().item()

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

    plt.title(r"Train/Test Loss & Test Accuracy (Two Layers, $\lambda_{MP}$=" + f"{args.lambda_MP})", fontsize=14)
    
    fig.tight_layout()
    out_img = 'twolayers.png'
    out_img = os.path.join(out_dir, out_img)
    fig.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Training complete. Loss curve saved to {out_img}")
    print(f"Final Test Accuracy: {test_accs[-1]:.4f}")

    # --- Save Weights ---
    print("Saving model weights to PyTorch model file...")
    out_model = 'twolayers_model.pt'
    
    out_model = os.path.join(out_dir, out_model)
    torch.save(model.state_dict(), out_model)
    print(f"Model saved to {out_model}")

    evaluate_all(model, device, out_dir)

    import json
    params_file = os.path.join(out_dir, "params.json")
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved run parameters to {params_file}")

if __name__ == "__main__":
    main()
