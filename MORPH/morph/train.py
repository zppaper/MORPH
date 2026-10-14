import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from .data import load_data
from .metrics import compute_metrics
from .utils import set_seeds

def collect_probs_and_labels(model, data_loader, device):
    (all_probs, all_true) = ([], [])
    model.eval()
    with torch.no_grad():
        for (bx_m, bx_mi, by) in data_loader:
            (bx_m, bx_mi) = (bx_m.to(device), bx_mi.to(device))
            logits = model(bx_m, bx_mi)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            if np.ndim(probs) == 0:
                probs = np.expand_dims(probs, 0)
            all_probs.extend(probs.tolist())
            all_true.extend(by.numpy().tolist())
    return (np.array(all_true).astype(int), np.array(all_probs, dtype=np.float64))

def run_experiment(dataset_name, seed, config, device, model_class):
    fold_records = []
    set_seeds(seed)
    res = load_data(dataset_name, config['data_root'], config['prior_dir'], config['driver_file'])
    if res[0] is None:
        raise FileNotFoundError(f'Missing input files for {dataset_name}')
    (df_mrna, df_mirna, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, path_prior, y) = res
    X_mrna = df_mrna.values.astype(np.float32)
    X_mirna = df_mirna.values.astype(np.float32)
    skf = StratifiedKFold(n_splits=config['folds'], shuffle=True, random_state=seed)
    fold_metrics = []
    for (fold_idx, (train_idx, test_idx)) in enumerate(skf.split(X_mrna, y)):
        (X_m_train, X_m_test) = (X_mrna[train_idx], X_mrna[test_idx])
        (X_mi_train, X_mi_test) = (X_mirna[train_idx], X_mirna[test_idx])
        (y_train, y_test) = (y[train_idx], y[test_idx])
        (sub_train_idx, sub_val_idx) = train_test_split(np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=seed)
        (tm_x, vm_x) = (X_m_train[sub_train_idx], X_m_train[sub_val_idx])
        (tmi_x, vmi_x) = (X_mi_train[sub_train_idx], X_mi_train[sub_val_idx])
        (ty, vy) = (y_train[sub_train_idx], y_train[sub_val_idx])
        (test_m_x, test_mi_x) = (X_m_test, X_mi_test)
        scaler_m = StandardScaler().fit(tm_x)
        tm_x = scaler_m.transform(tm_x)
        vm_x = scaler_m.transform(vm_x)
        test_m_x = scaler_m.transform(test_m_x)
        scaler_mi = StandardScaler().fit(tmi_x)
        tmi_x = scaler_mi.transform(tmi_x)
        vmi_x = scaler_mi.transform(vmi_x)
        test_mi_x = scaler_mi.transform(test_mi_x)
        train_dataset = TensorDataset(torch.tensor(tm_x), torch.tensor(tmi_x), torch.tensor(ty).float())
        val_dataset = TensorDataset(torch.tensor(vm_x), torch.tensor(vmi_x), torch.tensor(vy).float())
        test_dataset = TensorDataset(torch.tensor(test_m_x), torch.tensor(test_mi_x), torch.tensor(y_test).float())
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        model = model_class(dim_mrna=X_mrna.shape[1], dim_mirna=X_mirna.shape[1], gene_path_mask_df=gene_path_mask_df, mirna_path_mask_df=mirna_path_mask_df, reg_matrix=reg_matrix, driver_idx=driver_idx, path_prior=path_prior, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.0001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_score = float('-inf')
        patience_counter = 0
        for epoch in range(config['epochs']):
            model.train()
            for (bx_m, bx_mi, by) in train_loader:
                (bx_m, bx_mi, by) = (bx_m.to(device), bx_mi.to(device), by.to(device))
                logits_fusion = model(bx_m, bx_mi)
                loss = criterion(logits_fusion.squeeze(), by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            (val_probs, val_true) = ([], [])
            with torch.no_grad():
                for (bx_m, bx_mi, by) in val_loader:
                    (bx_m, bx_mi) = (bx_m.to(device), bx_mi.to(device))
                    logits = model(bx_m, bx_mi)
                    probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                    if np.ndim(probs) == 0:
                        probs = np.expand_dims(probs, 0)
                    val_probs.extend(probs.tolist())
                    val_true.extend(by.numpy().tolist())
            try:
                val_auc = roc_auc_score(val_true, val_probs)
            except:
                val_auc = 0.5
            if val_auc > best_val_score:
                best_val_score = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    break
        (val_true, val_probs) = collect_probs_and_labels(model, val_loader, device)
        (test_true, test_probs) = collect_probs_and_labels(model, test_loader, device)
        metrics_list = compute_metrics(test_true, test_probs)
        fold_metrics.append(metrics_list)
        fold_records.append({'dataset': dataset_name, 'seed': int(seed), 'fold': int(fold_idx + 1), 'auc': float(metrics_list[0]), 'aupr': float(metrics_list[1])})
        print(f'{dataset_name} seed={seed} fold={fold_idx + 1}: AUC={metrics_list[0]:.4f} AUPR={metrics_list[1]:.4f}', flush=True)
    fold_metrics = np.array(fold_metrics, dtype=np.float64)
    seed_mean = np.nanmean(fold_metrics, axis=0)
    return (seed_mean, fold_records)
