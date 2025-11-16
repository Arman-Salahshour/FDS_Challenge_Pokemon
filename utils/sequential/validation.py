from constants import *
from sklearn.model_selection import KFold

def kfold_train_for_config(
    X, y,
    hidden_dim=64,
    fc_hidden_dim=64,
    bidirectional=False,
    learning_rate=1e-3,
    dropout=0.2,
    num_layers=1,
    num_epochs=20,
    batch_size=32,
    n_splits=5
):
    """
    Runs K-fold CV for one set of hyperparameters.
    Returns:
      - mean_val_loss_per_epoch: np.array shape (num_epochs,)
      - mean_final_val_loss: float
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_val_loss_curves = []  # list of [epoch_losses] for each fold

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold_idx + 1}/{n_splits}...")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val = X[val_idx],   y[val_idx]

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),   torch.tensor(y_val, dtype=torch.float32))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        model = LSTMClassifier(
            input_dim=X.shape[2],
            hidden_dim=hidden_dim,
            fc_hidden_dim=fc_hidden_dim,
            bidirectional=bidirectional,
            dropout=dropout,
            num_layers=num_layers,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        epoch_val_losses = []

        for epoch in range(1, num_epochs + 1):
            train_loss, _ = run_epoch(train_loader, model, criterion, optimizer)
            val_loss, _   = run_epoch(val_loader,   model, criterion, optimizer=None)
            epoch_val_losses.append(val_loss)

            # print(f"    Epoch {epoch:02d}/{num_epochs} | "
            #       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        fold_val_loss_curves.append(epoch_val_losses)

    # Average across folds
    fold_val_loss_curves = np.array(fold_val_loss_curves)  # (K, num_epochs)
    mean_val_loss_per_epoch = fold_val_loss_curves.mean(axis=0)
    mean_final_val_loss = mean_val_loss_per_epoch[-1]

    return mean_val_loss_per_epoch, mean_final_val_loss