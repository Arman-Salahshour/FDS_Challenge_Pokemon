from constants import *
from validation import kfold_train_for_config

def tune_hyperparam(
    X, y,
    base_config,
    param_name,
    values,
    results_csv_path,
    plot_path
):
    """
    Tune one hyperparameter at a time.
    - param_name: string key in base_config to override
    - values: list of candidate values
    """
    all_results = []
    curves = {}  # param_value -> val_loss_per_epoch

    for val in values:
        print(f"\n=== Tuning {param_name} = {val} ===")
        cfg = base_config.copy()
        cfg[param_name] = val

        mean_curve, final_loss = kfold_train_for_config(
            X=X,
            y=y,
            hidden_dim=cfg["hidden_dim"],
            fc_hidden_dim=cfg["fc_hidden_dim"],
            bidirectional=cfg["bidirectional"],
            learning_rate=cfg["learning_rate"],
            num_epochs=cfg["num_epochs"],
            batch_size=cfg["batch_size"],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
            n_splits=5
        )

        all_results.append({
            param_name: val,
            "final_mean_val_loss": final_loss
        })
        curves[val] = mean_curve

    # Save summary CSV
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(results_csv_path, index=False)
    print(f"\nSaved results to {results_csv_path}")
    print(df_res.sort_values("final_mean_val_loss"))

    # Plot epoch vs mean val loss
    plt.figure(figsize=(8, 5))

    for val, curve in curves.items():
        # Dynamically create epochs based on the length of the current curve
        epochs = np.arange(1, len(curve) + 1)
        plt.plot(epochs, curve, label=f"{param_name}={val}")

    plt.xlabel("Epoch")
    plt.ylabel("Mean Val Loss (5-fold)")
    plt.title(f"Tuning {param_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Return best value (lowest final loss)
    best_row = df_res.loc[df_res["final_mean_val_loss"].idxmin()]
    best_value = best_row[param_name]
    best_loss = best_row["final_mean_val_loss"]
    print(f"Best {param_name}: {best_value} (final mean val loss = {best_loss:.4f})")
    return best_value, df_res