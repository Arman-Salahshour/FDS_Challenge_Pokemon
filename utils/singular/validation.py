from constants import *
# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _instantiate_model(model_class: Type, seed: Optional[int], **model_params) -> Any:
    """
    Instantiate a model, intelligently passing `random_state` if:
      - a seed is provided, and
      - the model supports `random_state` in get_params()
    """
    # If we can inspect params, check for random_state
    try:
        params = model_class().get_params()
        supports_random_state = 'random_state' in params
    except Exception:
        supports_random_state = False

    if seed is not None and supports_random_state:
        return model_class(random_state=seed, **model_params)
    return model_class(**model_params)


def _create_standard_pipeline(model_class: Type,
                              seed: Optional[int],
                              **model_params) -> Pipeline:
    """
    Create a standard pipeline: StandardScaler -> model.
    """
    model = _instantiate_model(model_class, seed, **model_params)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])
    return pipeline


# -------------------------------------------------------------------
# Core validation functions
# -------------------------------------------------------------------

def run_simple_validation(X: pd.DataFrame,
                          y: pd.Series,
                          model_class: Type = LogisticRegression,
                          test_size: float = 0.10,
                          seed: Optional[int] = 42,
                          verbose: bool = True,
                          **model_params) -> float:
    """
    Perform a simple train/validation split (default 90/10) and return accuracy.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target labels.
    model_class : type
        The estimator class to use (e.g., LogisticRegression).
    test_size : float
        Proportion of data used for validation.
    seed : int or None
        Random seed for reproducibility. If None, no fixed seed is used.
    verbose : bool
        If True, prints progress and metrics.
    model_params : dict
        Extra parameters passed to the model constructor.

    Returns
    -------
    float
        Validation accuracy.
    """
    if verbose:
        print(f"\n--- Running Simple Validation ({int((1 - test_size) * 100)}/{int(test_size * 100)} Split) ---")
        print(f"Model: {model_class.__name__}")

    # 1. Create the split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    # 2. Define the pipeline
    pipeline = _create_standard_pipeline(model_class, seed, **model_params)

    # 3. Train the model
    pipeline.fit(X_train, y_train)

    # 4. Evaluate the model
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    if verbose:
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("--------------------------------------------")

    return accuracy


def run_kfold_validation(X: pd.DataFrame,
                         y: pd.Series,
                         n_splits: int = 5,
                         model_class: Type = LogisticRegression,
                         seed: Optional[int] = 42,
                         verbose: bool = True,
                         **model_params) -> float:
    """
    Perform robust K-Fold cross-validation (default 5-fold) and return mean accuracy.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target labels.
    n_splits : int
        Number of folds for K-Fold CV.
    model_class : type
        The estimator class to use.
    seed : int or None
        Random seed for reproducibility in KFold and model (if supported).
    verbose : bool
        If True, prints progress and metrics.
    model_params : dict
        Extra parameters passed to the model constructor.

    Returns
    -------
    float
        Mean cross-validation accuracy.
    """
    if verbose:
        print(f"\n--- Running Robust {n_splits}-Fold Cross-Validation ---")
        print(f"Model: {model_class.__name__}")

    # 1. Define the K-Fold strategy
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 2. Define the pipeline
    pipeline = _create_standard_pipeline(model_class, seed, **model_params)

    # 3. Run the cross-validation
    if verbose:
        print("Running cross_val_score...")

    start_time = time.time()
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    end_time = time.time()

    # 4. Report the results
    mean_accuracy = float(np.mean(scores))
    std_accuracy = float(np.std(scores))

    if verbose:
        print(f"K-Fold validation complete. Took {end_time - start_time:.2f} seconds.")
        print(f"Individual Fold Accuracies: {np.round(scores, 4)}")
        print(f"Mean Accuracy:                {mean_accuracy:.4f}")
        print(f"Standard Deviation:           {std_accuracy:.4f}")
        print("--------------------------------------------")

    return mean_accuracy


# -------------------------------------------------------------------
# Optional: simple results store utility for comparing runs
# -------------------------------------------------------------------

def init_results_store() -> Dict[str, Dict[str, float]]:
    """
    Initialize a dictionary to store validation results for multiple experiments.

    Returns
    -------
    dict
        Example structure:
        {
            "logreg_baseline": {
                "simple": 0.83,
                "kfold": 0.81
            },
            ...
        }
    """
    return {}


def store_result(results: Dict[str, Dict[str, float]],
                 experiment_name: str,
                 metric_name: str,
                 score: float) -> None:
    """
    Store a metric under a named experiment in the results dictionary.
    """
    if experiment_name not in results:
        results[experiment_name] = {}
    results[experiment_name][metric_name] = score


# -------------------------------------------------------------------
# Example usage (remove or comment out if using as a library)
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Defined validation functions: 'run_simple_validation' and 'run_kfold_validation'.")
    print("Use 'init_results_store' and 'store_result' to keep track of experiment scores.")
