#!/usr/bin/env python3
"""Minimal training smoke test for HPC pipeline verification."""

import argparse
import json
from pathlib import Path
import time


def build_parser():
    """
    Builds CLI parser for run paths, config, and optional overrides.
    Returns:
        argparse.ArgumentParser: The parser object.
    """
    parser = argparse.ArgumentParser(description="Train a simple Iris dataset classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--run-dir", required=True, help="Directory to write run outputs.")

    # Optional CLI overrides. If omitted, values come from config/defaults.
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser


def load_yaml_config(config_path: Path):
    """
    Loads the YAML config form the given config file path using PyYAML.
    """
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read config files. Install with: pip install pyyaml"
        ) from exc

    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping/dictionary.")
    return config


def pick_value(cli_value, config, key, default):
    """
    Helper function to pick a value from the CLI, config, or default.

    Returns the CLI value if set, else the config value, else the default value.
    """
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def resolve_run_settings(args, config):
    """
    Resolves the final run settings from CLI overrides + config + defaults.
    It makes sure that the model training method doesn't receive invalid values; it throws an error beforehand (easier to debug).

    Returns a dictionary of the final run settings (called "settings" in the code, instead of "config").
    """
    settings = {
        "dataset": str(pick_value(args.dataset, config, "dataset", "iris")).lower(),
        "model": str(pick_value(args.model, config, "model", "sgd_logistic")).lower(),
        "test_size": float(pick_value(args.test_size, config, "test_size", 0.2)),
        "random_state": int(pick_value(args.random_state, config, "random_state", 42)),
        "epochs": int(pick_value(args.epochs, config, "epochs", 40)),
        "learning_rate": float(
            pick_value(args.learning_rate, config, "learning_rate", 0.03)
        ),
    }
    if settings["dataset"] != "iris":
        raise ValueError(f"Unsupported dataset '{settings['dataset']}'. Only 'iris' is supported.")
    if settings["model"] not in {"sgd_logistic", "logistic_regression"}:
        raise ValueError(
            f"Unsupported model '{settings['model']}'. Use 'sgd_logistic' or 'logistic_regression'."
        )
    return settings


def train_iris(settings):
    """
    Trains on Iris and collects epoch-wise history plus final predictions.
    Returns a dictionary of the training results.
    """
    try:
        from sklearn.datasets import load_iris
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import accuracy_score, classification_report, log_loss
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for this script. Install with: pip install scikit-learn"
        ) from exc

    dataset = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=settings["test_size"],
        random_state=settings["random_state"],
        stratify=dataset.target,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=settings["learning_rate"],
        random_state=settings["random_state"],
    )

    classes = sorted(set(y_train.tolist()))
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, settings["epochs"] + 1):
        if epoch == 1:
            clf.partial_fit(x_train, y_train, classes=classes)
        else:
            clf.partial_fit(x_train, y_train)

        train_proba = clf.predict_proba(x_train)
        val_proba = clf.predict_proba(x_test)
        train_pred = clf.predict(x_train)
        val_pred = clf.predict(x_test)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(log_loss(y_train, train_proba, labels=classes)))
        history["val_loss"].append(float(log_loss(y_test, val_proba, labels=classes)))
        history["train_acc"].append(float(accuracy_score(y_train, train_pred)))
        history["val_acc"].append(float(accuracy_score(y_test, val_pred)))

    final_pred = clf.predict(x_test)
    final_acc = float(accuracy_score(y_test, final_pred))
    class_report = classification_report(y_test, final_pred, output_dict=True)

    return {
        "dataset": dataset,
        "classifier": clf,
        "history": history,
        "x_test": x_test,
        "y_test": y_test,
        "y_pred": final_pred,
        "metrics": {
            "pipeline": "sklearn_iris_sgd_logistic",
            "num_train": int(len(x_train)),
            "num_test": int(len(x_test)),
            "accuracy": final_acc,
            "classification_report": class_report,
            "settings": settings,
        },
    }


def save_results(run_dir: Path, train_out):
    """
    Writes the metrics JSON and a small train.log summary file.
    """
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(train_out["metrics"], indent=2))

    log_path = run_dir / "train.log"
    log_path.write_text(
        "Iris training smoke test finished successfully.\n"
        f"Pipeline: {train_out['metrics']['pipeline']}\n"
        f"Accuracy: {train_out['metrics']['accuracy']:.4f}\n"
    )


def save_visualizations(run_dir: Path, train_out):
    """
    Saves the training curves, confusion matrix, and coefficient plots.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualizations. Install with: pip install matplotlib"
        ) from exc

    history = train_out["history"]
    dataset = train_out["dataset"]
    clf = train_out["classifier"]

    # 1) Train/validation curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Log Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], label="train_acc")
    axes[1].plot(history["epoch"], history["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # 2) Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        train_out["y_test"],
        train_out["y_pred"],
        display_labels=dataset.target_names,
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 3) Coefficients per feature/class
    coef = clf.coef_
    feature_names = list(dataset.feature_names)
    fig, ax = plt.subplots(figsize=(9, 5))
    for class_idx in range(coef.shape[0]):
        ax.plot(feature_names, coef[class_idx], marker="o", label=f"class {class_idx}")
    ax.set_title("Model Coefficients by Class")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Coefficient Value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "coefficients.png", dpi=150)
    plt.close(fig)


def main():
    """
    Parse args, train the classification model, and save artifacts in run directory.
    """

    # Parse the arguments from the command line (might not be all of the needed arguments).
    args = build_parser().parse_args()

    # Create the run directory (nothing happens if it already exists)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load the configuration from the YAML config file path.
    config = load_yaml_config(Path(args.config))

    # Resolve the final run settings from CLI overrides + config + defaults.
    settings = resolve_run_settings(args, config)

    # Train the classification model.
    start = time.time()
    train_out = train_iris(settings)
    train_out["metrics"]["elapsed_sec"] = round(time.time() - start, 4)

    save_results(run_dir, train_out)
    save_visualizations(run_dir, train_out)

    print(f"Run directory: {run_dir}")
    print(json.dumps(train_out["metrics"], indent=2))


if __name__ == "__main__":
    main()
