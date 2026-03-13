import argparse
from pathlib import Path


def build_parser():
    """
    Builds CLI parser for run paths, config, and optional overrides.
    Returns:
        argparse.ArgumentParser: The parser object.
    """
    parser = argparse.ArgumentParser(description="Train a simple Iris dataset classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--run-dir", required=True, help="Directory to write run outputs.")
    parser.add_argument("--dataset-path", required=True, help="Dataset path.", default='~/datasets')

    # Optional CLI overrides. If omitted, values come from config/defaults.
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--cuda", type=bool, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    
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
        "gpu_id": int(pick_value(args.gpu_id, config, "gpu_id", 2)),
        "cuda": bool(pick_value(args.cuda, config, "cuda", True)),
        "hidden_dim": int(pick_value(args.hidden_dim, config, "hidden_dim", 256)),
        "n_layers": int(pick_value(args.n_layers, config, "n_layers", 8)),
        "lr": float(pick_value(args.lr, config, "lr", 5e-5)),
        "sigma_min": float(pick_value(args.sigma_min, config, "sigma_min", 0)),
        "n_epochs": int(pick_value(args.n_epochs, config, "n_epochs", 10)),
        "train_batch_size": int(pick_value(args.train_batch_size, config, "train_batch_size", 128)),
        "inference_batch_size": int(pick_value(args.inference_batch_size, config, "inference_batch_size", 64)),
        "seed": int(pick_value(args.seed, config, "seed", 1234)),
    }
    if settings["dataset"] != "mnist":
        raise ValueError(f"Unsupported dataset '{settings['dataset']}'. Only 'mnist' is supported.")
    
    return settings

