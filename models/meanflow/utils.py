import argparse
import random
from pathlib import Path

import numpy as np
import torch


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_yaml_config(config_path: Path):
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
    return cli_value if cli_value is not None else config.get(key, default)


def _common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--run-dir", required=True, help="Directory to write run outputs.")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--cuda", type=str2bool, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def build_train_parser():
    parser = _common_parser("Train MeanFlow model.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--flow-ratio", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--num-sample-images", type=int, default=None)
    return parser


def build_infer_parser():
    parser = _common_parser("Run MeanFlow inference from checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (*.pt).")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--num-sample-images", type=int, default=None)
    return parser


def resolve_device(settings):
    if settings["cuda"] and torch.cuda.is_available():
        torch.cuda.set_device(settings["gpu_id"])
        return torch.device(f"cuda:{settings['gpu_id']}")
    return torch.device("cpu")


def resolve_train_settings(args, config):
    settings = {
        "dataset_path": str(Path(pick_value(args.dataset_path, config, "dataset_path", "~/datasets")).expanduser()),
        "dataset": str(pick_value(args.dataset, config, "dataset", "mnist")).lower(),
        "gpu_id": int(pick_value(args.gpu_id, config, "gpu_id", 0)),
        "cuda": bool(pick_value(args.cuda, config, "cuda", True)),
        "hidden_dim": int(pick_value(args.hidden_dim, config, "hidden_dim", 256)),
        "n_layers": int(pick_value(args.n_layers, config, "n_layers", 8)),
        "lr": float(pick_value(args.lr, config, "lr", 5e-5)),
        "n_epochs": int(pick_value(args.n_epochs, config, "n_epochs", 200)),
        "train_batch_size": int(pick_value(args.train_batch_size, config, "train_batch_size", 128)),
        "inference_batch_size": int(pick_value(None, config, "inference_batch_size", 64)),
        "flow_ratio": float(pick_value(args.flow_ratio, config, "flow_ratio", 0.5)),
        "seed": int(pick_value(args.seed, config, "seed", 1234)),
        "num_workers": int(pick_value(args.num_workers, config, "num_workers", 1)),
        "checkpoint_every": int(pick_value(args.checkpoint_every, config, "checkpoint_every", 10)),
        "sample_steps": int(pick_value(args.sample_steps, config, "sample_steps", 10)),
        "num_sample_images": int(pick_value(args.num_sample_images, config, "num_sample_images", 64)),
    }
    if settings["dataset"] not in {"mnist", "cifar10"}:
        raise ValueError(f"Unsupported dataset '{settings['dataset']}'.")
    settings["img_size"] = (32, 32, 3) if settings["dataset"] == "cifar10" else (28, 28, 1)
    settings["hidden_dims"] = [settings["hidden_dim"] for _ in range(settings["n_layers"])]
    return settings


def resolve_infer_settings(args, config):
    settings = {
        "dataset": str(pick_value(args.dataset, config, "dataset", "mnist")).lower(),
        "gpu_id": int(pick_value(args.gpu_id, config, "gpu_id", 0)),
        "cuda": bool(pick_value(args.cuda, config, "cuda", True)),
        "hidden_dim": int(pick_value(args.hidden_dim, config, "hidden_dim", 256)),
        "n_layers": int(pick_value(args.n_layers, config, "n_layers", 8)),
        "n_steps": int(pick_value(args.n_steps, config, "sample_steps", 10)),
        "num_sample_images": int(pick_value(args.num_sample_images, config, "num_sample_images", 64)),
        "seed": int(pick_value(args.seed, config, "seed", 1234)),
    }
    if settings["dataset"] not in {"mnist", "cifar10"}:
        raise ValueError(f"Unsupported dataset '{settings['dataset']}'.")
    settings["img_size"] = (32, 32, 3) if settings["dataset"] == "cifar10" else (28, 28, 1)
    settings["hidden_dims"] = [settings["hidden_dim"] for _ in range(settings["n_layers"])]
    return settings


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_run_dir(run_dir_arg, model_name):
    """
    Resolve and create run directory, restricted to repo-local runs/<model_name>/...
    """
    repo_root = Path(__file__).resolve().parents[2]
    allowed_root = (repo_root / "runs" / model_name).resolve()

    run_dir = Path(run_dir_arg).expanduser()
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    run_dir = run_dir.resolve()

    if run_dir != allowed_root and allowed_root not in run_dir.parents:
        raise ValueError(
            f"--run-dir must be inside '{allowed_root}'. Got '{run_dir}'."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
