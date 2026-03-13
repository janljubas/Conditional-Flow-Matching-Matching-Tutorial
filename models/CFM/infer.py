#!/usr/bin/env python3
"""Inference script for trained CFM checkpoints."""

import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from model import CFMModel
from utils import (
    build_infer_parser,
    load_yaml_config,
    prepare_run_dir,
    resolve_device,
    resolve_infer_settings,
    seed_everything,
)


@torch.no_grad()
def run_inference(settings, checkpoint_path: Path, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(settings["seed"])
    device = resolve_device(settings)
    model = CFMModel(
        image_resolution=settings["img_size"],
        hidden_dims=settings["hidden_dims"],
        sigma_min=settings["sigma_min"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    h, w, c = settings["img_size"]
    generated = model.sample(
        t_steps=settings["sample_steps"],
        shape=[settings["num_sample_images"], c, h, w],
        device=device,
    )
    out_path = samples_dir / f"samples_steps{settings['sample_steps']}.png"
    save_image(generated, out_path, nrow=8, normalize=True)

    metrics = {
        "pipeline": "cfm_inference",
        "checkpoint": str(checkpoint_path),
        "output_image": str(out_path),
        "settings": settings,
    }
    (run_dir / "inference_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def main():
    args = build_infer_parser().parse_args()
    config = load_yaml_config(Path(args.config))
    settings = resolve_infer_settings(args, config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    run_dir = prepare_run_dir(args.run_dir, "CFM")
    run_inference(settings, checkpoint_path, run_dir)


if __name__ == "__main__":
    main()
