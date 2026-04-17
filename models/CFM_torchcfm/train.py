#!/usr/bin/env python3
"""Train CFM model using the torchcfm library (Tong et al.)."""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.models.unet import UNetModel as UNetModelWrapper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from noise import build_noise_sampler
from utils import (
    build_train_parser,
    load_yaml_config,
    prepare_run_dir,
    resolve_device,
    resolve_train_settings,
    seed_everything,
)


def load_dataset(dataset, dataset_path, train_batch_size, inference_batch_size, num_workers):
    from torch.utils.data import DataLoader

    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    transform = ToTensor()
    if dataset == "cifar10":
        train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
        test_dataset = CIFAR10(dataset_path, transform=transform, train=False, download=True)
    else:
        train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
        test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def save_training_curve(run_dir: Path, history):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax.set_title("CFM (torchcfm) Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "train_loss.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def sample_euler(model, t_steps, shape, device, noise_sampler=None):
    """Generate samples by Euler-integrating the learned ODE.
    If noise_sampler is provided, initial x_0 is drawn from it."""
    if noise_sampler is not None:
        x = noise_sampler.sample(shape, device)
    else:
        x = torch.randn(size=shape, device=device)
    delta = 1.0 / max(t_steps - 1, 1)
    t_vals = torch.linspace(0, 1, t_steps, device=device)

    for i in range(t_steps - 1):
        t_cur = torch.full((shape[0],), t_vals[i].item(), device=device)
        x = x + model(t_cur, x) * delta
    return x


@torch.no_grad()
def save_sample_grid(model, run_dir: Path, sample_steps, num_sample_images, img_size, device, noise_sampler=None):
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    h, w, c = img_size
    generated = sample_euler(model, t_steps=sample_steps, shape=[num_sample_images, c, h, w], device=device, noise_sampler=noise_sampler)
    save_image(generated, samples_dir / f"samples_steps{sample_steps}.png", nrow=8, normalize=True)

    fig = plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title(f"torchcfm samples (steps={sample_steps})")
    plt.imshow(make_grid(generated.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
    fig.tight_layout()
    fig.savefig(samples_dir / f"grid_steps{sample_steps}.png", dpi=150)
    plt.close(fig)


def build_cfm(settings):
    """Build the flow matcher object based on config."""
    sigma = settings["cfm_sigma"]
    if settings["cfm_variant"] == "otcfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    return ConditionalFlowMatcher(sigma=sigma)


def train(settings, run_dir: Path):
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(settings["seed"])
    device = resolve_device(settings)

    h, w, c = settings["img_size"]
    model = UNetModelWrapper(
        dim=(c, h, w),
        num_channels=settings["num_channels"],
        num_res_blocks=settings["num_res_blocks"],
    ).to(device)

    noise_sampler = build_noise_sampler(settings)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"torchcfm UNet | Noise: {settings['noise_source']} | Params: {nparams:,} | CFM variant: {settings['cfm_variant']}")

    optimizer = AdamW(model.parameters(), lr=settings["lr"], betas=(0.9, 0.99))
    cfm = build_cfm(settings)

    train_loader, _ = load_dataset(
        settings["dataset"],
        settings["dataset_path"],
        settings["train_batch_size"],
        settings["inference_batch_size"],
        settings["num_workers"],
    )

    history = {"epoch": [], "train_loss": []}
    print("Start training (torchcfm)...")
    model.train()
    start = time.time()

    for epoch in range(settings["n_epochs"]):
        total_loss = 0.0
        for batch_idx, (x_1, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_1 = x_1.to(device)
            x_0 = noise_sampler.sample_like(x_1)

            # torchcfm computes x_t and u_t for us (Eq.14 + Eq.15)
            t, x_t, u_t = cfm.sample_location_and_conditional_flow(x_0, x_1)
            t = t.to(device)
            x_t = x_t.to(device)
            u_t = u_t.to(device)

            # torchcfm UNet signature: forward(t, x) where t is shape (B,)
            v_pred = model(t, x_t)

            loss = ((v_pred - u_t) ** 2).mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                print(f"\t\tCFM loss: {loss.item():.6f}  grad_norm: {grad_norm:.4f}")

        epoch_loss = total_loss / len(train_loader)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(epoch_loss))
        print(f"\tEpoch {epoch + 1} complete!\tCFM loss: {epoch_loss:.6f}")

        if (epoch + 1) % settings["checkpoint_every"] == 0 or epoch + 1 == settings["n_epochs"]:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "settings": settings,
                    "epoch_loss": float(epoch_loss),
                },
                checkpoints_dir / f"epoch_{epoch + 1:04d}.pt",
            )

    elapsed_sec = round(time.time() - start, 4)
    torch.save(
        {"model_state_dict": model.state_dict(), "settings": settings},
        checkpoints_dir / "last.pt",
    )
    save_training_curve(run_dir, history)
    save_sample_grid(
        model=model,
        run_dir=run_dir,
        sample_steps=settings["sample_steps"],
        num_sample_images=settings["num_sample_images"],
        img_size=settings["img_size"],
        device=device,
        noise_sampler=noise_sampler,
    )
    return {"history": history, "elapsed_sec": elapsed_sec}


def main():
    args = build_train_parser().parse_args()
    run_dir = prepare_run_dir(args.run_dir, "CFM_torchcfm")
    config = load_yaml_config(Path(args.config))
    settings = resolve_train_settings(args, config)

    out = train(settings, run_dir)
    metrics = {"pipeline": "cfm_torchcfm", "elapsed_sec": out["elapsed_sec"], "settings": settings}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "history.json").write_text(json.dumps(out["history"], indent=2))

    print("Finish!!")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
