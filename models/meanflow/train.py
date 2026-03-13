#!/usr/bin/env python3
"""Train MeanFlow model from notebook 02."""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.utils import make_grid, save_image

from model import MeanFlowModel
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
    transform = Compose([ToTensor(), Lambda(lambda x: x * 2 - 1)])
    if dataset == "cifar10":
        train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
        test_dataset = CIFAR10(dataset_path, transform=transform, train=False, download=True)
    else:
        train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
        test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def sample_r_t(batch_size, flow_ratio, device):
    t = torch.rand(batch_size, 1, 1, 1, device=device)
    r = torch.rand_like(t) * t
    mask = torch.rand_like(t) < flow_ratio
    r[mask] = t[mask]
    return r, t


def jvp(fn, z_t, r, t, v):
    _, du_dt = torch.autograd.functional.jvp(
        fn,
        (z_t, r, t),
        (v, torch.zeros_like(r), torch.ones_like(t)),
        create_graph=True,
    )
    return du_dt


def save_training_curve(run_dir: Path, history):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax.set_title("MeanFlow Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "train_loss.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_sample_grid(model, run_dir: Path, n_steps, num_sample_images, img_size, device):
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    h, w, c = img_size
    generated = model.sample(
        shape=[num_sample_images, c, h, w],
        n_steps=n_steps,
        device=device,
    )
    save_image(generated, samples_dir / f"samples_steps{n_steps}.png", nrow=8, normalize=True)

    fig = plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title(f"MeanFlow samples (steps={n_steps})")
    plt.imshow(make_grid(generated.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
    fig.tight_layout()
    fig.savefig(samples_dir / f"grid_steps{n_steps}.png", dpi=150)
    plt.close(fig)


def train(settings, run_dir: Path):
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(settings["seed"])
    device = resolve_device(settings)
    model = MeanFlowModel(
        image_resolution=settings["img_size"],
        hidden_dims=settings["hidden_dims"],
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=settings["lr"], betas=(0.9, 0.99))
    train_loader, _ = load_dataset(
        settings["dataset"],
        settings["dataset_path"],
        settings["train_batch_size"],
        settings["inference_batch_size"],
        settings["num_workers"],
    )

    history = {"epoch": [], "train_loss": []}
    start = time.time()
    print("Start training MeanFlow...")
    model.train()

    for epoch in range(settings["n_epochs"]):
        total_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            e = torch.randn_like(x)
            r, t = sample_r_t(x.shape[0], settings["flow_ratio"], device)

            z_t = model.get_z_t(x, t, e)
            v = model.get_instantaneous_velocity_v(e, x)
            u = model(z_t, r, t)
            du_dt = jvp(model.forward, z_t, r, t, v)
            u_tgt = v - (t - r) * du_dt

            loss = ((u - u_tgt.detach()) ** 2).mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                du_dt_mean = du_dt.abs().mean().item()
                print("\t\tMeanFlow loss:", loss.item(), " grad_norm:", grad_norm, " du_dt_mean:", du_dt_mean)

        epoch_loss = total_loss / len(train_loader)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(epoch_loss))
        print("\tEpoch", epoch + 1, "complete!  mean flow loss:", epoch_loss)

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
        n_steps=settings["sample_steps"],
        num_sample_images=settings["num_sample_images"],
        img_size=settings["img_size"],
        device=device,
    )
    return {"history": history, "elapsed_sec": elapsed_sec}


def main():
    args = build_train_parser().parse_args()
    run_dir = prepare_run_dir(args.run_dir, "meanflow")
    config = load_yaml_config(Path(args.config))
    settings = resolve_train_settings(args, config)

    out = train(settings, run_dir)
    metrics = {"pipeline": "meanflow_mnist", "elapsed_sec": out["elapsed_sec"], "settings": settings}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "history.json").write_text(json.dumps(out["history"], indent=2))

    print("Finish!!")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
