"""
Noise source abstraction for CFM training and sampling.

Supports:
  - "gaussian": standard N(0, I) noise (default)
  - "quantum":  pre-computed boson sampling data from quantum_data/ folder,
                affine-rescaled to zero mean / unit variance per dimension,
                then projected to the target image shape via a fixed random matrix.

The projection is necessary because quantum samples live in R^8 while images
live in R^(C*H*W).  A fixed random matrix preserves the correlation structure
of the quantum distribution while expanding it to the required dimensionality.
The projection seed is fixed so the mapping is deterministic across runs.
"""

import torch
from pathlib import Path


class GaussianNoiseSampler:
    """Standard Gaussian noise -- just wraps torch.randn_like."""

    def __init__(self, **kwargs):
        pass

    def sample_like(self, x):
        """Return Gaussian noise with the same shape/device/dtype as x."""
        return torch.randn_like(x)

    def sample(self, shape, device):
        """Return Gaussian noise of the given shape on the given device."""
        return torch.randn(size=shape, device=device)


class QuantumNoiseSampler:
    """
    Loads pre-computed quantum boson sampling vectors, rescales them,
    and projects to arbitrary target shapes on demand.

    Steps:
      1. Load all .pt files from quantum_data_path, concatenate into [N, D] pool
      2. Affine-rescale per dimension to zero mean, unit variance
      3. On each call, draw B random rows and project R^D -> R^(C*H*W) via
         a fixed random matrix, then reshape to [B, C, H, W]
    """

    def __init__(self, quantum_data_path, projection_seed=42, **kwargs):
        pool = self._load_pool(quantum_data_path)
        self.pool = self._rescale(pool)
        self.qdim = self.pool.shape[1]
        self._proj_cache = {}
        self._projection_seed = projection_seed

    @staticmethod
    def _load_pool(data_path):
        p = Path(data_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Quantum data path not found: {p}")
        files = sorted(p.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {p}")
        chunks = [torch.load(f, map_location="cpu", weights_only=True).float() for f in files]
        pool = torch.cat(chunks, dim=0)
        print(f"[QuantumNoiseSampler] Loaded {pool.shape[0]} samples of dim {pool.shape[1]} from {p}")
        return pool

    @staticmethod
    def _rescale(pool):
        """Affine rescale to zero mean, unit variance per dimension."""
        mu = pool.mean(dim=0, keepdim=True)
        std = pool.std(dim=0, keepdim=True).clamp(min=1e-8)
        return (pool - mu) / std

    def _get_projection(self, target_flat_dim, device, dtype):
        """Return a fixed random projection matrix [qdim, target_flat_dim].
        Cached per (target_flat_dim, device, dtype) to avoid regenerating."""
        key = (target_flat_dim, device, dtype)
        if key not in self._proj_cache:
            gen = torch.Generator().manual_seed(self._projection_seed)
            W = torch.randn(self.qdim, target_flat_dim, generator=gen, dtype=dtype)
            W = W / (self.qdim ** 0.5)
            self._proj_cache[key] = W.to(device)
        return self._proj_cache[key]

    def _draw_and_project(self, batch_size, shape_tail, device, dtype):
        """Draw batch_size quantum samples and project to shape_tail dims."""
        idx = torch.randint(0, self.pool.shape[0], (batch_size,))
        z = self.pool[idx].to(device=device, dtype=dtype)
        flat_dim = 1
        for s in shape_tail:
            flat_dim *= s
        W = self._get_projection(flat_dim, device, dtype)
        projected = z @ W
        return projected.view(batch_size, *shape_tail)

    def sample_like(self, x):
        """Return quantum noise with the same shape/device/dtype as x."""
        return self._draw_and_project(x.shape[0], x.shape[1:], x.device, x.dtype)

    def sample(self, shape, device):
        """Return quantum noise of the given shape on the given device."""
        return self._draw_and_project(shape[0], shape[1:], device, torch.float32)


def build_noise_sampler(settings):
    """Factory: build a noise sampler from settings dict."""
    source = settings.get("noise_source", "gaussian")
    if source == "gaussian":
        return GaussianNoiseSampler()
    elif source == "quantum":
        qpath = settings.get("quantum_data_path")
        if not qpath:
            raise ValueError("noise_source='quantum' requires 'quantum_data_path' in config.")
        return QuantumNoiseSampler(quantum_data_path=qpath)
    else:
        raise ValueError(f"Unknown noise_source '{source}'. Choose 'gaussian' or 'quantum'.")
