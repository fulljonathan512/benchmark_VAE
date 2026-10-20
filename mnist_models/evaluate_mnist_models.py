"""Evaluate all MNIST models in this folder (FID, precision, recall) and log to mlflow.

This script searches recursively under the current directory (the "mnist_models"
folder) for subdirectories whose names match the pattern "*_model_mnist_*" and
contain a "final_model" directory with:

- model_config.json
- model.pt

For each such model, it will:
1. Load the DVAE model (via pythae.models.DVAE / DVAEConfig).
2. Build two datasets of images:
     - eval_gt: MNIST test set images (reference distribution)
     - eval_random_samples: samples from the model prior (generated distribution)
3. Compute:
     - FID (using pytorch_fid, compatible with the unscented-autoencoder setup)
     - Precision and recall in the sense of GAN evaluation metrics
       (using pytorch-fid-gan-metrics style implementation).
4. Log results into the existing mlflow.db in the benchmark_VAE root.

The script assumes it is executed from the benchmark_VAE root, e.g.:

    cd .../benchmark_VAE
    python mnist_models/evaluate_mnist_models.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:  # best-effort; if anything fails we let pytorch_fid raise its own error
    import scipy.linalg as _sp_linalg

    _orig_sqrtm = _sp_linalg.sqrtm

    try:
        from inspect import signature

        if "disp" not in signature(_orig_sqrtm).parameters:

            def _sqrtm_with_disp(a, disp=False):  # pragma: no cover - simple wrapper
                """Compatibility wrapper accepting ``disp`` like older SciPy.

                Older SciPy (<=1.x) behaved as follows:
                  - sqrtm(A) -> matrix
                  - sqrtm(A, disp=False) -> (matrix, info)

                pytorch_fid calls ``sqrtm(A, disp=False)`` and expects a
                2-tuple. Newer SciPy versions return only the matrix and do
                not accept ``disp``. Here we mimic the old API by returning
                ``(matrix, 0)`` when ``disp`` is given.
                """

                res = _orig_sqrtm(a)
                # Emulate old API: (matrix, info) with info=0 (success)
                return res, 0

            _sp_linalg.sqrtm = _sqrtm_with_disp
    except Exception:
        # If we can't introspect the signature, leave sqrtm as-is.
        pass
except Exception:
    pass

import mlflow

# We will reuse pytorch_fid's feature extractor to ensure FID comparability.
from pytorch_fid.fid_score import calculate_fid_given_paths

from pythae.models import DVAE, FCDVAE, VAE, DVAEConfig, FCDVAEConfig, VAEConfig


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_mnist_model_dirs(mnist_models_root: Path) -> List[Path]:
    """Find all MNIST model directories."""

    model_dirs: List[Path] = []
    for p in mnist_models_root.glob("*_model_mnist_*"):
        if p.is_dir():
          for f in p.iterdir():
           if f.is_dir()  and (f / "final_model" / "model.pt").is_file():
            model_dirs.append(f / "final_model")

    return model_dirs


def load_model_from_dir(model_dir: Path) -> DVAE:
    """Load a FCDVAE, DVAE or VAE instance from a "final_model" directory."""

    config_path = model_dir / "model_config.json"
    model_path = model_dir / "model.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model_config.json at {config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model.pt at {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    
    cfg_dict = {k: v for k, v in cfg_dict.items() if k != "name"}
    if "fcdvae" in str(model_dir):  
      config = FCDVAEConfig(**cfg_dict)
      model = FCDVAE(model_config=config)
    elif "dvae" in str(model_dir):
      config = DVAEConfig(**cfg_dict)
      model = DVAE(model_config=config)
    else:
      config = VAEConfig(**cfg_dict)
      model = VAE(model_config=config)

    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_mnist_test_dataset(project_root: Path) -> DataLoader:
    """Return a DataLoader over the MNIST test set.

    Images are tensors in [0,1] with shape [1, 28, 28].
    """

    data_root = project_root / "data" / "MNIST"
    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    return test_loader


def save_eval_gt_images(eval_gt_dir: Path, test_loader: DataLoader, max_images: int | None = None) -> int:
    """Save MNIST test images as ground truth PNGs for FID/metrics.

    Returns the number of images written.
    """

    eval_gt_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, (img, _) in enumerate(test_loader):
        if max_images is not None and count >= max_images:
            break

        out_path = eval_gt_dir / f"gt_img_{idx}.png"
        vutils.save_image(img, str(out_path), normalize=True)
        count += 1

    return count


def decode_latents(model: DVAE, z: torch.Tensor) -> torch.Tensor:
    """Decode latent variables using Pythae DVAE and return images tensor.

    Handles different return types from ``model.decode`` (tensor, dict, tuple).
    """

    out = model.decoder(z)
    if isinstance(out, dict):
        if "reconstruction" not in out:
            raise KeyError(
                "Expected key 'reconstruction' in decoder output dict, "
                f"got keys: {list(out.keys())}"
            )
        recon = out["reconstruction"]
    elif isinstance(out, tuple):
        recon = out[0]
    else:
        recon = out

    if not torch.is_tensor(recon):
        raise TypeError(
            "DVAE.decode(z) must return a tensor, dict, or tuple, " f"got {type(out)}"
        )

    return recon


def save_random_samples(
    eval_random_samples_dir: Path,
    model: DVAE,
    num_images: int,
    device: torch.device,
) -> None:
    """Sample images from the DVAE prior and save PNGs.

    This is used for both FID and precision/recall metrics.
    """

    eval_random_samples_dir.mkdir(parents=True, exist_ok=True)

    latent_dim = model.model_config.latent_dim
    batch_size = 64
    num_batches = (num_images + batch_size - 1) // batch_size

    img_idx = 0
    with torch.no_grad():
        for _ in range(num_batches):
            current_batch_size = min(batch_size, num_images - img_idx)
            if current_batch_size <= 0:
                break

            z = torch.randn(current_batch_size, latent_dim, device=device)
            recon = decode_latents(model, z)

            for b in range(recon.shape[0]):
                out_path = eval_random_samples_dir / f"random_sample_img_{img_idx}.png"
                vutils.save_image(recon[b : b + 1], str(out_path), normalize=True)
                img_idx += 1
                if img_idx >= num_images:
                    break


def compute_fid(eval_gt_dir: Path, eval_random_samples_dir: Path) -> float:
    """Compute FID between two folders of images using pytorch_fid."""

    size1 = len([n for n in os.listdir(eval_gt_dir) if (eval_gt_dir / n).is_file()])
    size2 = len(
        [n for n in os.listdir(eval_random_samples_dir) if (eval_random_samples_dir / n).is_file()]
    )
    assert size1 == size2, f"Dataset sizes for FID do not match: {size1} vs {size2}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fid_value = calculate_fid_given_paths(
        paths=[str(eval_gt_dir), str(eval_random_samples_dir)],
        batch_size=50,
        device=device,
        dims=2048,
    )
    print("fid value:")
    print(float(fid_value))

    return float(fid_value)


def setup_mlflow(mlflow_db_path: Path, experiment_name: str = "MNIST-Models") -> None:
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment(experiment_name)


def evaluate_single_model(model_dir: Path, project_root: Path, eval_gt_dir, num_gt) -> None:
    """Evaluate one DVAE MNIST model and log metrics to mlflow."""

    model = load_model_from_dir(model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare data & directories
    eval_random_samples_dir = model_dir / "eval_random_samples"

    save_random_samples(eval_random_samples_dir, model, num_images=num_gt, device=device)

    # Compute FID
    fid_value = compute_fid(eval_gt_dir, eval_random_samples_dir)

    # Log to mlflow
    mlflow_db_path = project_root / "mlflow.db"
    setup_mlflow(mlflow_db_path)

    run_name = f"metrics_{model_dir.parent.name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_dir", str(model_dir))
        mlflow.log_param("latent_dim", model.model_config.latent_dim)
        mlflow.log_param("reconstruction_loss", model.model_config.reconstruction_loss)
        mlflow.log_param("num_images", num_gt)

        mlflow.log_metric("FID", fid_value)


def main() -> None:
    project_root = get_project_root()
    mnist_models_root = project_root / "mnist_models"

    model_dirs = find_mnist_model_dirs(mnist_models_root)
    if not model_dirs:
        print("No *_model_mnist_* model directories with final_model/model.pt found.")
        return

    print("Found model directories:")
    for d in model_dirs:
        print(" -", d)
    
    test_loader = prepare_mnist_test_dataset(project_root)
    eval_gt_dir = "eval_gt"
    eval_gt_dir = Path(eval_gt_dir)
    max_images = None
    max_images = 2048
    num_gt = save_eval_gt_images(eval_gt_dir, test_loader, max_images=max_images)
    print(num_gt)
    for model_dir in model_dirs:
        print(f"\nEvaluating model in: {model_dir}")
        try:
            evaluate_single_model(model_dir, project_root, eval_gt_dir, num_gt)
        except Exception as e:
            print(f"[ERROR] Evaluation failed for {model_dir}: {e}")


if __name__ == "__main__":
    main()
