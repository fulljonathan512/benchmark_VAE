"""Compute FID score for the trained MNIST DVAE model and log it to mlflow.

This script is designed to be *methodologically compatible* with the
`compute_fid_on_datasets` routine used in the other project
"unscented-autoencoder".

Key points for compatibility:
- Uses ``pytorch_fid.fid_score.calculate_fid_given_paths`` with the same
  arguments (``batch_size=50``, ``dims=2048``, CUDA if available).
- Operates on two directories of PNG images (``eval_gt`` and
  ``eval_random_samples``) similarly to the unscented-autoencoder project.

Differences to be aware of:
- Here, random samples are drawn directly from the DVAE prior instead of
  via a Gaussian Mixture Model fitted on latent features. This keeps the
  pipeline simple while still using the same FID implementation.

Usage (from within the benchmark_VAE repo root or any directory)::

    python "mnist_models/dvae_model_mnist_2_mse_lcd_2/compute_fid_dvae_mnist.py"

Requirements:
- ``torch``, ``torchvision``, ``pytorch_fid``, ``mlflow``, ``pythae``
- The trained model files in::

      mnist_models/dvae_model_mnist_2_mse_lcd_2/VAE_training_2026-10-08_12-12-50/final_model

The script will:
1. Load ``model_config.json``, ``training_config.json``, and ``model.pt``.
2. Build a DVAE instance matching the training configuration.
3. Create two image folders::

       .../eval_gt
       .../eval_random_samples

   and fill them with MNIST test images and DVAE-generated samples.
4. Compute FID between those folders.
5. Log FID and some metadata into the existing ``mlflow.db`` database.
"""

import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

# ---------------------------------------------------------------------------
# Compatibility patch for SciPy / pytorch_fid
#
# Newer SciPy versions changed the signature of scipy.linalg.sqrtm and removed
# the ``disp`` keyword argument. pytorch_fid still calls ``sqrtm(..., disp=False)``.
# This wrapper makes sqrtm accept ``disp`` and ignores it if necessary.
# ---------------------------------------------------------------------------
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

from pytorch_fid.fid_score import calculate_fid_given_paths
import mlflow

from pythae.models import DVAE, DVAEConfig


def get_paths() -> dict:
    """Return important absolute paths used in this script.

    The script is placed in:
        benchmark_VAE/mnist_models/dvae_model_mnist_2_mse_lcd_2/
    and the trained model is in the subfolder:
        VAE_training_2026-10-08_12-12-50/final_model
    """

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # .../benchmark_VAE

    model_dir = (
        project_root
        / "mnist_models"
        / "dvae_model_mnist_2_mse_lcd_2"
        / "VAE_training_2026-10-08_12-12-50"
        / "final_model"
    )

    mlflow_db_path = project_root / "mlflow.db"

    # Output directories for images used in FID computation
    eval_gt_dir = model_dir / "eval_gt"
    eval_random_samples_dir = model_dir / "eval_random_samples"

    return {
        "project_root": project_root,
        "model_dir": model_dir,
        "mlflow_db_path": mlflow_db_path,
        "eval_gt_dir": eval_gt_dir,
        "eval_random_samples_dir": eval_random_samples_dir,
    }


def load_dvae(model_dir: Path) -> DVAE:
    """Load the DVAE model from the given `final_model` directory.

    The directory is expected to contain:
    - model_config.json  (DVAEConfig)
    - model.pt           (state_dict of DVAE)
    """

    config_path = model_dir / "model_config.json"
    model_path = model_dir / "model.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model_config.json at {config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model.pt at {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # Remove the "name" field if present; DVAEConfig does not need it.
    cfg_dict = {k: v for k, v in cfg_dict.items() if k != "name"}

    dvae_config = DVAEConfig(**cfg_dict)
    model = DVAE(model_config=dvae_config)

    state_dict = torch.load(model_path, map_location="cpu")
    # In case the checkpoint was saved with extra keys, try to extract
    # a pure state_dict.
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    return model


def prepare_mnist_test_dataset(project_root: Path):
    """Return a DataLoader over the MNIST test set.

    Images are returned as tensors in [0,1] with shape [1, 28, 28].
    """

    data_root = project_root / "data" / "MNIST"
    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    # Use batch_size=1 so saved images closely match the unscented-autoencoder
    # code which saved one image per file.
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    return test_loader


def save_eval_gt_images(eval_gt_dir: Path, test_loader: DataLoader, max_images: int | None = None):
    """Save MNIST test images as ground truth PNGs for FID computation."""

    eval_gt_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, (img, _) in enumerate(test_loader):
        if max_images is not None and count >= max_images:
            break

def save_random_samples(
    eval_random_samples_dir: Path, model: DVAE, num_images: int, device: torch.device
):
    """Sample from the DVAE prior and save as PNGs for FID computation.

    For compatibility with the unscented-autoencoder pipeline we:
    - generate exactly ``num_images`` samples,
    - save them with ``torchvision.utils.save_image(..., normalize=True)``.
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

            # Sample from standard normal prior
            z = torch.randn(current_batch_size, latent_dim, device=device)

            # Use Pythae's decode method. Depending on the implementation this
            # may return
            #   - a tensor,
            #   - a dict like {"reconstruction": tensor, ...}, or
            #   - a tuple whose first element is the reconstruction tensor.
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
                    "DVAE.decode(z) must return a tensor, dict, or tuple, "
                    f"got {type(out)}"
                )

            # recon: [B, C, H, W]
            for b in range(recon.shape[0]):
                if img_idx >= num_images:
                    break
                out_path = eval_random_samples_dir / f"random_sample_img_{img_idx}.png"
                vutils.save_image(recon[b : b + 1], str(out_path), normalize=True)
                img_idx += 1
                if img_idx >= num_images:
                    break


def compute_fid(eval_gt_dir: Path, eval_random_samples_dir: Path) -> float:
    """Compute FID between two folders of images using pytorch_fid.

    This mirrors `VAEXperiment.compute_fid_on_datasets` from the
    unscented-autoencoder project.
    """

    # Basic sanity check: same number of files in both dirs
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

    return float(fid_value)


def setup_mlflow(mlflow_db_path: Path, experiment_name: str = "MNIST-DVAE") -> None:
    """Configure mlflow to use the existing SQLite tracking DB."""

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment(experiment_name)


def main() -> None:
    paths = get_paths()

    project_root: Path = paths["project_root"]
    model_dir: Path = paths["model_dir"]
    mlflow_db_path: Path = paths["mlflow_db_path"]
    eval_gt_dir: Path = paths["eval_gt_dir"]
    eval_random_samples_dir: Path = paths["eval_random_samples_dir"]

    print(f"Project root: {project_root}")
    print(f"Model dir:   {model_dir}")

    # Load model
    model = load_dvae(model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare data
    test_loader = prepare_mnist_test_dataset(project_root)

    # To keep runtime reasonable we can limit the number of images used for FID.
    # Setting this to None will use the full test set.
    max_images = None
    max_images = 2048

    # Save ground-truth and generated images
    print("Saving eval_gt images...")
    save_eval_gt_images(eval_gt_dir, test_loader, max_images=max_images)

    num_images = len([n for n in os.listdir(eval_gt_dir) if (eval_gt_dir / n).is_file()])
    print(f"Number of eval_gt images: {num_images}")

    print("Saving random samples from DVAE prior...")
    save_random_samples(eval_random_samples_dir, model, num_images=num_images, device=device)

    # Compute FID
    print("Computing FID...")
    fid_value = compute_fid(eval_gt_dir, eval_random_samples_dir)
    print(f"FID (MNIST DVAE): {fid_value}")

    # Log to mlflow
    try:
        setup_mlflow(mlflow_db_path)

        run_name = "FID_dvae_model_mnist_2_mse_lcd_2"
        with mlflow.start_run(run_name=run_name):
            # Log basic parameters about the model and evaluation
            mlflow.log_param("model_dir", str(model_dir))
            mlflow.log_param("latent_dim", model.model_config.latent_dim)
            mlflow.log_param("reconstruction_loss", model.model_config.reconstruction_loss)
            mlflow.log_param("gaussian_approximation", model.model_config.gaussian_approximation)
            mlflow.log_param("number_gaussian_points", model.model_config.number_gaussian_points)

            mlflow.log_param("num_images", num_images)

            mlflow.log_metric("FID", fid_value)

            # Optionally log the directories as artifacts (this can be large).
            mlflow.log_artifacts(str(eval_gt_dir), artifact_path="eval_gt")
            mlflow.log_artifacts(str(eval_random_samples_dir), artifact_path="eval_random_samples")

    except Exception as e:
        # Do not break the pipeline if MLflow logging fails
        print(f"[MLflow] Logging failed: {e}")


if __name__ == "__main__":
    main()
