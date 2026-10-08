import logging
import pathlib
from pathlib import Path

import mlflow
import numpy as np
from torchvision import datasets

from pythae.models import DVAE, DVAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import MLFlowCallback

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

absolute_path = Path(__file__).resolve().parent

def dvaeModels(train_data, eval_data, latent_dim, rec_loss, gauss_app, num_gauss_p, folder_name):
  model_config = DVAEConfig(
    input_dim=(218, 178, 3),
    latent_dim=latent_dim,
    reconstruction_loss=rec_loss,
    gaussian_approximation= gauss_app,
    number_gaussian_points=num_gauss_p
  )

  model = DVAE(
    model_config=model_config
  )

  training_config = BaseTrainerConfig.from_json_file('base_training_config.json')
  training_config.output_dir = folder_name

  mlflow_db_path = pathlib.Path(f"{absolute_path}/../mlflow.db")
  mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
  mlflow.set_experiment("CELEBA-Dataset")

  callbacks = []
  mlflow_cb = MLFlowCallback()
  mlflow_cb.setup(
    training_config=training_config,
    model_config=model_config,
    run_name=folder_name
  )
  callbacks.append(mlflow_cb)

  pipeline = TrainingPipeline(training_config=training_config, model=model)

  pipeline(train_data=train_data, eval_data=eval_data,callbacks=callbacks)

def getDataset(folderpath):
  if(not Path(f"{absolute_path}/{folderpath}/train_data.npz").exists()):
    logger.info("Download train_data")
    train_set = datasets.CelebA(
      root="./data",
      split="train",
      download=True,
    )

    train_datas = train_set.data
    train_labels= np.array(train_set.targets)
    np.savez(f"{absolute_path}/{folderpath}/train_data.npz", data=train_datas, label=train_labels)

  if(not Path(f"{absolute_path}/{folderpath}/eval_data.npz").exists()):
    logger.info("Download eval_data")
    eval_set = datasets.CelebA(
      root="./data",
      split="valid",
      download=True,
    )

    eval_datas = eval_set.data
    eval_labels= np.array(eval_set.targets)
    np.savez(f"{absolute_path}/{folderpath}/eval_data.npz", data=eval_datas, label=eval_labels)

  train_data = (
        np.load(f"{absolute_path}/{folderpath}/train_data.npz")["data"]
        / 255.0
    )
  eval_data = (
      np.load(f"{absolute_path}/{folderpath}/eval_data.npz")["data"]
      / 255.0
  )

  train_data = np.resize(train_data, (60, 218, 178, 3))
  eval_data = np.resize(eval_data, (6, 218, 178, 3))

  return train_data, eval_data
  
def main():
  train_data, eval_data = getDataset("./data/CelebA")

  print("Download succesful")
  return 0

  latent_dims = [2,4]#[2,4,16,32,256]
  rec_loss = ["mse"]
  gaus_approxs = ["lcd"] #["fib", "lcd"]
  num_gauss_ps = [2] #[2,3,4,5]
  for ld in latent_dims:
    for ga in gaus_approxs:
      for ngp in num_gauss_ps:
        if(ga != "fib" or ld <= 6):
          dvaeModels(
            train_data=train_data, 
            eval_data=eval_data, 
            latent_dim=ld, 
            rec_loss=rec_loss[0], 
            gauss_app= ga, 
            num_gauss_p=ngp,
            folder_name=f"dvae_model_mnist_{ld}_{rec_loss[0]}_{ga}_{ngp}"
          )

if __name__ == "__main__":
  main()