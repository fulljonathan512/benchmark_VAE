import logging
import pathlib
from pathlib import Path

import mlflow
from dvae_pipeline import getDataset

from pythae.models import VAE, VAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import MLFlowCallback

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

absolute_path = Path(__file__).resolve().parent

def vaeModels(train_data, eval_data, latent_dim, rec_loss, folder_name):
  model_config = VAEConfig(
    input_dim=(3, 32, 32),
    latent_dim=latent_dim,
    reconstruction_loss=rec_loss,
  )

  model = VAE(
    model_config=model_config
  )

  training_config = BaseTrainerConfig.from_json_file('base_training_config.json')
  training_config.output_dir = folder_name

  mlflow_db_path = pathlib.Path(f"{absolute_path}/../mlflow.db")
  mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
  mlflow.set_experiment("CIFAR10-Dataset")


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

  
def main():
  train_data, eval_data = getDataset("./data/Cifar10")

  latent_dims = [2,4]#[2,4,16,32,256]
  rec_loss = ["mse"]
  for ld in latent_dims:
    vaeModels(
      train_data=train_data, 
      eval_data=eval_data, 
      latent_dim=ld, 
      rec_loss=rec_loss[0], 
      folder_name=f"vae_model_cifar10_{ld}_{rec_loss[0]}"
    )

if __name__ == "__main__":
  main()