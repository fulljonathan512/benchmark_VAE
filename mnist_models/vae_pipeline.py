import logging
import pathlib
from pathlib import Path

import mlflow
from dvae_pipeline import getDataset

from pythae.models import VAE, VAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import MLFlowCallback

logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

absolute_path = Path(__file__).resolve().parent

def vaeModels(train_data, eval_data, latent_dim, rec_loss, folder_name):
  model_config = VAEConfig(
    input_dim=(1, 28, 28),
    latent_dim=latent_dim,
    reconstruction_loss=rec_loss,
  )

  model = VAE(
    model_config=model_config
  )

  training_config_path = absolute_path / 'base_training_config.json'
  training_config = BaseTrainerConfig.from_json_file(str(training_config_path))
  training_config.output_dir = str(absolute_path / 'results' / folder_name)
  
  mlflow_db_path = pathlib.Path(f"{absolute_path}/../results/mlflow.db")
  mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
  mlflow.set_experiment("MNIST-Dataset")

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
  train_data, eval_data = getDataset("./data/MNIST")

  latent_dims = [2,3,4,5,6,7,8,16,32,64,128,256]
  rec_loss = ["mse"]
  models = len(latent_dims) * len(rec_loss)
  i = 1
  for ld in latent_dims:
    print(f"train model: vae_mnist_{ld}_{rec_loss[0]}  | ({i}/{models})")
    try:
      vaeModels(
        train_data=train_data, 
        eval_data=eval_data, 
        latent_dim=ld, 
        rec_loss=rec_loss[0], 
        folder_name=f"vae_model_mnist_{ld}_{rec_loss[0]}"
      )
    except Exception as e:
      print(f"Operation failed: {e}")
    i = i+1

if __name__ == "__main__":
  main()