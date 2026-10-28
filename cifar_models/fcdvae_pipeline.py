import logging
import pathlib
from pathlib import Path

import mlflow
from dvae_pipeline import getDataset

from pythae.models import FCDVAE, FCDVAEConfig
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

def fcdvaeModels(train_data, eval_data, latent_dim, rec_loss, gauss_app, num_gauss_p, folder_name):
  model_config = FCDVAEConfig(
    input_dim=(3, 32, 32),
    latent_dim=latent_dim,
    reconstruction_loss=rec_loss,
    gaussian_approximation= gauss_app,
    number_gaussian_points=num_gauss_p
  )

  model = FCDVAE(
    model_config=model_config
  )

  training_config_path = absolute_path / 'base_training_config.json'
  training_config = BaseTrainerConfig.from_json_file(str(training_config_path))
  training_config.output_dir = str(absolute_path / 'results' / folder_name)
  
  
  mlflow_db_path = pathlib.Path(f"{absolute_path}/../results/mlflow.db")
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

  pipeline(train_data=train_data, eval_data=eval_data, callbacks=callbacks)

def main():
  train_data, eval_data = getDataset("./data/Cifar10")
  
  latent_dims = [2,3,4,5,6,7,8,16,32,64]
  rec_loss = ["mse"]
  gaus_approxs = ["fib", "lcd"]
  num_gauss_ps = [1,2,3,4,5,6,7,8,9,10]
  models = len(latent_dims) * len(rec_loss) * len(gaus_approxs) * len(num_gauss_ps)
  i = 1
  for ld in latent_dims:
    for ga in gaus_approxs:
      for ngp in num_gauss_ps:
        if(ga != "fib" or ld <= 6):
          print(f"train model: dvae_cifar_{ld}_{rec_loss[0]}_{ga}_{ngp}  | ({i}/{models})")
          try:
            fcdvaeModels(
              train_data=train_data, 
              eval_data=eval_data, 
              latent_dim=ld, 
              rec_loss=rec_loss[0], 
              gauss_app= ga, 
              num_gauss_p=ngp,
              folder_name=f"fcdvae_model_cifar10_{ld}_{rec_loss[0]}_{ga}_{ngp}"
            )
          except Exception as e:
            print(f"Operation failed: {e}")
        i = i+1

if __name__ == "__main__":
  main()