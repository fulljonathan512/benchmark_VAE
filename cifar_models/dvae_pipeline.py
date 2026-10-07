import logging
import pathlib
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

def dvaeModels(train_data, eval_data, latent_dim, rec_loss, gauss_app, num_gauss_p, folder_name):
  model_config = DVAEConfig(
    input_dim=(3, 32, 32),
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

  callbacks = [] # the TrainingPipeline expects a list of callbacks
  mlflow_cb = MLFlowCallback() # Build the callback 
  # SetUp the callback 
  mlflow_cb.setup(
  training_config=training_config, # training config
  model_config=model_config, # model config
  run_name=folder_name # specify your mlflow run
  )
  callbacks.append(mlflow_cb) # Add it to the callbacks list

  pipeline = TrainingPipeline(training_config=training_config, model=model)

  pipeline(train_data=train_data, eval_data=eval_data,callbacks=callbacks)

def getDataset(folderpath):
  if(not pathlib.Path(f"{folderpath}/train_data.npz").exists()):
    logger.info("Download train_data")
    train_set = datasets.CIFAR10(
      root="./data/Cifar10",
      train=True,
      download=True,
    )
        
    train_datas = train_set.data
    train_labels = np.array(train_set.targets)
    np.savez(f"{folderpath}/train_data.npz", data=train_datas, label=train_labels)
    

  if(not pathlib.Path(f"{folderpath}/eval_data.npz").exists()):
    logger.info("Download eval_data")
    eval_set = datasets.CIFAR10(
      root="./data/Cifar10",
      train=False,
      download=True,
    )
    
    eval_datas = eval_set.data
    eval_labels = np.array(eval_set.targets)
    np.savez(f"{folderpath}/eval_data.npz", data=eval_datas, label=eval_labels)

  train_data = (
        np.load(f"{folderpath}/train_data.npz")["data"]
        / 255.0
    )
  eval_data = (
      np.load(f"{folderpath}/eval_data.npz")["data"]
      / 255.0
  )

  train_data = np.resize(train_data, (60,32,32,3))
  eval_data = np.resize(eval_data, (6,32,32,3))

  return train_data, eval_data
  
def main():
  train_data, eval_data = getDataset("./data/Cifar10")

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
            folder_name=f"dvae_model_cifar10_{ld}_{rec_loss[0]}_{ga}_{ngp}"
          )

if __name__ == "__main__":
  main()