import numpy as np

from pythae.models import FCDVAE, FCDVAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig


def fcdvaeModels(train_data, eval_data, latent_dim, rec_loss, gauss_app, num_gauss_p, folder_name):
  model_config = FCDVAEConfig(
    input_dim=(1, 28, 28),
    latent_dim=latent_dim,
    reconstruction_loss=rec_loss,
    gaussian_approximation= gauss_app,
    number_gaussian_points=num_gauss_p
  )

  model = FCDVAE(
    model_config=model_config
  )

  training_config = BaseTrainerConfig.from_json_file('base_training_config.json')
  training_config.output_dir = folder_name

  pipeline = TrainingPipeline(training_config=training_config, model=model)

  pipeline(train_data=train_data, eval_data=eval_data)

def main():
  train_data = (
      np.load('../examples/scripts/data/mnist/train_data.npz')["data"]
      / 255.0
  )
  eval_data = (
      np.load('../examples/scripts/data/mnist/eval_data.npz')["data"]
      / 255.0
  )
  latent_dims = [2,4,16,32,256]
  rec_loss = ["mse"]
  gaus_approxs = ["fib", "lcd"]
  num_gauss_ps = [2,3,4,5]
  for ld in latent_dims:
    for ga in gaus_approxs:
      for ngp in num_gauss_ps:
        if(ga != "fib" or ld <= 6):
          fcdvaeModels(
            train_data=train_data, 
            eval_data=eval_data, 
            latent_dim=ld, 
            rec_loss=rec_loss[0], 
            gauss_app= ga, 
            num_gauss_p=ngp,
            folder_name=f"fcdvae_model_mnist_{ld}_{rec_loss[0]}_{ga}_{ngp}"
          )

if __name__ == "__main__":
  main()