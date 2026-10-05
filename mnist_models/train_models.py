from dvae_pipline import main as dvae
from fcdvae_pipline import main as fcdvae
from vae_pipline import main as vae

print("-----Train dvae-----")
dvae()

print("-----Train fcdvae-----")
fcdvae()

print("-----Train vae-----")
vae()