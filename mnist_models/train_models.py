from dvae_pipeline import main as dvae
from fcdvae_pipeline import main as fcdvae
from vae_pipeline import main as vae

print("-----Train dvae-----")
dvae()

print("-----Train fcdvae-----")
fcdvae()

print("-----Train vae-----")
vae()