import torch

from examples.pl_basics.hl_forge_modules.ex_02_autoencoder.autoencoder import AutoEncoderModule, Decoder, Encoder
from hlforge.core.module import CompositeModel


def test_autoencoder_module():
    encoder = Encoder(hidden_dim=32, latent_dim=4)
    decoder = Decoder(hidden_dim=32, latent_dim=4)
    model = CompositeModel(backbone=encoder, head=decoder)

    # AutoEncoderModule is empty, inherits BaseLightningModule
    module = AutoEncoderModule(model=model, criterion=torch.nn.MSELoss())

    x = torch.randn(4, 1, 28, 28)
    # The model is CompositeModel(backbone=Encoder, head=Decoder)
    # Encoder flattens, Decoder reconstructs.
    # Forward pass should be: x -> Flatten+Encoder -> Latent -> Decoder -> Rec
    out = module(x)
    assert out.shape == (4, 784)
