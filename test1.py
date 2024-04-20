# Import necessary PyTorch libraries
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

# Set the device to MPS if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Additional libraries for visualization and utilities
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import Compose, Grayscale, Normalize, ToTensor

# Import the adapted Echo noise functions
from echo import echo_loss, echo_sample

# Define transformations: Convert to grayscale, resize if needed, and normalize the data
transform = Compose(
    [
        Grayscale(num_output_channels=1),  # Convert to grayscale
        ToTensor(),
        Normalize((0.5,), (0.5,)),  # Normalize with single channel
    ]
)

# Load the CIFAR-10 dataset
dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Splitting dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

print("Data loaders created for training and validation.")

from segmentation_models_pytorch import Unet


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dims = latent_dims

        self.unet = Unet(
            encoder_name="resnet18",
            encoder_depth=3,
            encoder_weights=None,
            decoder_use_batchnorm=True,
            decoder_channels=(latent_dims[2], latent_dims[1], latent_dims[0]),
            decoder_attention_type=None,
            in_channels=input_shape[0],
            classes=latent_dims[-1],
            activation=None,
        )

        # Output layers
        self.out_mean = nn.Conv2d(latent_dims[-1], input_shape[0], kernel_size=1)
        self.out_log_var = nn.Conv2d(latent_dims[-1], input_shape[0], kernel_size=1)

    def forward(self, x):
        x = self.unet(x)
        f_x = torch.tanh(self.out_mean(x))
        log_var = torch.sigmoid(self.out_log_var(x))
        return f_x, log_var


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Decoder(nn.Module):
    def __init__(self, latent_dims, output_shape, timestep_dim=128):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.output_shape = output_shape
        self.timestep_dim = timestep_dim

        self.conv1 = nn.Conv2d(1, latent_dims[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            latent_dims[0], latent_dims[1], kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            latent_dims[1], latent_dims[2], kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            latent_dims[2], latent_dims[3], kernel_size=3, stride=2, padding=1
        )
        self.conv5 = nn.Conv2d(
            latent_dims[3], latent_dims[4], kernel_size=3, stride=1, padding=1
        )

        self.timestep_mlp = nn.Sequential(
            nn.Linear(timestep_dim, latent_dims[4]),
            nn.ReLU(),
            nn.Linear(latent_dims[4], latent_dims[4]),
        )

        self.deconv1 = nn.ConvTranspose2d(
            latent_dims[4] * 2, latent_dims[3], kernel_size=3, stride=1, padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            latent_dims[3], latent_dims[2], kernel_size=4, stride=2, padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            latent_dims[2], latent_dims[1], kernel_size=3, stride=1, padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            latent_dims[1], latent_dims[0], kernel_size=4, stride=2, padding=1
        )
        self.deconv5 = nn.ConvTranspose2d(
            latent_dims[0], output_shape[0], kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, t):
        timestep_emb = get_timestep_embedding(t, self.timestep_dim)
        timestep_emb = self.timestep_mlp(timestep_emb)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        x = torch.cat(
            [x, timestep_emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])],
            dim=1,
        )

        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))

        return x


class EchoModel(nn.Module):
    def __init__(self, input_shape, latent_dims, output_shape, T=1000, batch_size=100):
        super(EchoModel, self).__init__()
        self.input_shape = input_shape
        self.latent_dims = latent_dims
        self.output_shape = output_shape
        self.T = T
        self.batch_size = batch_size

        self.encoder = Encoder(input_shape, latent_dims)
        self.decoder = Decoder(latent_dims, output_shape)

        # Define the noise schedule
        self.alpha = self.create_noise_schedule(T)

    def create_noise_schedule(self, T):
        alpha = torch.linspace(0.9999, 1e-5, T)
        return alpha

    def forward(self, x):
        f_x, sx_matrix = self.encoder(x)
        print(f"First element of f_x: {f_x[0]}")
        print(f"First element of s_x: {sx_matrix[0]}")
        # print(f"Shape of fx: {f_x.shape}")
        # print(f"Shape of Sx: {sx_matrix.shape}")
        # print(f"Shape of input: {x.shape}")
        # assert(False)

        # Convert log-variance to diagonal elements of S(x)
        # diagonal_sx = torch.exp(log_var)

        # Create the full square matrix representation of S(x)
        # sx_matrix = torch.diag_embed(diagonal_sx)

        # Generate the noise variable z using echo_sample
        z = echo_sample((f_x, sx_matrix))
        print(f"First element of z: {z[0]}")
        assert False
        # print(f"Shape of output: {z.shape}")
        torch.cuda.empty_cache()

        # print(f"Shape of x: {x.shape}")
        # print(f"Shape of z: {z.shape}")
        # alpha_T = self.alpha[-1]
        # sqrt_alpha_T = torch.sqrt(alpha_T)
        # sqrt_one_minus_alpha_T = torch.sqrt(1 - alpha_T)
        # x_T = sqrt_alpha_T * x + sqrt_one_minus_alpha_T * z
        alpha_T = self.alpha[-1]
        sqrt_alpha_T = torch.sqrt(alpha_T)
        sqrt_one_minus_alpha_T = torch.sqrt(1 - alpha_T)

        # Reshape z for broadcasting: [100, 28] -> [100, 1, 28, 1]
        # z_reshaped = z.unsqueeze(1).unsqueeze(-1)

        # Repeat z across the spatial dimensions to match x's shape: [100, 1, 28, 1] -> [100, 1, 28, 28]
        # z_broadcasted = z_reshaped.expand(-1, 1, x.shape[2], x.shape[3])

        # Perform the weighted sum
        x_T = sqrt_alpha_T * x + sqrt_one_minus_alpha_T * z

        # Perform the reconstruction process using Algorithm 2
        reconstructed_x = self.reconstruct(x_T, z, f_x, sx_matrix)
        return reconstructed_x, f_x, sx_matrix

    def reconstruct(self, x_t, z, f_x, sx_matrix):
        x_s = x_t
        for s in range(self.T - 1, 0, -1):
            t = torch.tensor([s] * x_t.size(0), dtype=torch.long).to(x_t.device)
            sqrt_alpha_s = torch.sqrt(self.alpha[s])
            sqrt_one_minus_alpha_s = torch.sqrt(1 - self.alpha[s])

            # Estimate the original image using the decoder
            x_0_hat = self.decoder(x_s, t)

            # Calculate the estimated noise using Eq. (3)
            z_hat = (x_s - sqrt_alpha_s * x_0_hat) / sqrt_one_minus_alpha_s

            # Calculate D(x_0_hat, s) and D(x_0_hat, s-1) using Eq. (5) and (6)
            D_x_0_hat_s = sqrt_alpha_s * x_0_hat + sqrt_one_minus_alpha_s * z_hat
            D_x_0_hat_s_minus_1 = (
                torch.sqrt(self.alpha[s - 1]) * x_0_hat
                + torch.sqrt(1 - self.alpha[s - 1]) * z_hat
            )

            # Update x_s using Eq. (7)
            x_s = x_s - D_x_0_hat_s + D_x_0_hat_s_minus_1

        return x_s


import time  # Importing time to log the duration


def debug_loop(
    model,
    optimizer,
    train_loader,
    device,
    num_epochs,
    loss_weights,
    accumulation_steps=2,
):
    model.eval()  # Set the model to evaluation mode

    # Get a single batch from the train loader
    batch_data, _ = next(iter(train_loader))
    batch_data = batch_data.to(device)

    # Call the model on the batch data
    with torch.no_grad():  # Disable gradient calculation
        reconstructed_x, f_x, sx_matrix = model(batch_data)

    # Print the shapes of the model outputs
    print(f"Shape of reconstructed_x: {reconstructed_x.shape}")
    print(f"Shape of f_x: {f_x.shape}")
    print(f"Shape of sx_matrix: {sx_matrix.shape}")

    return model


# Define the input shape, latent dimensions, and output shape
input_shape = (1, 32, 32)  # Shape for grayscale CIFAR-10 (1 channel, 32x32 images)
latent_dims = [32, 64, 128, 256, 512]  # Updated latent dimensions
output_shape = (1, 32, 32)  # Shape for grayscale CIFAR-10

# Create an instance of the EchoModel
model = EchoModel(input_shape, latent_dims, output_shape).to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define the number of epochs and loss weights
num_epochs = 100
loss_weights = {
    "reconstruction": 1.0,
    "mi_penalty": 0.0,
}  # Adjust the weights as needed

# Train the model
trained_model = debug_loop(
    model, optimizer, train_loader, device, num_epochs, loss_weights
)