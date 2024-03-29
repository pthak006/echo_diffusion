import torch

from echo import echo_loss, echo_sample

# Set the random seed for reproducibility
torch.manual_seed(42)

# Define the dimensions
batch_size = 100
z_dim = 50

# Generate mock values for fx (now with shape [batch_size, 1, z_dim, z_dim])
fx = torch.randn(batch_size, 1, z_dim, z_dim)

# Generate mock values for Sx (also with shape [batch_size, 1, z_dim, z_dim])
Sx_diag = torch.rand(batch_size, 1, z_dim, z_dim) * 0.9  # Ensure spectral radius < 1
Sx = torch.diag_embed(Sx_diag)

print(f"Shape of fx: {fx.shape}")
print(f"Shape of Sx: {Sx.shape}")

# Test echo_sample function
echo_output = echo_sample([fx, Sx], d_max=99, batch_size=batch_size)
print("Echo Sample Output:")
print(echo_output)
print("Echo Sample Output Shape:", echo_output.shape)

# Test echo_loss function
echo_loss_output = echo_loss([fx, Sx])
print("\nEcho Loss Output:")
print(echo_loss_output)
print("Echo Loss Output Shape:", echo_loss_output.shape)
