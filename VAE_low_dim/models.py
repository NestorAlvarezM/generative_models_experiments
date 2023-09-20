import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input if needed
        encoded = self.encoder(x)
        mean, log_var = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, output_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_var

# Example usage:
input_dim = 784  # Example MNIST data
hidden_dims = [256, 128]  # List of hidden layer dimensions
latent_dim = 20

vae = VAE(input_dim, hidden_dims, latent_dim)
input_data = torch.randn(32, input_dim)  # Replace with your input data
reconstructed_data, mean, log_var = vae(input_data)
print(reconstructed_data.shape)

encoder = Encoder(input_dim, hidden_dims, latent_dim)
encoder = torch.randn(32, input_dim)  # Replace with your input data
encoder_output= vae(input_data)
print(reconstructed_data.shape)