import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, stride=2) # 36 -> 17
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2) # 17 -> 8
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=1) # 8 -> 4
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(4*4*32, latent_dim)
        self.linear_mu = nn.Linear(latent_dim, latent_dim)
        self.linear_sigma = nn.Linear(latent_dim, latent_dim)

        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.linear1(self.flatten(x)))

        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))

        z = mu + sigma*self.N.sample(mu.shape)

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(latent_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim, 4*4*32)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 4, 4))

        self.deconv1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1) # 4->8
        self.deconv2 = nn.ConvTranspose2d(16, 8, 3, stride=2) # 8->17
        self.deconv3 = nn.ConvTranspose2d(8, 1, 3, stride=2, output_padding=1) # 17 -> 36

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.batch_norm3 = nn.BatchNorm2d(8)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.relu(self.batch_norm1(self.unflatten(x)))
        x = F.relu(self.batch_norm2(self.deconv1(x)))
        x = F.relu(self.batch_norm3(self.deconv2(x)))
        x = F.relu(self.deconv3(x))

        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
