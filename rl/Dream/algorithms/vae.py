import torch
import torch.nn as torch_nn
from torch.nn import functional as F

import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(225, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, 19)
        self.fc32 = nn.Linear(64, 19)



        self.fc3 = nn.Linear(19, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 48)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h31 = F.relu(self.fc31(h2))
        h32 = F.relu(self.fc32(h2))
        return h31, h32

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        return h5

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 225))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 225), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def loss(self, x, y, mu, log_var, return_losses=False):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        recon_loss = torch.sum(torch.mean(torch.square(x - y), dim=0))
        loss = recon_loss + self.kl_lambda * kl_loss
        if return_losses:
            return loss, recon_loss, self.kl_lambda * kl_loss
        else:
            return loss

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples