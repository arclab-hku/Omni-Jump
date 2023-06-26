import numpy as np
from termcolor import cprint

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn import functional as F



class CeEncoder(nn.Module):
    def __init__(self, num_encoder_obs, encoder_hidden_dims, decoder_hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_encoder_obs, encoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2])
        self.fc_logvar = nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2])

        self.decoder = nn.Sequential(
            nn.Linear(encoder_hidden_dims[2], decoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dims[0], decoder_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dims[1], decoder_hidden_dims[2]),
            nn.Sigmoid()
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def forward(self, x):
        mean, logvar = self.encode(x)
        z_feature = self.reparameterize(mean, logvar)
        next_obs_feature = self.decode(z_feature)
        return next_obs_feature, z_feature, mean, logvar
    def loss_function1(self, x, y, mu, log_var):
        recon_loss = torch.sum(torch.mean(torch.square(x - y), dim=0))


        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recon_loss + self.kl_lambda * kl_loss

        return loss
    def loss_function(self, mu, log_var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        return kl_loss

class CeEncoder1(nn.Module):
    def __init__(self):
        super(CeEncoder1, self).__init__()

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
        return self.decode(z), z, mu, logvar

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

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()



        # ---- Priv Info ----
        # self.priv_mlp = kwargs['priv_mlp_units']

        self.encoder_mlp = kwargs['priv_mlp_units']
        self.decoder_mlp = kwargs['decoder_mlp_units']

        self.priv_info = kwargs['priv_info']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.priv_info_stage2 = kwargs['proprio_adapt']

        self.proprio_adapt_input = num_obs
        self.proprio_adapt_output = kwargs['proprio_adapt_out_dim']



        self.HistoryLen = kwargs['HistoryLen']
        self.velLen = kwargs['velLen']

        num_encoder_input = num_obs * self.HistoryLen

        num_decoder_input = num_obs + self.proprio_adapt_output


        num_actor_input = num_obs + self.proprio_adapt_output

        num_critic_input = num_obs + self.velLen + self.priv_info_dim  #####

        self.encoder_decoder = CeEncoder(num_encoder_input, self.encoder_mlp, self.decoder_mlp)

        cprint(f"Encoder MLP: {self.encoder_decoder}", 'green', attrs=['bold'])

        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_input, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_input, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)



    def act(self, obs_dict, **kwargs):
        # self.update_distribution(observations)
        mean, std, _, e, e_gt,  _, _  = self._actor_critic(obs_dict)

        self.distribution = Normal(mean, mean * 0. + std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        # actions_mean = self.actor(observations)
        # used for testing
        actions_mean, _, _, _, _ , _, _ = self._actor_critic(obs_dict)
        return actions_mean

    def evaluate(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, value, extrin, extrin_gt,  _, _  = self._actor_critic(obs_dict)
        return value

    def extrin_loss(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, _, extrin, _ , _, _ = self._actor_critic(obs_dict)
        return extrin

    def extrin_gt_loss(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, _, _, extrin_gt, _, _ = self._actor_critic(obs_dict)

        return extrin_gt


    def encoder_loss(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, _, _, _,mu, logvar = self._actor_critic(obs_dict)

        return mu, logvar
    def _actor_critic(self, obs_dict):

        obs = obs_dict['obs']
        obs_vel = obs_dict['priv_vel_info'][:, 0:3]
        obs_hight = obs_dict['priv_vel_info'][:, 11:198]
        obs_his = obs_dict['proprio_hist']
        obs_his = obs_his.flatten(1)


        next_obs_feature, z_feature,  mu_z, logvar_z = self.encoder_decoder(obs_his)
        kl_loss = self.encoder_decoder.loss_function(mu_z, logvar_z)
        extrin = torch.tanh(z_feature)

        extrin_gt = torch.tanh(next_obs_feature)

        actor_obs = torch.cat([obs, z_feature], dim=-1)  #### 45 + 19 = 64

        critic_obs = torch.cat([obs_vel, obs, obs_hight], dim=-1)  ## 45+3+187 = 235


        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std


        return mu, mu * 0 + sigma, value, extrin, extrin_gt, kl_loss, logvar_z


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


