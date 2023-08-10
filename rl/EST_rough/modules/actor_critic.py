import numpy as np
from termcolor import cprint

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class DmEncoder(nn.Module):
    def __init__(self, num_encoder_obs, encoder_hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_encoder_obs, encoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2]),
            nn.Tanh(),
        )

    def forward(self, dm):
        """
        Encodes depth map
        Input:
            dm: a depth map usually shape (187)
        """
        return self.encoder(dm)





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
        self.priv_mlp = kwargs['priv_mlp_units']
        self.priv_info = kwargs['priv_info']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.priv_info_stage2 = kwargs['proprio_adapt']

        self.proprio_adapt_input = num_obs
        self.proprio_adapt_output = kwargs['proprio_adapt_out_dim']

        self.velLen = kwargs['velLen']

        num_actor_input = num_obs + self.proprio_adapt_output

        num_critic_input = num_obs  + self.priv_info_dim  ##### 45 + 3 + 187

        self.dm_encoder = DmEncoder(num_obs, self.priv_mlp)
        cprint(f"Encoder MLP: {self.dm_encoder}", 'green', attrs=['bold'])


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
        mean, std, _, e, e_gt = self._actor_critic(obs_dict)

        self.distribution = Normal(mean, mean * 0. + std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        # actions_mean = self.actor(observations)
        # used for testing
        actions_mean, _, _, _, _ = self._actor_critic(obs_dict)
        return actions_mean

    def evaluate(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, value, extrin, extrin_gt = self._actor_critic(obs_dict)
        return value

    def extrin_loss(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, _, extrin, _ = self._actor_critic(obs_dict)
        return extrin

    def extrin_gt_loss(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, _, _, extrin_gt = self._actor_critic(obs_dict)

        return extrin_gt
    def _actor_critic(self, obs_dict):

        obs = obs_dict['obs']
        obs_vel = obs_dict['privileged_info'][:, 0:3]
        obs_hight = obs_dict['privileged_info'][:, 3:200]
        # obs_contact = obs_dict['priv_vel_info'][:, 7:11]

        extrin_gt = obs_dict['privileged_info'][:, 0:11]

        extrin_en = self.dm_encoder(obs)



        actor_obs = torch.cat([obs, extrin_en], dim=-1)  ## 45 + 11
        critic_obs = torch.cat([obs_vel, obs, obs_hight], dim=-1)  ## 45+3+187 = 235
        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std


        extrin_gt = torch.tanh(extrin_gt)

        return mu, mu * 0 + sigma, value, extrin_en, extrin_gt


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CELU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


