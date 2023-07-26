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
        )

    def forward(self, dm):
        """
        Encodes depth map
        Input:
            dm: a depth map usually shape (187)
        """
        return self.encoder(dm)


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProprioAdaptTConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, output_size)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


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

        mlp_input_shape = num_obs

        # ---- Priv Info ----
        self.priv_mlp = kwargs['priv_mlp_units']
        self.priv_info = kwargs['priv_info']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.priv_info_stage2 = kwargs['proprio_adapt']

        self.proprio_adapt_input = num_obs
        self.proprio_adapt_output = kwargs['proprio_adapt_out_dim']
        mlp_input_shape += self.proprio_adapt_output

        if self.priv_info:
            self.dm_encoder = DmEncoder(num_obs, self.priv_mlp)
            cprint(f"Encoder MLP: {self.dm_encoder}", 'green', attrs=['bold'])

            if self.priv_info_stage2:
                self.adapt_tconv = ProprioAdaptTConv(input_size=self.proprio_adapt_input,
                                                     output_size=self.proprio_adapt_output)
                cprint(f"Adaptation Conv: {self.adapt_tconv}", 'green', attrs=['bold'])
        else:
            cprint('Vanilla Actor Critic', 'green', attrs=['bold'])

        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_shape, actor_hidden_dims[0]))
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
        critic_layers.append(nn.Linear(mlp_input_shape, critic_hidden_dims[0]))
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


        extrin_gt = obs_dict['privileged_info'][:, 0:11]


        extrin_en = self.dm_encoder(obs)


        extrin = torch.tanh(extrin_en)
        extrin_gt = torch.tanh(extrin_gt)


        actor_obs = torch.cat([obs, extrin], dim=-1)
        critic_obs = torch.cat([obs, extrin_gt], dim=-1)

        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std


        return mu, mu * 0 + sigma, value, extrin, extrin_gt


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


