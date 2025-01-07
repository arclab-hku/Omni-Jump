import numpy as np
from termcolor import cprint

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

import os

#from .legged_robot_config import LeggedRobotCfg

 
class DmEncoder(nn.Module):  # SHAP values output
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

        #self.cfg = cfg

        # ---- Priv Info ----
        self.priv_mlp = kwargs['priv_mlp_units']
        self.priv_info = kwargs['priv_info']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.priv_info_stage2 = kwargs['proprio_adapt']

        self.proprio_adapt_input = num_obs
        self.proprio_adapt_output = kwargs['proprio_adapt_out_dim']

        self.estLen = kwargs['estLen']
        self.HistoryLen = kwargs['HistoryLen']

        self.num_actor_input = num_obs + self.estLen #+ 197#+264 # +197 means adding the measure heights in the actor. estLen is 13.

        num_critic_input = num_obs + self.priv_info_dim ##### 45 + 3 + 187 #  num_obs is proprio info(45), priv_info_dim is the whold compute obs dim (11*17+3+4+4+2)

        self.num_encoder_input = num_obs * self.HistoryLen
        self.encoder_mlp = kwargs['priv_mlp_units'] # remember to change the paras in robot_config.py.

        self.dm_encoder = DmEncoder(self.num_encoder_input,  self.encoder_mlp)

        cprint(f"Encoder MLP: {self.dm_encoder}", 'green', attrs=['bold'])

        self.extrin_file_dir = '/home/leo/research/leggedR/adaptive_legged_gym/rl/Gen_his/utils/shap_data/aliengo2_extrin.npy'
        #self.extrin_file = 'extrin.npy'
        self.proprio_hist_file_dir = '/home/leo/research/leggedR/adaptive_legged_gym/rl/Gen_his/utils/shap_data/aliengo2_proprio_hist.npy'
        #self.proprio_hist_file = 'proprio_hist.npy'
        self.extrin_list, self.proprio_hist_list = [], []


        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_actor_input, actor_hidden_dims[0]))
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
        return extrin # dim = cfg.priv_mlp_units[2]

    def extrin_gt_loss(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, _, _, extrin_gt = self._actor_critic(obs_dict)

        return extrin_gt
    def _actor_critic(self, obs_dict):
        
        # vel tracking:
        # obs = obs_dict['obs']
        # obs_vel = obs_dict['privileged_info'][:, 20:23]
        # obs_feet_pos = obs_dict['privileged_info'][:, 8:20]
        # obs_z = obs_dict['privileged_info'][:, 0:1]
        # obs_mass = obs_dict['privileged_info'][:, 1:5]
        # obs_ang_vel = obs_dict['privileged_info'][:, 5:8]
        # #obs_hight = obs_dict['privileged_info'][:, 3:200]
        # obs_hight = obs_dict['privileged_info'][:, 23:220]#+264]
        # obs_proprio_hist = obs_dict['proprio_hist']
        # obs_his = obs_proprio_hist
        # extrin_gt = obs_dict['privileged_info'][:, 0:23]

        # ZXY tracking and feet pos:
        obs = obs_dict['obs']
        #obs_vel = obs_dict['privileged_info'][:, 7:10]
        obs_vel = obs_dict['privileged_info'][:, 10:13]
        obs_feet_pos = obs_dict['privileged_info'][:, 3:7]
        obs_zxy = obs_dict['privileged_info'][:, 0:3]
        #obs_mass = obs_dict['privileged_info'][:, 3:7]
        obs_ang_vel = obs_dict['privileged_info'][:, 7:10]
        #obs_hight = obs_dict['privileged_info'][:, 3:200]
        #obs_hight = obs_dict['privileged_info'][:, 10:207]#+264]
        obs_hight = obs_dict['privileged_info'][:, 13:210]
        obs_proprio_hist = obs_dict['proprio_hist']
        obs_his = obs_proprio_hist
        #extrin_gt = obs_dict['privileged_info'][:, 0:10] 
        extrin_gt = obs_dict['privileged_info'][:, 0:13] 
        # Z tracking:
        # obs = obs_dict['obs']
        # obs_vel = obs_dict['privileged_info'][:, 4:7]#obs_dict['privileged_info'][:, 10:13]
        # #obs_feet_pos = obs_dict['privileged_info'][:, 3:7]
        # #obs_zxy = obs_dict['privileged_info'][:, 0:3]
        # obs_z = obs_dict['privileged_info'][:, 0:1]
        # #obs_mass = obs_dict['privileged_info'][:, 3:7]
        # obs_ang_vel = obs_dict['privileged_info'][:, 1:4]#obs_dict['privileged_info'][:, 7:10]
        # #obs_hight = obs_dict['privileged_info'][:, 3:200]
        # obs_hight = obs_dict['privileged_info'][:, 7:204]#[:, 13:210]#+264]
        # obs_proprio_hist = obs_dict['proprio_hist']
        # obs_his = obs_proprio_hist
        # extrin_gt = obs_dict['privileged_info'][:, 0:7]#[:, 0:13]            

        # cprint(f"obs_sffsdgg: {obs.shape,  obs_his.shape, obs, obs_his}", 'green', attrs=['bold'])
        
        # print('obs_proprio_hist', obs_proprio_hist.shape, obs_proprio_hist)
         
        # cprint(f"obs_his: {obs_his.shape, obs_his}", 'red', attrs=['bold'])

    # for SHAP analysis：
        extrin_en = self.dm_encoder(obs_his) # output dim=13
        #extrin_en.reshape(self.cfg.num_envs, self.cfg.num_histroy_obs, self.cfg.num_observations) # envs, 5, 46
        #print('----------------obs_his dimension is:', obs_his.size()) #obs_his: (num_envs, 46*5=230) 
        #print('-------------------------------------------')
    # store the value as test_DATAset for SHAP analysis
        self.extrin_list += [extrin_en.detach().cpu().numpy()]
        self.proprio_hist_list += [obs_dict['proprio_hist'].detach().cpu().numpy()]
        extrin_array = np.array(self.extrin_list).reshape(-1, extrin_en.shape[1])
        #proprio_hist_array = np.array(self.proprio_hist_list).reshape(-1, 20, 46)
    #store the input obs_his here.
        #np.save(self.proprio_hist_file_dir, proprio_hist_array)
    # store the encoder output here 
        #np.save(self.extrin_file_dir, extrin_array)

        # # extrin = torch.tanh(extrin_en)
        extrin_gt = torch.tanh(extrin_gt)
        #print('----------------estimation height is:', extrin_en.size()) # extrin_en shape is(num_envs, estimation_length=13)
        #print('-------------------------------------------')
        #actor_obs = torch.cat([ obs, extrin_en], dim=-1)  ## 45 + 3
        actor_obs = torch.cat([extrin_en, obs], dim=-1)#torch.cat([ obs, extrin_en, obs_hight], dim=-1) # 45 + 3 + 197
        #critic_obs = torch.cat([obs_z, obs_mass, obs_ang_vel, obs_feet_pos, obs_vel, obs, obs_hight], dim=-1)  ## 45+3+197 = 245
        #critic_obs = torch.cat([obs_z, obs_ang_vel, obs_vel, obs, obs_hight], dim=-1)  ## 45+3+197 = 245
        critic_obs = torch.cat([obs_zxy, obs_feet_pos, obs_ang_vel, obs_vel, obs, obs_hight], dim=-1)  ## 45+3+197 = 245

        # extrin_en and the input observation of critics最好一一对应
        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std

        return mu, mu * 0 + sigma, value, extrin_en, extrin_gt


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


