import pandas as pd
import numpy as np
from sklearn import datasets, model_selection, ensemble
import shap
import seaborn as sns

import torch
import matplotlib.pyplot as plt
import sys 
sys.path.append("../modules")
#from rl.Gen_his.modules.actor_critic import ActorCritic
from actor_critic import DmEncoder, ActorCritic


extrin_file = '/home/leo/research/leggedR/adaptive_legged_gym/rl/Gen_his/utils/shap_data/aliengo2_extrin.npy'
proprio_hist_file = '/home/leo/research/leggedR/adaptive_legged_gym/rl/Gen_his/utils/shap_data/aliengo2_proprio_hist.npy'

extrin = np.load(extrin_file)[20:]
proprio_hist = np.load(proprio_hist_file)[20:]

print("Points in the trajctory: ", extrin.shape, proprio_hist.shape)
#print(extrin[270:350])

# load the NN model
train_cfg = {
    'priv_mlp_units': [256, 128, 13],
    'priv_info' : True,
    'priv_info_dim' : 210,
    'proprio_adapt' : True,
    'proprio_adapt_out_dim' : 11,
    'estLen':13,
    'HistoryLen':5,
}
actor_critic = ActorCritic(num_obs=46,
                           num_actions=12,
                           actor_hidden_dims=[512, 256, 128],
                           critic_hidden_dims=[512, 256, 128],
                           activation='elu',
                           init_noise_std=1.0,
                           **train_cfg)

dm_encoder = DmEncoder(46*20, [256, 128, 13])

path = '/home/leo/research/leggedR/adaptive_legged_gym/outputs/aliengo/gen_his/20hist_P_0_all/stage1_nn/model_850.pt'
loaded_dict = torch.load(path)
#for key in loaded_dict:
#    print(key)
#print('--------------------------------------loaded_dict---------------', loaded_dict['model_state_dict'])
dm_encoder.load_state_dict(loaded_dict['dm_encoder_state_dict'])
# actor_critic.eval()

model = dm_encoder#actor_critic.dm_encoder

# 背景图像样本
proprio_hist = torch.from_numpy(proprio_hist)
proprio_hist = proprio_hist.flatten(1)

test_idx = 194# 268 before, 275-300 #110
test = model(proprio_hist[test_idx].unsqueeze(0))
test = torch.tanh(test)
print("************ test ", test.shape, test)

proprio_hist_bg = proprio_hist
proprio_hist_test = proprio_hist[test_idx].unsqueeze(0)
print("background shape ", proprio_hist_bg.shape)
print("test shape ", proprio_hist_test.shape)

e = shap.DeepExplainer(model, proprio_hist_bg)
print("****************** finish init explainer ******************")
shap_values = e.shap_values(proprio_hist_test)
print("test shape value length ", len(shap_values))

shap_values_all_outputs = np.zeros((proprio_hist_test.shape[1],))
for i in range(len(shap_values)):
    shap_values_all_outputs += np.absolute(shap_values[i].squeeze())
print(shap_values_all_outputs[-1])

shap_values_all_outputs_all_obs = np.zeros((20,))  # sum all observation shap value in 1 hist step
for i in range(20):
    shap_values_all_outputs_all_obs[i] = np.sum(shap_values_all_outputs[i*46 : (i+1)*46])
print("20 hist ", shap_values_all_outputs_all_obs)
t = np.arange(shap_values_all_outputs_all_obs.shape[0])

# TODO: the data for the second heatmap
# test_idx2 = 190# 268 before, 275-300 #110
# test2 = model(proprio_hist[test_idx].unsqueeze(0))
# test2 = torch.tanh(test)
# print("************ test ", test.shape, test)

# proprio_hist_test2 = proprio_hist[test_idx2].unsqueeze(0)

# shap_values2 = e.shap_values(proprio_hist_test2)

# shap_values_all_outputs2 = np.zeros((proprio_hist_test2.shape[1],))
# for i in range(len(shap_values2)):
#     shap_values_all_outputs2 += np.absolute(shap_values2[i].squeeze())

# shap_values_all_outputs_all_obs2 = np.zeros((20,))  # sum all observation shap value in 1 hist step
# for i in range(20):
#     shap_values_all_outputs_all_obs2[i] = np.sum(shap_values_all_outputs2[i*46 : (i+1)*46])
# print("20 hist ", shap_values_all_outputs_all_obs2)

# Drawing:
plt.figure(figsize=[5,2.0])
# bar chart 
#plt.bar(t, np.square(shap_values_all_outputs_all_obs)) # 想办法优化一下此处SHAP的value值的大小 看起来更有分辨度
# heatmap
heat_image = np.array([shap_values_all_outputs_all_obs])
#heat_image = shap_values_all_outputs_all_obs.reshape(20,1)
print('-------------heatMap',heat_image.shape)
heatmap = plt.imshow(heat_image, cmap='viridis',vmin=0.25, vmax=1.35)
plt.colorbar(heatmap, orientation='vertical', fraction=0.02, pad=0.04)
#sns.heatmap(heat_image, annot=True, cmap='YlGnBu')

plt.tick_params(labelsize=8)
#plt.ylim(0., 1.6)  # only used in bar chart
plt.ylabel("Shapley value", fontsize=10) # only used in bar char
plt.xlabel("Propriocetion History", fontsize=10)
plt.tight_layout()
plt.show()


# print("for feature 1, shap shape ", shap_values[0].shape)
# shap.summary_plot(shap_values, proprio_hist_test)

# drawing test:
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,1.5))
# heat_image = np.array([shap_values_all_outputs_all_obs])
# heat_image2 = np.array([shap_values_all_outputs_all_obs2])
# sns.heatmap(heat_image, annot=True, cmap='YlGnBu', ax = ax1)
# sns.heatmap(heat_image2, annot=True, cmap='YlGnBu', ax=ax2)
# plt.show()
