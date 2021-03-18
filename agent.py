import sys
sys.path.append(".")
from DDPG import network
import copy
import torch
import numpy as np
from torch import nn, optim
import os
class DDPG_agent():
    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr, tau, gamma):
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.model = network.DDPGnetwork(obs_dim, act_dim)
        self.target_model = network.DDPGnetwork(obs_dim, act_dim)
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.actor_optim = optim.Adam(self.model.actor.parameters(), self.actor_lr)
        self.critic_optim = optim.Adam(self.model.critic.parameters(), self.critic_lr)
    def predict(self, obs):
        with torch.no_grad():
            self.model.to(self.device)
            obs = np.expand_dims(obs, axis = 0)
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            return self.model(obs).detach().cpu().numpy()[0]

    def learn(self, obs, action, reward, next_obs, terminal):

        self._critic_learn(obs, action, reward, next_obs,terminal)
        self._actor_learn(obs)
        self.sync_target()

    def _actor_learn(self, obs):
        self.model.to(self.device)
        self.model.train()
        # print("obs.shape {}".format(obs.shape))
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action = self.model(obs)
        # print("action.shape {}".format(action.shape))
        obs_and_act = torch.cat([obs, action], dim = -1)
        # print("obs_and_act.shape {}".format(obs_and_act.shape))
        Q = self.target_model.value(obs_and_act)
        # print("Q.shape {}".format(Q.shape))
        loss = torch.mean(-1.0 * Q)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def _critic_learn(self, obs, act, reward, next_obs, terminal):
        self.model.to(self.device)
        self.model.train()
        terminal = np.expand_dims(terminal, axis = -1)
        reward = np.expand_dims(reward, axis = -1)
        obs, act, reward, next_obs, terminal = torch.tensor(obs, dtype = torch.float32), torch.tensor(act, dtype = torch.float32), torch.tensor(reward, dtype = torch.float32), torch.tensor(next_obs, dtype = torch.float32), torch.tensor(terminal, dtype = torch.float32)
        obs, act, reward, next_obs, terminal = obs.to(self.device), act.to(self.device), reward.to(self.device), next_obs.to(self.device), terminal.to(self.device)
        
        # print("obs.shape {}".format(obs.shape))
        # print("act.shape {}".format(act.shape))
        # print("reward.shape {}".format(reward.shape))
        # print("next_obs.shape {}".format(next_obs.shape))
        # print("terminal.shape {}".format(terminal.shape))
        self.target_model.to(self.device)
        self.target_model.eval()
        with torch.no_grad():
            next_action = self.target_model(next_obs)
            # print("next_action.shape {}".format(next_action.shape))
            obs_and_act = torch.cat([next_obs, next_action.detach()], dim = -1)
            # print("obs_and_act.shape {}".format(obs_and_act.shape))
            next_Q = self.target_model.value(obs_and_act)
            target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
            # print("target_Q.shape {}".format(target_Q.shape))

        obs_and_act2 = torch.cat([obs, act], dim = -1) 
        # print("obs_and_act2.shape {}".format(obs_and_act2.shape))
        Q = self.model.value(obs_and_act2)
        # print("Q.shape {}".format(Q.shape))
        loss = nn.MSELoss()(Q, target_Q.detach())
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()


    def sync_target(self, decay=None, share_vars_parallel_executor=None):
        """ self.target_model从self.model复制参数过来，可设置软更新参数
        """
        self.target_model.to("cpu")
        self.model.to("cpu")
        if decay is None:
            decay = self.tau
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - decay) +
                param.data * decay
            )
    def save(self, name):
        torch.save(self.model, os.path.join("DDPG", name + ".pth"))
    def load(self, path):
        self.model = torch.load(path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
        self.sync_target()
