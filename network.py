import torch
from torch import nn
import numpy as np
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
class DDPGnetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DDPGnetwork, self).__init__()
        self.actor = actorNet(obs_dim, act_dim)
        self.critic = criticNet(obs_dim + act_dim, 1)
    def forward(self, obs):
        return self.actor(obs)
    def value(self, obs_and_action):
        return self.critic(obs_and_action)

class actorNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(actorNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 32, True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, act_dim, True)
        self.tanh3 = nn.Tanh()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.tanh3(x)
        return x

class criticNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(criticNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 32, True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, act_dim, True)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    import copy
    t = DDPGnetwork(10, 1)
    t_actor = []
    t_critic = []
    for p in t.actor.parameters():
        t_actor.append(copy.deepcopy(p))
    for p in t.critic.parameters():
        t_critic.append(copy.deepcopy(p))

    obs = torch.rand((100, 10))
    from torch import optim
    optimizer = optim.Adam(t.actor.parameters(), lr = 0.1)
    optimizer2 = optim.Adam(t.critic.parameters(), lr = 0.1)
    t.train()
    action = t(obs)
    obs_and_action = torch.cat([obs, action], dim = -1)
    Q = t.value(obs_and_action)
    loss = -1 * torch.mean(Q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("actor param", "_"*50)
    # for i, p in enumerate(t.actor.parameters()):
    #     print(torch.sum(t_actor[i] - p))
    # print("critic param", "_"*50)
    # for i, p in enumerate(t.critic.parameters()):
    #     print(torch.sum(t_critic[i] - p))
    optimizer2.zero_grad()
    for p in t.critic.parameters():
        print(p.grad)

    
        


    
    