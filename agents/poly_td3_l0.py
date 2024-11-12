import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import DataLoader
from agents.nn.layers import L0Dense

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, state_dim, coef_dim, max_action, poly, h_dim=256, scale=400.0, action_dim=1, min_action=-1.0,
                 device='cuda'):
        super(Actor, self).__init__()

        droprate = 0.2

        self.actor_coef_1 = L0Dense(in_features=coef_dim, out_features=1, bias=False, local_rep=False,
                                    droprate_init=droprate)
        self.actor_coef_2 = L0Dense(in_features=coef_dim, out_features=1, bias=False, local_rep=False,
                                    droprate_init=droprate)
        self.actor_coef_3 = L0Dense(in_features=coef_dim, out_features=1, bias=False, local_rep=False,
                                    droprate_init=droprate)
        self.actor_coef_4 = L0Dense(in_features=coef_dim, out_features=1, bias=False, local_rep=False,
                                    droprate_init=droprate)
        self.actor_coef_5 = L0Dense(in_features=coef_dim, out_features=1, bias=False, local_rep=False,
                                    droprate_init=droprate)
        self.actor_coef_6 = L0Dense(in_features=coef_dim, out_features=1, bias=False, local_rep=False,
                                    droprate_init=droprate)
        self.actor_coef_7 = L0Dense(in_features=coef_dim, out_features=1, bias=False, local_rep=False,
                                    droprate_init=droprate)
        self.actor_coef_8 = L0Dense(in_features=coef_dim, out_features=1, bias=False, local_rep=False,
                                    droprate_init=droprate)

        self.max_action = max_action
        self.min_action = min_action

        self.poly = poly
        self.scale = scale

        self.device = device

    def forward(self, x):
        x[:, 0] = x[:, 0]/self.scale
        x[:, 1] = x[:, 1]/self.scale
        p = torch.from_numpy(self.poly.fit_transform((x).cpu().numpy())).to(self.device)
        a1 = self.actor_coef_1(p)
        a2 = self.actor_coef_2(p)
        a3 = self.actor_coef_3(p)
        a4 = self.actor_coef_4(p)
        a5 = self.actor_coef_5(p)
        a6 = self.actor_coef_6(p)
        a7 = self.actor_coef_7(p)
        a8 = self.actor_coef_8(p)
        a = torch.cat([a1, a2, a3, a4, a5, a6, a7, a8], dim=1)
        a = self.max_action * torch.tanh(a)
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim=32):
        super(Critic, self).__init__()

        action_dim = action_dim

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.l3 = nn.Linear(h_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, h_dim)
        self.l5 = nn.Linear(h_dim, h_dim)
        self.l6 = nn.Linear(h_dim, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, degree_pi=2, feature_scale=400.0, optimizer='Adam',
                 h_dim=256, tau=0.005, reg_coeff=0.001, min_action=-1.0, device='cuda'):

        self.poly_pi = PolynomialFeatures(degree=degree_pi)
        x = np.ones((1, state_dim))
        p = self.poly_pi.fit_transform(x)
        coef_dim = p.shape[1]
        print("policy polynomial of order ", degree_pi)
        print("with {} coefficients".format(coef_dim))
        print(self.poly_pi.get_feature_names_out())

        self.actor = Actor(state_dim, coef_dim, max_action, poly=self.poly_pi, scale=feature_scale,
                           action_dim=action_dim, min_action=min_action, device=device).to(device)
        self.actor_target = Actor(state_dim, coef_dim, max_action, poly=self.poly_pi, scale=feature_scale,
                                  action_dim=action_dim, min_action=min_action, device=device).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, h_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, h_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.tau = tau
        self.reg_coeff = reg_coeff

        self.device = device

    def select_action(self, state):
        self.actor.eval()
        self.actor.training = False
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def reset(self):
        pass

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, policy_noise=0.2, noise_clip=0.3,
              policy_freq=2, clipped_noise=False):

        self.critic.train()
        self.actor.train()
        self.actor.training = True
        for it in range(iterations):

            # Sample replay buffer
            x, u, y, r, d = replay_buffer.sample(batch_size)
            state = x
            action = u
            next_state = y
            reward = r
            done = 1.0 - d

            if clipped_noise:
                # Select action according to policy and add clipped noise
                noise = torch.FloatTensor(action.size()).data.normal_(0, policy_noise).to(self.device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            else:
                next_action = self.actor_target(next_state).detach()

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                l0_norm_actor = -self.reg_coeff * (self.actor.actor_coef_1.regularization() +
                                                   self.actor.actor_coef_2.regularization() +
                                                   self.actor.actor_coef_3.regularization() +
                                                   self.actor.actor_coef_4.regularization() +
                                                   self.actor.actor_coef_5.regularization() +
                                                   self.actor.actor_coef_6.regularization() +
                                                   self.actor.actor_coef_7.regularization() +
                                                   self.actor.actor_coef_8.regularization())
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean() + l0_norm_actor

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor_poly.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic_poly.pth' % (directory, filename))

    def load(self, filename, directory):
        if not torch.cuda.is_available():
            self.actor.load_state_dict(torch.load('%s/%s_actor_poly.pth' % (directory, filename), map_location='cpu'))
            self.critic.load_state_dict(torch.load('%s/%s_critic_poly.pth' % (directory, filename), map_location='cpu'))
        else:
            self.actor.load_state_dict(torch.load('%s/%s_actor_poly.pth' % (directory, filename)))
            self.critic.load_state_dict(torch.load('%s/%s_critic_poly.pth' % (directory, filename)))

def load(filename, directory):
    if not torch.cuda.is_available():
        return torch.load('%s/%s_all_poly.pth' % (directory, filename), map_location='cpu')
    else:
        return torch.load('%s/%s_all_poly.pth' % (directory, filename))