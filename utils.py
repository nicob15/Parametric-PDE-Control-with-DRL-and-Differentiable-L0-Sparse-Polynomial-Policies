import numpy as np
import cv2
import torch
try:
    import cPickle as pickle
except:
    import pickle

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def get_all_samples(self, nr_samples=20000):
        if self.size < nr_samples:
            size = self.size
        else:
            size = nr_samples
        return (
            torch.FloatTensor(self.state[:size]).to(self.device),
            torch.FloatTensor(self.action[:size]).to(self.device),
            torch.FloatTensor(self.next_state[:size]).to(self.device),
            torch.FloatTensor(self.reward[:size]).to(self.device),
            torch.FloatTensor(self.not_done[:size]).to(self.device)
        )


