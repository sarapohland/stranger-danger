import abc
import numpy as np
import numpy.linalg as la


class Policy(object):
    def __init__(self):
        self.trainable = False
        self.uncertainty = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        self.env = None

    @abc.abstractmethod
    def configure(self, config):
        return

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def get_model(self):
        return self.model

    @abc.abstractmethod
    def predict(self, state, eps=None):
        return

    @abc.abstractmethod
    def get_params(self):
        return {}

    @staticmethod
    def reach_destination(state):
        self_state = state.self_state
        if la.norm((self_state.py - self_state.gy, self_state.px - self_state.gx)) < self_state.radius:
            return True
        else:
            return False
