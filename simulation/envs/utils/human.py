import os
import torch

from numpy.linalg import norm
from numpy.random import normal
from simulation.envs.utils.agent import Agent
from simulation.envs.policy.orca import ORCA
from simulation.envs.utils.action import ActionXY, ActionRot
from simulation.envs.utils.state import JointState, FullState


class Human(Agent):
    def __init__(self, config, section, seed=None):
        super().__init__(config, section)
        self.epsilon = 0.0

        # Load model weights for CADRL model
        if self.policy is not None and self.policy.name == 'CADRL':
            model_dir = '../simulation/envs/policy/cadrl/'
            model_weights = os.path.join(model_dir, 'rl_model.pth')
            self.policy.get_model().load_state_dict(torch.load(model_weights))
            self.policy.phase = 'test'
            self.policy.set_device(torch.device("cpu"))

    # A function to set how random we the human to be
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # A function to set the max speed of the human policy
    def set_policy_speed(self, max_speed):
        self.policy.max_speed = max_speed

    # A function to set the time horizon of the human policy
    def set_policy_horizon(self, time_horizon):
        self.policy.time_horizon = time_horizon

    # A function to set the safety space of the human policy
    def set_policy_safety(self, safety_space):
        self.policy.safety_space = safety_space

    # A function to set the max number of neighbors considered by the human policy
    def set_policy_neighbors(self, max_neighbors):
        self.policy.max_neighbors = max_neighbors

    def act(self, ob):
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)

        # We assume a fully-unpredictable person will have this distribution of potential actions
        noise = normal(scale=self.v_pref, size=2)

        if isinstance(action, ActionXY):
            return ActionXY(self.epsilon*noise[0] + (1-self.epsilon)*action.vx, self.epsilon*noise[1] + (1-self.epsilon)*action.vy)
        else:
            return action # NOTE: CANNOT ADD NOISE TO NON-HOLONOMIC ACTION
