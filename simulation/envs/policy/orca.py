import rvo2
import logging
import itertools
import numpy as np
import numpy.linalg as la

from simulation.envs.policy.policy import Policy
from simulation.envs.utils.action import ActionXY


class ORCA(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0.2
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 1.5
        self.time_horizon_obst = 0.5
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    # Returns the parameters of our policy.
    def get_params(self):
        # Add or remove parameters as necessary.
        params = {"trainable":self.trainable,
                  "multiagent_training":self.multiagent_training,
                  "kinematics":self.kinematics,
                  "safety_space":self.safety_space,
                  "neighbor_dist":self.neighbor_dist,
                  "max_neighbors":self.max_neighbors,
                  "time_horizon":self.time_horizon,
                  "time_horizon_obst":self.time_horizon_obst,
                  "radius":self.radius,
                  "max_speed":self.max_speed}
        return params

    def predict(self, state):
        self_state = state.self_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            self.sim.addAgent(self_state.position, *params, self_state.radius + 0.01 + self.safety_space,
                              self_state.v_pref, self_state.velocity)
            for agent_state in state.human_states:
                self.sim.addAgent(agent_state.position, *params, agent_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, agent_state.velocity)
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, agent_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, agent_state.position)
                self.sim.setAgentVelocity(i + 1, agent_state.velocity)

        # Set the preferred velocity to be a vector in the direction of the goal of at most max_speed magnitude (speed).
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = la.norm(velocity)
        pref_vel = self.max_speed * velocity / speed if speed > self.max_speed else velocity

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, agent_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action
