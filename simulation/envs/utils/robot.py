import numpy as np
from numpy.linalg import norm
from shapely.geometry import Point

from simulation.envs.utils.agent import Agent
from simulation.envs.policy.orca import ORCA
from simulation.envs.utils.state import JointState, FullState

class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

        # Waypoints
        self.waypoints = None
        self.wypt_idx = None
        self.wypt_radius = None
        self.wx = self.gx
        self.wy = self.gy

    def set_waypoints(self, waypoints, wypt_radius):
        self.waypoints = waypoints
        self.wypt_idx = 0
        self.wypt_radius = wypt_radius
        self.wx = waypoints[0][0]
        self.wy = waypoints[0][1]

    def set_safety(self, safety):
        self.safety = safety

    def check_waypoint(self, ob, walls, wx, wy):
        if walls is not None:
            if walls.intersects(Point((wx, wy)).buffer(self.radius + 0.2)):
                return False
        return True

    def get_full_state(self, ob=None):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta, wx=self.wx, wy=self.wy)

    def act(self, ob, eps=None, walls=None):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(ob), ob)

        # Update the next waypoint if necessary
        if self.waypoints:
            dist_to_waypoint = norm([self.px - self.wx, self.py - self.wy])
            while dist_to_waypoint < self.wypt_radius and self.wypt_idx + 1 < len(self.waypoints):
                self.wypt_idx += 1
                self.wx = self.waypoints[self.wypt_idx][0]
                self.wy = self.waypoints[self.wypt_idx][1]
                dist_to_waypoint = norm([self.px - self.wx, self.py - self.wy])
            # Move the waypoint side to side if it is too close to an obstacle.
            shift_dir = np.array([self.wy - self.py, self.px - self.wx])
            shift_dir = shift_dir/np.linalg.norm(shift_dir)
            shift_attempts = 1
            while not self.check_waypoint(ob, walls, self.wx, self.wy):
                self.wx += ((-1)**shift_attempts)*shift_attempts*0.1*shift_dir[0]
                self.wy += ((-1)**shift_attempts)*shift_attempts*0.1*shift_dir[1]
                shift_attempts += 1

        if self.policy.uncertainty:
            action = self.policy.predict(state, eps)
        else:
            action = self.policy.predict(state)
        return action
