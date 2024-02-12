import torch
import logging
import itertools
import numpy as np
import torch.nn as nn
import numpy.linalg as la

from torch.nn.functional import softmax
from shapely.geometry import Polygon

from simulation.envs.policy.policy import Policy
from simulation.envs.utils.action import ActionRot, ActionXY
from simulation.envs.utils.state import ObservableState, FullState
from control.policy.value_network import ValueNetwork


class SARL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'
        self.trainable = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim
        self.multiagent_training = True

        # Safety features
        self.safety = None
        self.walls = None

    def configure(self, config):
        self.gamma = config.getfloat('rl', 'gamma')

        self.kinematics = config.get('action_space', 'kinematics')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')

        mlp1_dims = [int(x) for x in config.get('network', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('network', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('network', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('network', 'attention_dims').split(', ')]
        with_global_state = config.getboolean('network', 'with_global_state')

        self.model = ValueNetwork(self.joint_state_dim, self.self_state_dim, mlp1_dims,
                                    mlp2_dims, mlp3_dims, attention_dims, with_global_state)
        logging.info('Policy: Baseline SARL')

        # Safety features
        self.safety = config.getboolean('safety', 'safety')
        self.slow = config.getboolean('safety', 'slow')
        self.margin = config.getfloat('safety', 'margin')
        self.spread = config.getint('safety', 'spread')

        if self.safety:
            logging.info('Safety controller is activated.')
        else:
            logging.info('Safety controller is unactivated.')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_walls(self, walls):
        self.walls = walls

    def get_attention_weights(self):
        return self.model.attention_weights

    def build_action_space(self, v_pref):
        holonomic = self.kinematics == 'holonomic'
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, -np.pi/4)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space
        self.free_directions = np.full(len(action_space), True)

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                        state.wx, state.wy, state.v_pref, state.theta)
            else:
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
                next_px = state.px + next_vx * self.time_step
                next_py = state.py + next_vy * self.time_step
                next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius,
                                        state.wx, state.wy, state.v_pref, next_theta)
        else:
            raise ValueError('State to be propagated is not of a known type.')
        return next_state

    def compute_reward(self, nav, obs):
        # check for collision
        dmin = float('inf')
        collision = False
        for i, ob in enumerate(obs):
            dist = la.norm((nav.px - ob.px, nav.py - ob.py)) - nav.radius - ob.radius
            if dist < 0:
                collision = True
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = la.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius

        # compute reward
        if reaching_goal:
            reward = 1
        elif collision:
            reward = -0.25
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0
        return reward

    def transform(self, state, eps=None):
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
        state_tensor = self.rotate(state_tensor)
        return state_tensor

    def rotate(self, state):
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            theta = torch.zeros_like(v_pref)

        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)

        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state

    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase and device attributes have to be set.')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase.')

        # Stop moving once the goal is reached
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)

        # Build the action space at the start
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
            if self.safety:
                self.build_radar(state.self_state.v_pref, state.self_state.radius)

        # Update free_directions
        self.find_free_directions(state.self_state, state.human_states)

        # Robot has observed human
        if state.human_states:
            # Select action according to epsilon greedy policy
            probability = np.random.random()
            if self.phase == 'train' and probability < self.epsilon:
                max_action = self.action_space[np.random.choice(len(self.action_space))]
            else:
                self.action_values = list()
                max_value = float('-inf')
                max_action = None
                naive_value = float('-inf')
                for free, action in zip(self.free_directions, self.action_space):
                    # Get input to value network for given action
                    next_self_state = self.propagate(state.self_state, action)
                    if self.query_env:
                        next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                    else:
                        next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                           for human_state in state.human_states]
                        reward = self.compute_reward(next_self_state, next_human_states)
                    batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                                  for next_human_state in next_human_states], dim=0)
                    rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)

                    # Compute value of action using value network
                    next_state_value = self.model(rotated_batch_input).data.item()
                    value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value

                    if value > naive_value:
                        naive_value = value
                    if free:
                        self.action_values.append(value)
                        if value > max_value:
                            max_value = value
                            max_action = action
                    else:
                        self.action_values.append(np.nan)

                if max_action is None:
                    raise ValueError('Value network is not well trained.')

                # if there was a "better" action without safety, then the safety is active
                self.safety_active = max_value != naive_value

            if self.phase == 'train':
                self.last_state = self.transform(state)

            return max_action

        # Robot has not observed human
        else:
            # Select action towards goal position
            px, py = state.self_state.px, state.self_state.py
            wx, wy = state.self_state.wx, state.self_state.wy
            theta  = state.self_state.theta
            speed = min(np.linalg.norm((wy-py, wx-px)), state.self_state.v_pref)
            angle = np.arctan2(wy-py, wx-px)
            if self.kinematics == 'holonomic':
                action = ActionXY(speed * np.cos(angle), speed * np.sin(angle))
            else:
                rot = angle - theta
                rot = (rot + np.pi) % (2 * np.pi) - np.pi
                action = ActionRot(speed, rot)
            return self.review_action(action, state)

    def build_radar(self, v_pref, robot_radius):
        if self.slow:
            # Use positive margin for safety buffer. Use 0 margin to disable safety buffer.
            if self.margin >= 0:
                margins = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * self.margin + robot_radius for i in range(self.speed_samples)]
            # Use negative margin for a default 2-second following space.
            else:
                margins = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * 2 * v_pref + robot_radius for i in range(self.speed_samples)]
        # If we aren't slowing down, we just use widest margins.
        else:
            if self.margin >= 0:
                margins = [self.margin + robot_radius]
            else:
                margins = [2*v_pref + robot_radius]

        # Rotations are standard.
        holonomic = self.kinematics == 'holonomic'
        if self.rotations is None:
            if holonomic:
                rotations = np.linspace(0, 2*np.pi, self.rotation_samples, endpoint=False)
            else:
                rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)
        else:
            rotations = self.rotations

        radar_vertices = []
        if holonomic:
            theta = np.pi / self.rotation_samples
        else:
            theta = (np.pi / 4) / self.rotation_samples
        tan_theta = np.tan(theta)
        plus_minus_one = [1, -1]
        for rotation, margin in itertools.product(rotations, margins):
            new_vertices = []
            radar_line = [margin * np.cos(rotation), margin * np.sin(rotation)]
            margin_width = robot_radius
            radar_offset = [margin_width * np.sin(rotation), -margin_width * np.cos(rotation)]
            new_vertices += [(radar_line[0] + sign * radar_offset[0], radar_line[1] + sign * radar_offset[1]) for sign in plus_minus_one]
            new_vertices += [(-sign * radar_offset[0], -sign * radar_offset[1]) for sign in plus_minus_one]
            radar_vertices.append(new_vertices)

        self.num_margins = len(margins)
        self.margins = margins
        self.rotations = rotations
        self.radar_vertices = radar_vertices

    def find_free_directions(self, robo, obs):
        # If the safety controller is not active, or we don't have any walls, then all directions are free.
        if not self.safety or self.walls is None:
            self.free_directions[1:] = True
            return
        holonomic = self.kinematics == 'holonomic'

        free_list = []
        sin_t = np.sin(robo.theta)
        cos_t = np.cos(robo.theta)

        for indexo, vertices in enumerate(self.radar_vertices):
            if holonomic:
                shifted_vertices = [(robo.position[0] + vertex[0], robo.position[1] + vertex[1]) for vertex in vertices]
            else:
                # rotate counterclockwise by robo.theta radians
                rotated_vertices = [(cos_t*vertex[0] - sin_t*vertex[1], sin_t*vertex[0] + cos_t*vertex[1]) for vertex in vertices]
                shifted_vertices = [(robo.position[0] + vertex[0], robo.position[1] + vertex[1]) for vertex in rotated_vertices]
            radar_wedge = Polygon(shifted_vertices)
            free = True
            if self.walls.intersects(Polygon(shifted_vertices)):
                free = False

            xboii = [coolio[0] for coolio in shifted_vertices]*2
            yboii = [coolio[1] for coolio in shifted_vertices]*2

            if self.slow:
                free_list.append(free)
            else:
                free_list += [free for i in range(self.speed_samples)]

        free_list = np.array(free_list)
        roller = np.ones((len(free_list), 1 + 2*self.spread))
        roller[:, 0] = free_list
        for i in range(1, self.spread + 1):
            if holonomic:
                roller[:, i] = np.roll(free_list, i * self.speed_samples)
                roller[:, i + self.spread] = np.roll(free_list, -i * self.speed_samples)
            else: # non-holonomic spread does not wrap around
                roller[i*self.speed_samples:, i] = free_list[:-i*self.speed_samples]
                roller[:-i*self.speed_samples, i] = free_list[i*self.speed_samples:]

        # Update our list of free directions
        self.free_directions[1:] = roller.all(axis=1)

    # provide an option to find the nearest safe action
    def review_action(self, proposed_action, state):
        # If we don't care about safety, just return the proposed action
        if not self.safety:
            return proposed_action

        # Set up the things if necessary.
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.safety and self.radar_vertices is None:
            self.build_radar(state.self_state.v_pref, state.self_state.radius)

        # Identify areas that have obstacles in the way
        self.find_free_directions(state.self_state, state.human_states)

        # Determine x and y components of proposed action
        holonomic = self.kinematics == 'holonomic'
        if holonomic:
            proposed_ax = proposed_action[0]
            proposed_ay = proposed_action[1]
        else:
            # if we want to rotate a bunch, just rotate without moving forward
            proposed_ax = proposed_action[0] * np.cos(proposed_action[1])
            proposed_ay = proposed_action[0] * np.sin(proposed_action[1])

        # We find the closest action to the proposed action that is still safe
        closest_action = 0
        min_dist = np.linalg.norm([proposed_ax, proposed_ay])
        naive_min_dist = min_dist
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
        for i, action in enumerate(self.action_space):
            if holonomic:
                ax = action[0]
                ay = action[1]
            else:
                ax = action[0] * np.cos(action[1])
                ay = action[0] * np.sin(action[1])
            new_dist = np.linalg.norm([proposed_ax - ax, proposed_ay - ay])
            if new_dist < min_dist and self.free_directions[i]:
                min_dist = new_dist
                closest_action = i
            if new_dist < naive_min_dist:
                naive_min_dist = new_dist
        # log when the safety controller is actively preventing an "optimal" path
        self.safety_active = min_dist != naive_min_dist

        if self.slow:
            return self.action_space[closest_action]
        else:
            if holonomic:
                speed = np.linalg.norm(proposed_action)
                action_direction = self.action_space[closest_action]
                speed_factor = speed/np.linalg.norm(action_direction)
                return ActionXY(action_direction.vx * speed_factor, action_direction.vy * speed_factor)
            else:
                speed = proposed_action[0]
                action_direction = self.action_space[closest_action][1]
                return ActionRot(speed, action_direction)
