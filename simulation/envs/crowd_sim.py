import gym
import rvo2
import logging
import numpy as np
import numpy.linalg as la

from matplotlib import patches
import matplotlib.lines as mlines

from simulation.envs.utils.info import *
from simulation.envs.utils.human import Human
from simulation.envs.utils.scenarios import GenerateHumans
from simulation.envs.utils.functions import point_to_segment_dist
from uncertainty.estimate_epsilons import estimate_epsilons


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.dmin = None
        self.robot = None
        self.humans = None
        self.open_space = None
        self.previous_distance = None

        # Visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

        # Counters
        self.global_time = None
        self.human_times = None

    def configure(self, config):
        # Simulation configurations
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.end_on_collision = config.getboolean('sim', 'end_on_collision')
        self.plan_human_path = config.getboolean('sim', 'plan_human_path')
        self.perpetual = config.getboolean('sim', 'perpetual')
        self.randomness = config.getboolean('sim', 'randomness')
        self.scenario = config.get('sim', 'scenario')
        self.room_dims = [config.getfloat('sim', 'room_width'), config.getfloat('sim', 'room_height')]

        # Reward function
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.time_penalty = config.getfloat('reward', 'time_penalty')
        self.progress_reward = config.getfloat('reward', 'progress_reward')
        self.goal_radius = config.getfloat('reward', 'goal_radius')

        # Number/density of humans
        self.num_humans = config.getint('sim', 'num_humans')
        self.max_human_num = config.getint('sim', 'max_human_num')
        self.min_human_num = config.getint('sim', 'min_human_num')
        self.max_human_dens = config.getfloat('sim', 'max_human_dens')
        self.min_human_dens = config.getfloat('sim', 'min_human_dens')
        self.random_human_num = config.getboolean('sim', 'random_human_num')
        self.random_human_dens = config.getboolean('sim', 'random_human_dens')

        # Human parameters
        self.min_radius = config.getfloat('humans', 'min_radius')
        self.max_radius = config.getfloat('humans', 'max_radius')
        self.randomize_radius = config.getboolean('humans', 'randomize_radius')
        self.min_v_pref = config.getfloat('humans', 'min_v_pref')
        self.max_v_pref = config.getfloat('humans', 'max_v_pref')
        self.randomize_v_pref = config.getboolean('humans', 'randomize_v_pref')
        self.min_neigh_dist = config.getfloat('humans', 'min_neigh_dist')
        self.max_neigh_dist = config.getfloat('humans', 'max_neigh_dist')
        self.randomize_neigh_dist = config.getboolean('humans', 'randomize_neigh_dist')
        self.min_horizon = config.getfloat('humans', 'min_horizon')
        self.max_horizon = config.getfloat('humans', 'max_horizon')
        self.randomize_horizon = config.getboolean('humans', 'randomize_horizon')

        # Counters
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000,
                                'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000,
                            'val': config.getint('env', 'val_size'),
                            'test': config.getint('env', 'test_size')}

    def set_robot(self, robot):
        self.robot = robot

    def generate_humans(self, human_num, scenario, max_epsilon, seed):
        self.humans = []
        gen = GenerateHumans(self.room_dims, self.discomfort_dist, self.goal_radius, seed)

        for i in range(human_num):
            human = Human(self.config, 'humans')
            if scenario == 'circle':
                human = gen.generate_circle_human(human, self.robot, self.humans)
            elif scenario == 'perpedendicular':
                human = gen.generate_perpendicular_human(human, self.robot, self.humans)
            elif scenario == 'opposite':
                human = gen.generate_opposite_human(human, self.robot, self.humans)
            elif scenario == 'same':
                human = gen.generate_same_human(human, self.robot, self.humans)
            elif scenario == 'random':
                human = gen.generate_random_human(human, self.robot, self.humans)
            else:
                raise ValueError('Scenarios: circle, perpedendicular, opposite, same, random')

            if human is not None:
                human.set_epsilon(np.random.uniform(0, max_epsilon))
                if self.randomize_radius:
                    human.radius = human.policy.radius = np.random.uniform(self.min_radius, self.max_radius)
                if self.randomize_v_pref:
                    human.v_pref = human.policy.max_speed = np.random.uniform(self.min_v_pref, self.max_v_pref)
                if self.randomize_neigh_dist:
                    human.policy.neighbor_dist = np.random.uniform(self.min_neigh_dist, self.max_neigh_dist)
                if self.randomize_horizon:
                    human.policy.time_horizon = np.random.uniform(self.min_horizon, self.max_horizon)
                self.humans.append(human)

    def choose_random_goal(self, human, px, py):
        gx = np.random.random() * self.room_dims[0] - self.room_dims[0]/2
        gy = np.random.random() * self.room_dims[1] - self.room_dims[1]/2

        for agent in [self.robot] + self.humans:
            min_dist = human.radius + agent.radius + self.discomfort_dist
            if la.norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                return gx, gy, True

        return gx, gy, False

    def get_epsilons(self, estimate=False):
        if estimate:
            human_positions = [[self.states[i][1][j].position for i in range(len(self.states))]
                                for j in range(len(self.humans))]
            return estimate_epsilons(human_positions, self.time_step)
        return np.array([human.epsilon for human in self.humans])

    def get_human_times(self):
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref, self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if la.norm(vel_pref) > 1:
                    vel_pref /= la.norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def check_observability(self, nav, agent):
        # Check if agent is within detection range
        dist = la.norm([(nav.px - agent.px), (nav.py - agent.py)]) - nav.radius - agent.radius
        if dist > nav.range:
            return False

        # Check if agent is being blocked by another agent
        agents = self.humans + [self.robot]
        other_agents = [x for x in agents if (x != nav and x != agent)]
        for other in other_agents:
            dist = point_to_segment_dist(nav.px, nav.py, agent.px, agent.py, other.px, other.py)
            if dist < other.radius:
                return False

        return True

    def get_observation(self, nav):
        if nav.sensor != 'coordinates':
            raise ValueError("Agent sensor not implemented")

        agents = self.humans + [self.robot]
        other_agents = [x for x in agents if x != nav]
        full_ob = [agent.get_observable_state() for agent in other_agents if agent.visible]

        if nav.observability == 'full':
            return full_ob
        elif nav.observability == 'partial':
            partial_ob = []
            for agent, obs in zip(other_agents, full_ob):
                if self.check_observability(nav, agent):
                    partial_ob.append(obs)
            return partial_ob
        else:
            raise ValueError("Observability should be full or partial")

    def get_next_observation(self, nav, human_actions):
        if nav.sensor != 'coordinates':
            raise ValueError("Agent sensor not implemented")

        agents = self.humans + [self.robot]
        other_agents = [x for x in agents if x != nav]
        full_ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]

        if nav.observability == 'full':
            return full_ob
        elif nav.observability == 'partial':
            partial_ob = []
            for agent, obs in zip(other_agents, full_ob):
                if self.check_observability(nav, agent):
                    partial_ob.append(obs)
            return partial_ob
        else:
            raise ValueError("Observability should be full or partial")

    def reset(self, phase='test', test_case=None, max_epsilon=0):
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        # Set random seed
        if test_case is not None:
            self.case_counter[phase] = test_case
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}
        seed = counter_offset[phase] + self.case_counter[phase]
        np.random.seed(seed)

        # Calculate open space based on room size
        self.open_space = self.room_dims[0] * self.room_dims[1]

        # Set robot initial position and goal
        px, py = 0, -(self.room_dims[1]/2 - 1)
        gx, gy = 0, (self.room_dims[1]/2 - 1)
        self.robot.set(px, py, gx, gy, 0, 0, 0)

        # Get initial distance of robot to goal
        self.previous_distance = la.norm([(self.robot.px - self.robot.get_goal_position()[0]),
                                                (self.robot.py - self.robot.get_goal_position()[1])])

        # Set human and object positions
        if self.random_human_num:
            human_num = np.random.randint(self.min_human_num, self.max_human_num+1)
        elif self.random_human_dens:
            min_human_num = int(self.min_human_dens * self.open_space)
            max_human_num = int(self.max_human_dens * self.open_space)
            human_num = np.random.randint(min_human_num, max_human_num+1)
        else:
            human_num = self.num_humans

        # Generate humans
        scenarios  = ['circle', 'perpedendicular', 'opposite', 'same', 'random', 'random']
        perpetuals = [False, False, False, False, False, True]
        if self.randomness:
            idx = np.random.randint(0,6)
            self.scenario = scenarios[idx]
            self.perpetual = perpetuals[idx]
        if phase == 'test':
            logging.info("Generating environment with {} humans".format(human_num))
        self.generate_humans(human_num, self.scenario, max_epsilon, seed)

        # Initialize time
        self.global_time = 0
        self.human_times = [0] * len(self.humans)
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step
        self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]

        # Initialize visualization lists
        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # Get current observation
        return self.get_observation(self.robot)

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        human_actions = []
        for human in self.humans:
            if human.reached_destination() and self.perpetual:
                while True:
                    gx, gy, collide = self.choose_random_goal(human, human.px, human.py)
                    if not collide:
                        break
                human.set_goal_position([gx, gy])
            ob = self.get_observation(human)
            human_actions.append(human.act(ob))

        # Check for collisions between robot and humans
        dmin_human = float('inf')
        collision_human = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < dmin_human:
                dmin_human = closest_dist
            if closest_dist < 0:
                collision_human = True
                human_collided_with = human
                break
        self.dmin = dmin_human

        # Check if robot reached the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = la.norm(end_position - np.array(self.robot.get_goal_position())) < \
                                    (self.robot.radius + self.goal_radius)

        # Compute the reward and check if episode ended
        done = False
        info = Nothing()
        reward = -self.time_penalty
        goal_distance = la.norm([(end_position[0] - self.robot.get_goal_position()[0]),
                                        (end_position[1] - self.robot.get_goal_position()[1])])
        progress = self.previous_distance - goal_distance
        self.previous_distance = goal_distance
        reward += self.progress_reward * progress

        if reaching_goal:
            reward += self.success_reward
            done = True
            info = ReachGoal()
        elif self.global_time >= self.time_limit - 1:
            done = True
            info = Timeout()
        elif collision_human:
            reward += self.collision_penalty
            done = self.end_on_collision
            info = HumanCollision(self.robot.get_position(), human_collided_with)
        elif dmin_human < self.discomfort_dist:
            reward += (dmin_human - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            info = Danger(dmin_human)

        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            ob = self.get_observation(self.robot)
        else:
            ob = self.get_next_observation(self.robot, human_actions)

        return ob, reward, done, info

    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'blue'
        human_color = 'green'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-self.room_dims[0]/2, self.room_dims[0]/2)
            ax.set_ylim(-self.room_dims[1]/2, self.room_dims[1]/2)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=True, color=human_color)
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color=robot_color))
            plt.show()

        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.room_dims[0]/2 - 2, self.room_dims[0]/2 + 2)
            ax.set_ylim(-self.room_dims[1]/2 - 2, self.room_dims[1]/2 + 2)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                                for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=True, color=human_color)
                                for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(len(self.humans) + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(len(self.humans))]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.show()

        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_aspect('equal')
            ax.set_xlim(-self.room_dims[0]/2 - 2, self.room_dims[0]/2 + 2)
            ax.set_ylim(-self.room_dims[1]/2 - 2, self.room_dims[1]/2 + 2)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=True, color=human_color)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.gcf().text(0.45, 0.85, 'Time: {}'.format(0), fontsize=10)
            ax.add_artist(time)

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                state[0].py + radius * np.sin(state[0].theta))) for state in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(len(self.humans) + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]

                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                if self.robot.kinematics == 'holonomic':
                    # when any key is pressed draw the action value plot
                    fig, axis = plt.subplots()
                    speeds = [0] + self.robot.policy.speeds
                    rotations = np.append(self.robot.policy.rotations, [np.pi * 2])
                    r, th = np.meshgrid(speeds, rotations)
                    z = np.array(self.action_values[global_step % len(self.states)][1:])
                    z = np.ma.array(z, mask=np.isnan(z))
                    z = (z - np.min(z)) / (np.max(z) - np.min(z))
                    z = np.reshape(z, (16, 5))
                    polar = plt.subplot(projection="polar")
                    polar.tick_params(labelsize=16)
                    mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                    plt.plot(rotations, r, color='k', ls='none')
                    plt.grid()
                    cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                    cbar = plt.colorbar(mesh, cax=cbaxes)
                    cbar.ax.tick_params(labelsize=16)
                    plt.show()
                else:
                    print("Unable to visualize action space for non-holonomic kinematics")

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
