import logging
import numpy as np
import numpy.linalg as la

MAX_TRIES = 1000000

class GenerateHumans():
    def __init__(self, room_dims, discomfort_dist, goal_radius, seed):
        np.random.seed(seed)
        self.room_dims = room_dims
        self.goal_radius = goal_radius
        self.discomfort_dist = 0.1

    def generate_circle_human(self, human, robot, humans):
        # set radius of circle
        circle_radius = np.minimum(self.room_dims[0], self.room_dims[1]) / 2 - 1

        # choose initial human position on circumference of circle
        count, collide = 0, True
        while collide == True:
            angle = np.random.random() * np.pi * 2
            px_noise = np.random.random() - 0.5
            py_noise = np.random.random() - 0.5
            px = circle_radius * np.cos(angle) + px_noise
            py = circle_radius * np.sin(angle) + py_noise
            collide = False
            if not collide:
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if la.norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        count += 1
                        break
            if count > MAX_TRIES:
                logging.info('Unable to place human in {} tries'.format(MAX_TRIES))
                return None
            if not collide:
                break

        # choose human goal to be opposite of initial position on circle
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_perpendicular_human(self, human, robot, humans):
        # choose initial human position on side of room
        sign = 1 if np.random.random() > 0.5 else -1
        count, collide = 0, True
        while collide == True:
            px_noise = np.random.random() - 0.5
            py_noise = np.random.random() - 0.5
            px = (self.room_dims[0]/2 - 1) * sign + px_noise
            py = (np.random.random() - 0.5) * (self.room_dims[1] - 2) + py_noise
            collide = False
            if not collide:
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if la.norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        count += 1
                        break
            if count > MAX_TRIES:
                logging.info('Unable to place human in {} tries'.format(MAX_TRIES))
                return None
            if not collide:
                break

        # choose human goal position on opposite side of room
        count, collide = 0, True
        while collide == True:
            gx_noise = np.random.random() - 0.5
            gy_noise = np.random.random() - 0.5
            gx = (self.room_dims[0]/2 - 1) * -sign + gx_noise
            gy = (np.random.random() - 0.5) * (self.room_dims[1] - 2) + gy_noise
            collide = False
            if not collide:
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if la.norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        count += 1
                        break
            if count > MAX_TRIES:
                logging.info('Unable to place human in {} tries'.format(MAX_TRIES))
                return None
            if not collide:
                break

        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_opposite_human(self, human, robot, humans):
        # choose initial human position at top of room
        count, collide = 0, True
        while collide == True:
            px_noise = np.random.random() - 0.5
            py_noise = np.random.random() - 0.5
            px = (np.random.random() - 0.5) * (self.room_dims[0] - 2) + px_noise
            py = self.room_dims[1]/2 - 1 + py_noise
            collide = False
            if not collide:
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if la.norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        count += 1
                        break
            if count > MAX_TRIES:
                logging.info('Unable to place human in {} tries'.format(MAX_TRIES))
                return None
            if not collide:
                break

        # choose human goal position at bottom of room
        count, collide = 0, True
        while collide == True:
            gx_noise = np.random.random() - 0.5
            gy_noise = np.random.random() - 0.5
            gx = (np.random.random() - 0.5) * (self.room_dims[0] - 2) + gx_noise
            gy = -(self.room_dims[1]/2 - 1) + gy_noise
            collide = False
            if not collide:
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if la.norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        count += 1
                        break
            if count > MAX_TRIES:
                logging.info('Unable to place human in {} tries'.format(MAX_TRIES))
                return None
            if not collide:
                break

        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_same_human(self, human, robot, humans):
        # choose initial human position at bottom of room
        count, collide = 0, True
        while collide == True:
            px_noise = np.random.random() - 0.5
            py_noise = np.random.random() - 0.5
            px = (np.random.random() - 0.5) * (self.room_dims[0] - 2) + px_noise
            py = -(self.room_dims[1]/2 - 1) + py_noise
            collide = False
            if not collide:
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if la.norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        count += 1
                        break
            if count > MAX_TRIES:
                logging.info('Unable to place human in {} tries'.format(MAX_TRIES))
                return None
            if not collide:
                break

        # choose human goal position at top of room
        count, collide = 0, True
        while collide == True:
            gx_noise = np.random.random() - 0.5
            gy_noise = np.random.random() - 0.5
            gx = (np.random.random() - 0.5) * (self.room_dims[0] - 2) + gx_noise
            gy = self.room_dims[1]/2 - 1 + gy_noise
            collide = False
            if not collide:
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if la.norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        count += 1
                        break
            if count > MAX_TRIES:
                logging.info('Unable to place human in {} tries'.format(MAX_TRIES))
                return None
            if not collide:
                break

        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_random_human(self, human, robot, humans):
        # choose random initial human position
        count, collide = 0, True
        while collide == True:
            px = np.random.random() * self.room_dims[0] - self.room_dims[0]/2
            py = np.random.random() * self.room_dims[1] - self.room_dims[1]/2
            collide = False
            if not collide:
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if la.norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        count += 1
                        break
            if count > MAX_TRIES:
                logging.info('Unable to place human in {} tries'.format(MAX_TRIES))
                return None
            if not collide:
                break

        # choose random human goal position
        count, collide = 0, True
        while collide == True:
            gx = np.random.random() * self.room_dims[0] - self.room_dims[0]/2
            gy = np.random.random() * self.room_dims[1] - self.room_dims[1]/2
            collide = False
            if not collide:
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if la.norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        count += 1
                        break
            if count > MAX_TRIES:
                logging.info('Unable to place human in {} tries'.format(MAX_TRIES))
                return None

        human.set(px, py, gx, gy, 0, 0, 0)
        return human
