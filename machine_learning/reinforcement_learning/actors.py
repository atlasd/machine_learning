import numpy as np
from itertools import product
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()

OOB_TOKEN = "#"
FINAL_TOKEN = "F"
START_TOKEN = "S"


class Car:
    def __init__(self, x, y, x_speed=0, y_speed=0):
        self.starting_point = (x, y)

        self.x = x
        self.y = y

        self.last_x = 0
        self.last_y = 0

        self.x_speed = x_speed
        self.y_speed = y_speed

        self.x_accel = 0
        self.y_accel = 0

    def set_acceleration(self, x_accel, y_accel):
        self.x_accel = x_accel
        self.y_accel = y_accel

    def limit_speed(self):
        if self.x_speed > 5:
            self.x_speed = 5
        if self.y_speed > 5:
            self.y_speed = 5

        if self.x_speed < -5:
            self.x_speed = -5
        if self.y_speed < -5:
            self.y_speed = -5

    def take_step(self):
        accel_succeeds = np.random.uniform(0, 1, size=2) > 0.2
        self.x_speed += self.x_accel if accel_succeeds[0] else 0
        self.y_speed += self.y_accel if accel_succeeds[1] else 0
        self.limit_speed()

        self.last_x = self.x
        self.last_y = self.y

        self.x += self.x_speed
        self.y += self.y_speed

    def revert_step(self):
        self.x_speed = 0
        self.y_speed = 0
        self.x = self.last_x
        self.y = self.last_y

    def back_to_start(self):
        self.x, self.y = self.starting_point
        self.x_speed = 0
        self.y_speed = 0

    def get_location(self):
        return (self.x, self.y)

    def get_last_location(self):
        return self.last_x, self.last_y

    def get_actions(self):
        return list(product([-1, 0, 1], [-1, 0, 1]))

    def get_state(self):
        return self.get_location() + (self.x_speed, self.y_speed)


class Track:
    def __init__(self, filename, harsh_crash_variant=False):
        f = open(filename)
        self.size = tuple(map(int, f.readline().strip().split(",")))
        self.track = list(
            np.array(list(map(lambda x: list(x.strip()), f.readlines()))).T
        )
        self.harsh_crash_variant = harsh_crash_variant

        self.starting_points = [
            (j, i)
            for i in range(self.size[0])
            for j in range(self.size[1])
            if self.track[j][i] == START_TOKEN
        ]

        self.finish_points = [
            (j, i)
            for i in range(self.size[0])
            for j in range(self.size[1])
            if self.track[j][i] == FINAL_TOKEN
        ]

        self.oob_points = [
            (j, i)
            for i in range(self.size[0])
            for j in range(self.size[1])
            if self.track[j][i] == OOB_TOKEN
        ]

        f.close()
        self.states = []

    def get_states(self):
        if not self.states:
            for row in range(self.size[1]):
                for col in range(self.size[0]):
                    if self.track[row][col] not in [OOB_TOKEN, FINAL_TOKEN]:
                        speeds = product(range(-5, 6), range(-5, 6))
                        for speed in speeds:
                            self.states.append((row, col) + speed)
        return self.states

    def get_current_state(self):
        return self.agent.get_state()

    def start_track(self):
        logger.info("Initializing track...")
        starting_point = np.random.choice(np.arange(len(self.starting_points)), size=1)[
            0
        ]
        logger.info(f"Car at: {self.starting_points[starting_point]}")
        self.agent = Car(*self.starting_points[starting_point])

    def get_actions(self):
        return self.agent.get_actions()

    def get_next_state(self, state, action):
        self.agent = Car(*state)
        self.agent.set_acceleration(*action)
        self.agent.take_step()

        status = self.check_agent_location()
        return status, self.agent.get_state()

    def is_off_board(self):
        if (self.agent.x >= len(self.track) or self.agent.x < 0) or (
            self.agent.y >= len(self.track[0]) or self.agent.y < 0
        ):
            return True
        return False

    @staticmethod
    def between_unordered(x, y, target):
        if (x >= target and target >= y) or (y >= target and target >= x):
            return True
        return False

    def check_agent_cross_finish(self):
        for finish_state in self.finish_points:
            cross_x = Track.between_unordered(
                self.agent.x, self.agent.last_x, finish_state[0]
            )
            cross_y = Track.between_unordered(
                self.agent.y, self.agent.last_y, finish_state[1]
            )
            if cross_x and cross_y:
                return True
        return False

    def check_agent_cross_oob(self):
        for oob in self.oob_points:
            cross_x = Track.between_unordered(self.agent.x, self.agent.last_x, oob[0])
            cross_y = Track.between_unordered(self.agent.y, self.agent.last_y, oob[1])
            if cross_x and cross_y:
                return True
        return False

    def check_agent_location(self):
        if self.check_agent_cross_finish():
            self.agent.revert_step()
            return "FINISHED"

        if self.is_off_board():
            if self.harsh_crash_variant:
                self.agent.back_to_start()
            else:
                self.agent.revert_step()
            return "REVERTED"

        spot = self.track[self.agent.x][self.agent.y]

        if spot == FINAL_TOKEN:
            return "FINISHED"

        if spot == OOB_TOKEN or self.check_agent_cross_oob():
            if self.harsh_crash_variant:
                self.agent.back_to_start()
            else:
                self.agent.revert_step()
            return "REVERTED"

        return "MOVED"
