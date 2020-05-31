import numpy as np
from itertools import product
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()

OOB_TOKEN = "#"
FINAL_TOKEN = "F"
START_TOKEN = "S"

FINISH_STATUS = "FINISHED"
REVERTED_STATUS = "REVERTED"
MOVED_STATUS = "MOVED"


class Car:
    """
    Basic class to handle the
    operations of the racecar.
    """

    def __init__(self, x, y, x_speed=0, y_speed=0):
        """
        Parameters
        ----------
        x : int
            The starting X position of the car

        y : int
            The starting y position of the car

        x_speed  : int
            The starting X velocity of the car

        y_speed : int
            The starting y velocity of the car
        """
        # Set the starting point
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
        """
        Set the accelration of the car
        """
        self.x_accel = x_accel
        self.y_accel = y_accel

    def limit_speed(self):
        """
        Caps the speed at 5
        """
        if self.x_speed > 5:
            self.x_speed = 5
        if self.y_speed > 5:
            self.y_speed = 5

        if self.x_speed < -5:
            self.x_speed = -5
        if self.y_speed < -5:
            self.y_speed = -5

    def take_step(self):
        """
        Take a step on the track
        """
        # Randomly choose if acceleration will be random
        accel_succeeds = np.random.uniform(0, 1, size=2) > 0.2

        # Set the speed based on the acceleration if success chosen.
        self.x_speed += self.x_accel if accel_succeeds[0] else 0
        self.y_speed += self.y_accel if accel_succeeds[1] else 0

        # Ensure the speed is less than 5
        self.limit_speed()

        # Track the previous state so we can revert it if we need to
        self.last_x = self.x
        self.last_y = self.y

        # Take the step
        self.x += self.x_speed
        self.y += self.y_speed

    def revert_step(self):
        """
        If the car goes OOB, revert the step and set speed to zero
        """
        self.x_speed = 0
        self.y_speed = 0
        self.x = self.last_x
        self.y = self.last_y

    def back_to_start(self):
        """
        If the car goes OOB, send it back to the starting place.
        """
        self.x, self.y = self.starting_point
        self.x_speed = 0
        self.y_speed = 0

    def get_location(self):
        """
        Get the current location tuple.
        """
        return (self.x, self.y)

    def get_last_location(self):
        """
        Get the previous location
        """
        return self.last_x, self.last_y

    def get_actions(self):
        """
        Get all possible actions
        """
        return list(product([-1, 0, 1], [-1, 0, 1]))

    def get_state(self):
        """
        Get the current state (location + speed)
        """
        return self.get_location() + (self.x_speed, self.y_speed)


class Track:
    """
    Class to handle the functions of the track.
    """

    def __init__(self, filename, harsh_crash_variant=False):
        """
        Parameters:
        -----------
        filename : str
            Filename of the track

        harsh_crash_variant : bool
            If True, car is set to starting line
            on crash. If False, last step is reverted and
            velo is set to zero.
        """
        # Read in the file
        f = open(filename)

        # Parse the size and the track
        self.size = tuple(map(int, f.readline().strip().split(",")))
        self.track = list(
            np.array(list(map(lambda x: list(x.strip()), f.readlines()))).T
        )

        # Set crash variant
        self.harsh_crash_variant = harsh_crash_variant

        # Record all of the starting points
        self.starting_points = [
            (j, i)
            for i in range(self.size[0])
            for j in range(self.size[1])
            if self.track[j][i] == START_TOKEN
        ]

        # Record all of the finish points
        self.finish_points = [
            (j, i)
            for i in range(self.size[0])
            for j in range(self.size[1])
            if self.track[j][i] == FINAL_TOKEN
        ]

        # Record all of the boundary points
        self.oob_points = [
            (j, i)
            for i in range(self.size[0])
            for j in range(self.size[1])
            if self.track[j][i] == OOB_TOKEN
        ]

        f.close()
        self.states = []

    def get_states(self):
        """
        Get all of the possible env states
        """
        # If not yet created, set the states.
        if not self.states:
            # For each row and each column...
            for row in range(self.size[1]):
                for col in range(self.size[0]):
                    # If a valid location, add it to states tracker
                    if self.track[row][col] not in [OOB_TOKEN, FINAL_TOKEN]:
                        # Add all speeds for the location
                        speeds = product(range(-5, 6), range(-5, 6))
                        for speed in speeds:
                            self.states.append((row, col) + speed)
        return self.states

    def get_current_state(self):
        """
        Get the current state of the agent
        """
        return self.agent.get_state()

    def start_track(self, starting_point=None):
        """
        Initialize the track. Will choose
        random starting point if none is passed

        Parameters:
        -----------
        starting_point : int
            The index of the starting point to
            use. If none is passed,
            randomly chosen
                """
        if not starting_point:
            # Choose random starting point
            starting_point = np.random.choice(
                np.arange(len(self.starting_points)), size=1
            )[0]

        # Initialize car
        self.agent = Car(*self.starting_points[starting_point])

    def get_actions(self):
        """
        Return the actions
        """
        return self.agent.get_actions()

    def get_next_state(self, state, action):
        """
        Given a current state and an action,
        get the next state by applying the step.
        """
        # Put the car at the state
        self.agent = Car(*state)

        # Set the acceleration based on the action
        self.agent.set_acceleration(*action)

        # Take the step
        self.agent.take_step()

        # Make sure still in bounds
        status = self.check_agent_location()

        # Return the status token and the new state
        return status, self.agent.get_state()

    def is_off_board(self):
        """
        Check if car has gone off the board
        """
        if (self.agent.x >= len(self.track) or self.agent.x < 0) or (
            self.agent.y >= len(self.track[0]) or self.agent.y < 0
        ):
            return True
        return False

    @staticmethod
    def between_unordered(x, y, target):
        """
        Helper to determine if target is between
        x and y
        """
        if (x >= target and target >= y) or (y >= target and target >= x):
            return True
        return False

    def check_agent_cross_finish(self):
        """
        Check if agent cross the finish line
        """
        # Loop through finish tokens
        for finish_state in self.finish_points:
            # If the car cross the x and y of the finish point
            # Return True and location
            cross_x = Track.between_unordered(
                self.agent.x, self.agent.last_x, finish_state[0]
            )
            cross_y = Track.between_unordered(
                self.agent.y, self.agent.last_y, finish_state[1]
            )
            if cross_x and cross_y:
                return True, cross_x, cross_y

        # Otherwise False
        return False, None, None

    def check_agent_cross_oob(self):
        """
        Check if agent went out of bounds
        """
        # Loop thru OOB points
        for oob in self.oob_points:
            # If the car cross the x and y of the OOB point
            # Return True and location
            cross_x = Track.between_unordered(self.agent.x, self.agent.last_x, oob[0])
            cross_y = Track.between_unordered(self.agent.y, self.agent.last_y, oob[1])
            if cross_x and cross_y:
                return True, cross_x, cross_y
        return False, None, None

    def check_agent_location(self):
        """
        Make sure agent is in a valid spot and hasn't
        gone out of bounds finished the track
        """
        # Check if finish line was crossed
        crossed_finished, finish_x, finish_y = self.check_agent_cross_finish()
        # Check if OOB was crossed
        crossed_oob, cross_x, cross_y = self.check_agent_cross_oob()

        # If both were crossed, check if finish line
        # was crossed first or not
        if crossed_oob and crossed_finished:
            if self.agent.last_x < finish_x and finish_x < cross_x:
                if self.agent.last_y < finish_y and finish_y < cross_y:
                    return FINISH_STATUS

                if self.agent.last_y > finish_y and finish_y > cross_y:
                    return FINISH_STATUS

            if self.agent.last_x > finish_x and finish_x > cross_x:
                if self.agent.last_y < finish_y and finish_y < cross_y:
                    return FINISH_STATUS

                if self.agent.last_y > finish_y and finish_y > cross_y:
                    return FINISH_STATUS

        # If finish cross and not OOB, return finished
        if crossed_finished and not crossed_oob:
            self.agent.revert_step()
            return FINISH_STATUS

        # If off the board, enforce crash rules
        if self.is_off_board():
            if self.harsh_crash_variant:
                self.agent.back_to_start()
            else:
                self.agent.revert_step()
            return REVERTED_STATUS

        spot = self.track[self.agent.x][self.agent.y]

        # If OOB or crossed OOB, enforce crash rules
        if spot == OOB_TOKEN or crossed_oob:
            if self.harsh_crash_variant:
                self.agent.back_to_start()
            else:
                self.agent.revert_step()
            return REVERTED_STATUS

        # If finished, return finish
        if spot == FINAL_TOKEN:
            return FINISH_STATUS

        # Otherwise, regular move
        return MOVED_STATUS
