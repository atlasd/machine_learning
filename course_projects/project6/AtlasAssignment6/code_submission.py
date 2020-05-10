from joblib import Parallel, delayed
import tqdm
import logging
import json
from toolz import dicttoolz
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
FINISH_TOKEN = "FINISHED"
MOVING_TOKEN = "MOVING"

"""
This section defines the track and the car
"""


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


"""
This section defines the policy class
"""
import numpy as np
import random


class Policy:
    """
    Class to handle basic interactions of the
    policy
    """

    def __init__(self, actions, states):
        """
        Parameters:
        ----------
        actions: list
            List of possible actions for each state

        states : list
            List of environment states
        """
        # Initialize policy to 1 for all state action pairs
        self.policy = {state: {action: 1 for action in actions} for state in states}
        self.actions = actions
        self.states = states
        self.exploit_mode = False

    def get_optimal_action(self, state):
        """
        Get the optimal action via the policy

        Parameters:
        -----------
        state : tuple
            A tuple representing a state
        """
        # Get the policy for the state
        action_dict = self.policy.get(state)

        # Shuffle the list to get random choice for tie
        action_list = list(action_dict.items())
        random.shuffle(action_list)

        # Get the smallest action of the set
        actions = sorted(
            dict(action_list), key=self.policy.get(state).get, reverse=False
        )
        return actions[0]

    def get_optimal_action_eps_greedy(self, state, eps):
        """
        Get the action via eps greedy policy

        Parameters:
        -----------
        state : tuple
            Tuple representing the state

        eps : float
            The eps value for the eps-greedy selection
        """
        # Choose a random value from Unif[0, 1]
        random_value = np.random.uniform(0, 1, size=1)[0]

        # If random value < eps, random action
        if random_value <= eps:
            actions = list(self.policy.get(state).keys())
            random.shuffle(actions)
            return actions[0]

        # Otherwise, get the optimal action
        return self.get_optimal_action(state)

    def get_optimal_action_value(self, state):
        """
        Function to get the optimal action for a state
        based on the policy
        """
        optimal_action = self.get_optimal_action(state)
        return self.policy[state][optimal_action]

    def update_policy(self, state, action, new_term):
        """
        Add a new term to the current value estimation for the policy
        """
        self.policy[state][action] += new_term


"""
This section contains the algorithms code
"""


class BasePolicy:
    """
    Class of shared functionality for
    all the RL algorithms
    """

    def __init__(
        self,
        policy,
        max_iter,
        actor,
        discount_factor,
        stat_collection_freq=100,
        fname=None,
        max_exploit=1000,
    ):
        """
        Parameters:
        -----------
        policy : Policy
            A policy object

        max_iter : int
            The maximum number of training iterations
            to run.

        actor : Track
            A track class to solve

        discount_factor : float
            Discount factor for the learning process

        stat_collection_freq : int
            The number of iterations to collect
            performance over

        fname : str
            The filename to dump the results to.

        max_exploit : int
            The maximum number of iterations to solve the track
        """
        self.policy = policy
        self.actor = actor
        self.max_iter = max_iter
        self.discount_factor = discount_factor
        self.stats = {}
        self.stat_collection_freq = stat_collection_freq
        self.fname = fname
        self.max_exploit = max_exploit

        # Initialize the track
        self.actor.start_track()

    def single_iteration(self):
        """
        Placeholder for a single learning iteration
        """
        raise NotImplementedError

    def explore(self):
        """
        Function to conduct the exploration process
        """
        # Conduct max_iter iterations
        for iter_num in tqdm.tqdm(range(self.max_iter)):
            # If stat_collection_freq, run an exploit
            if (iter_num + 1) % self.stat_collection_freq == 0:
                logger.info("Collecting stats...")
                self.collect_stats(iter_num)

            self.single_iteration()

        self.collect_stats(iter_num)

        # Dump the results to file
        if self.fname:
            json.dump(
                {
                    "policy": dicttoolz.keymap(
                        str,
                        dicttoolz.valmap(
                            lambda d: dicttoolz.keymap(str, d), self.policy.policy,
                        ),
                    ),
                    "stats": self.stats,
                },
                open(self.fname, "w"),
            )
        logger.info(msg=self.stats)

    def load_from_file(self, filename):
        """
        Function to load policy from file
        """
        from_file = json.load(open(filename))
        self.policy.policy = dicttoolz.valmap(
            lambda d: dicttoolz.keymap(eval, d),
            dicttoolz.keymap(eval, from_file.get("policy")),
        )
        self.stats = from_file.get("stats")

    def exploit(self, starting_point=None):
        """
        Run an exploitation iteration.


        Parameters:
        -----------
        starting_point : int
            Index of starting point to use.
        """
        # Initialize track
        self.actor.start_track(starting_point=starting_point)

        # Initialize trackers
        is_finished = "MOVED"
        n_steps = 0
        path = []

        # While not finished
        while is_finished != FINISH_TOKEN and n_steps < self.max_exploit:
            # Get the current state
            current_state = self.actor.get_current_state()

            # Add to path
            path.append(current_state)

            # Get the next action
            action = self.policy.get_optimal_action(current_state)

            # Do action and take step
            self.actor.agent.set_acceleration(*action)
            self.actor.agent.take_step()

            # Check the location to see if finished
            is_finished = self.actor.check_agent_location()
            n_steps += 1

        # Return the number of steps and the path taken
        return (path, n_steps)

    def collect_stats(self, iter_number):
        """
        Function to collect the number of
        steps until completion to see the
        learning trace.

        Parameters:
        -----------
        iter_number : int
            Iteration of colleciton
        """
        # Exploit from each starting point ten time
        self.stats[iter_number] = [
            self.exploit(start)[1]
            for _ in range(10)
            for start in np.arange(len(self.actor.starting_points))
        ]


class ValueIteration(BasePolicy):
    """
    Implementation of ValueIteration policy
    """

    def __init__(
        self,
        policy,
        max_iter,
        actor,
        discount_factor=0.1,
        stat_collection_freq=10,
        **kwargs,
    ):
        """
        Parameters:
        -----------
        policy : Policy
            A policy object

        max_iter : int
            The maximum number of training iterations
            to run.

        actor : Track
            A track class to solve

        discount_factor : float
            Discount factor for the learning process

        stat_collection_freq : int
            The number of iterations to collect
            performance over

        fname : str
            The filename to dump the results to.

        **kwargs
            To pass to parent class
        """
        super().__init__(
            policy=policy,
            max_iter=max_iter,
            actor=actor,
            discount_factor=discount_factor,
            stat_collection_freq=stat_collection_freq,
            **kwargs,
        )

    def __str__(self):
        return "ValueIteration"

    def single_iteration(self):
        """
        Update the whole policy using dynamic programming.
        Function conducts a single update
        """
        # Iterate through states
        for state in self.policy.states:
            status = None
            # Iterate through actions in each state
            for action, value in self.policy.policy[state].items():
                # Record the next state
                status, state_prime = self.actor.get_next_state(
                    state=state, action=action
                )

                # If finished, add zero to the policy
                if status == FINISH_TOKEN:
                    self.policy.update_policy(state=state, action=action, new_term=0)

                # Otherwise, update with approx value of next state
                else:
                    next_val = self.policy.get_optimal_action_value(state_prime)
                    self.policy.update_policy(
                        state=state,
                        action=action,
                        new_term=self.discount_factor * next_val,
                    )


class QLearning(BasePolicy):
    """
    Class to conduct QLearning
    """

    def __init__(
        self, policy, max_iter, actor, discount_factor, learning_rate=0.1, **kwargs
    ):
        """
        Parameters:
        -----------
        policy : Policy
            A policy object

        max_iter : int
            The maximum number of training iterations
            to run.

        actor : Track
            A track class to solve

        discount_factor : float
            Discount factor for the learning process

        learning_rate : float
            Learning rate for updates

        **kwargs
            To pass to parent class
        """
        super().__init__(
            policy=policy,
            max_iter=max_iter,
            actor=actor,
            discount_factor=discount_factor,
            stat_collection_freq=10000,
            **kwargs,
        )
        self.learning_rate = learning_rate

    def __str__(self):
        return "QLearning"

    def single_iteration(self):
        # Initialize track
        status = MOVING_TOKEN
        n_iter = 0
        self.actor.start_track()

        # While not finished, explore
        while status != FINISH_TOKEN:
            n_iter += 1

            # Make epsilon greedy update
            eps = max(1 / n_iter, 0.05)

            # Get the current state
            state = self.actor.get_current_state()

            # Get the eps greedy action choice
            action = self.policy.get_optimal_action_eps_greedy(state=state, eps=eps)

            # Get sprime (and go to that state)
            status, state_prime = self.actor.get_next_state(state=state, action=action)

            # Get the approx value of that state
            sprime_qstar = self.policy.get_optimal_action_value(state=state_prime)

            # Get current state estimate
            s_qstar = self.policy.policy[state][action]

            # Update policy by temporal difference
            self.policy.update_policy(
                state=state,
                action=action,
                new_term=(
                    self.learning_rate
                    * (1 + self.discount_factor * sprime_qstar - s_qstar)
                ),
            )
        logger.info(f"Number of Steps to Complete: {n_iter}")


class SARSA(BasePolicy):
    """
    Class to conduct SARSA exploration
    """

    def __init__(
        self, policy, max_iter, actor, discount_factor, learning_rate=0.1, **kwargs
    ):
        """
        Parameters:
        -----------
        policy : Policy
            A policy object

        max_iter : int
            The maximum number of training iterations
            to run.

        actor : Track
            A track class to solve

        discount_factor : float
            Discount factor for the learning process

        learning_rate : float
            Learning rate for updates

        **kwargs
            To pass to parent class
        """
        super().__init__(
            policy=policy,
            max_iter=max_iter,
            actor=actor,
            discount_factor=discount_factor,
            stat_collection_freq=1000,
            **kwargs,
        )
        self.learning_rate = learning_rate

    def __str__(self):
        return "SARSA"

    def single_iteration(self):
        """
        Conduct single exploration experiment
        """
        status = MOVING_TOKEN
        n_iter = 0
        self.actor.start_track()

        while status != FINISH_TOKEN:
            n_iter += 1
            # Get eps greedy propbability
            eps = max(1 / n_iter, 0.05)

            # Get current state
            state = self.actor.get_current_state()

            # Get the action via eps-greedy selection
            action = self.policy.get_optimal_action_eps_greedy(state=state, eps=eps)

            # Get the next state (and move agent there)
            status, state_prime = self.actor.get_next_state(state=state, action=action)

            # Get the eps-greedy best action for that state
            action_prime = self.policy.get_optimal_action_eps_greedy(
                state=state_prime, eps=eps
            )

            # Get the state that results from that action
            sprime_aprime_q = self.policy.policy[state_prime][action_prime]

            # Get the current Q-value
            s_qstar = self.policy.policy[state][action]

            # Update the policy via the difference
            self.policy.update_policy(
                state=state,
                action=action,
                new_term=self.learning_rate
                * (1 + self.discount_factor * sprime_aprime_q - s_qstar),
            )
        logger.info(f"Number of steps: {n_iter}")


"""
This section defines all of the variants to run and their 
hyperparameters.
"""

DIR = "/Users/home/Documents/JHU/machine_learning/course_projects/project6"
TRACKS = ["L-track.txt", "O-track.txt", "R-track.txt"]
ALGOS = [ValueIteration, QLearning, SARSA]
HARSH_CRASH = [True, False]
ALGO_KWARGS = {
    ("L-track.txt", ValueIteration, True): dict(discount_factor=0.7, max_iter=250,),
    ("L-track.txt", ValueIteration, False): dict(discount_factor=0.8, max_iter=100,),
    ("L-track.txt", QLearning, True): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=100000,
    ),
    ("L-track.txt", QLearning, False): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=40000,
    ),
    ("L-track.txt", SARSA, True): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=125000,
    ),
    ("L-track.txt", SARSA, False): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=4000,
    ),
    ("O-track.txt", QLearning, True): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=175000,
    ),
    ("O-track.txt", QLearning, False): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=75000,
    ),
    ("O-track.txt", ValueIteration, True): dict(discount_factor=0.1, max_iter=500,),
    ("O-track.txt", ValueIteration, False): dict(discount_factor=0.1, max_iter=600,),
    ("O-track.txt", SARSA, True): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=200000,
    ),
    ("O-track.txt", SARSA, False): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=75000
    ),
    ("R-track.txt", ValueIteration, True): dict(max_iter=1500, discount_factor=0.001),
    ("R-track.txt", ValueIteration, False): dict(max_iter=1500, discount_factor=0.001),
    ("R-track.txt", QLearning, True): dict(
        discount_factor=0.3, learning_rate=0.1, max_iter=50000,
    ),
    ("R-track.txt", QLearning, False): dict(
        discount_factor=0.3, learning_rate=0.1, max_iter=7500,
    ),
    ("R-track.txt", SARSA, True): dict(
        discount_factor=0.3, learning_rate=0.1, max_iter=75000
    ),
    ("R-track.txt", SARSA, False): dict(
        discount_factor=0.3, learning_rate=0.1, max_iter=75000,
    ),
}


for track, algo, crash_variant in product(TRACKS, ALGOS, HARSH_CRASH):
    ALGO_KWARGS[(track, algo, crash_variant)].update(
        {"fname": f"{track}_{algo}_{crash_variant}"}
    )


def run_experiment(track, algo, crash_variant, kwargs):
    """
    Runs an experiment for a given track, algorithm and
    crash variant
    """
    np.random.seed(73)
    track_obj = Track(f"{DIR}/{track}", harsh_crash_variant=crash_variant)
    track_obj.start_track()
    logger.info(f"Initializing track {track} with harsh_variant={crash_variant}")

    ALGO_KWARGS.update(
        {"fname": f"{track}_{algo}_{crash_variant}_{kwargs.get('discount_factor')}"}
    )

    # Create policy
    pi = Policy(states=track_obj.get_states(), actions=track_obj.get_actions())

    # Initialize algorithm
    algo_object = algo(policy=pi, actor=track_obj, **kwargs)

    logger.info(f"Updating policy via {algo}...")
    # Run the exploration
    algo_object.explore()

    # Exploit to determine effectiveness of algorithm
    logger.info(f"Avg Steps to Solve:")
    logger.info(
        np.mean(
            [
                algo_object.exploit(i)[1]
                for i in range(len(algo_object.actor.starting_points))
                for _ in range(10)
            ]
        )
    )


if __name__ == "__main__":
    # Run all of the experiments
    results = Parallel(n_jobs=-1)(
        delayed(run_experiment)(
            track=track,
            algo=algo,
            crash_variant=crash_variant,
            kwargs=dict(
                ALGO_KWARGS.get((track, algo, crash_variant)), max_exploit=1000
            ),
        )
        for track, algo, crash_variant in product(TRACKS, ALGOS, HARSH_CRASH)
    )
