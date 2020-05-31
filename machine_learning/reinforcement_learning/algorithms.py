import tqdm
import logging
import json
from toolz import dicttoolz
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FINISH_TOKEN = "FINISHED"
MOVING_TOKEN = "MOVING"


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
