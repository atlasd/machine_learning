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
