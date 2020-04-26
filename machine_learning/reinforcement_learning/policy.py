import numpy as np
import random


class Policy:
    def __init__(self, actions, states):
        self.policy = {state: {action: 1 for action in actions} for state in states}
        self.actions = actions
        self.states = states
        self.exploit_mode = False

    def set_exploit_mode(self):
        self.exploit_mode = True

    def get_optimal_action(self, state):

        actions = sorted(
            self.policy.get(state), key=self.policy.get(state).get, reverse=False
        )

        return actions[0]

    def get_optimal_action_eps_greedy(self, state, eps):
        random_value = np.random.uniform(0, 1, size=1)[0]
        if random_value <= eps:
            actions = list(self.policy.get(state).keys())
            random.shuffle(actions)

            return actions[0]
        return self.get_optimal_action(state)

    def get_optimal_action_value(self, state):
        optimal_action = self.get_optimal_action(state)
        return self.policy[state][optimal_action]

    def update_policy(self, state, action, new_term):
        self.policy[state][action] += new_term
