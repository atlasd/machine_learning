from machine_learning.reinforcement_learning import actors, policy
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FINISH_TOKEN = "FINISHED"
MOVING_TOKEN = "MOVING"


class BasePolicy:
    def __init__(
        self, policy, max_iter, actor, discount_factor, stat_collection_freq=100
    ):
        self.policy = policy
        self.actor = actor
        self.max_iter = max_iter
        self.discount_factor = discount_factor
        self.actor.start_track()
        self.stats = {}
        self.stat_collection_freq = stat_collection_freq

    def single_iteration(self):
        raise NotImplementedError

    def explore(self):
        for iter_num in tqdm.tqdm(range(self.max_iter)):
            if (iter_num + 1) % self.stat_collection_freq == 0:
                logger.infp("Collecting stats...")
                self.collect_stats(iter_num)
            self.single_iteration()
        self.collect_stats(iter_num)
        logger.info(msg=self.stats)

    def exploit(self):
        self.actor.start_track()
        is_finished = "MOVED"
        n_steps = 0
        path = []
        while is_finished != FINISH_TOKEN and n_steps < 1000:
            current_state = self.actor.get_current_state()
            path.append(current_state)
            action = self.policy.get_optimal_action(current_state)
            self.actor.agent.set_acceleration(*action)
            self.actor.agent.take_step()
            is_finished = self.actor.check_agent_location()
            n_steps += 1
        return (path, n_steps)

    def collect_stats(self, iter_number):
        self.stats[iter_number] = [self.exploit()[1] for _ in range(10)]
        logger.info(f"Exploit Stats - Iteration number: {iter_number}")
        logger.info(self.stats[iter_number])


class ValueIteration(BasePolicy):
    def __init__(self, policy, max_iter, actor, discount_factor=0.1):
        super().__init__(
            policy=policy,
            max_iter=max_iter,
            actor=actor,
            discount_factor=discount_factor,
            stat_collection_freq=3,
        )

    def single_iteration(self):
        for state in self.policy.states:
            status = None
            for action, value in self.policy.policy[state].items():
                status, state_prime = self.actor.get_next_state(
                    state=state, action=action
                )
                if status == FINISH_TOKEN:
                    self.policy.update_policy(state=state, action=action, new_term=0)

                else:
                    next_val = self.policy.get_optimal_action_value(state_prime)
                    self.policy.update_policy(
                        state=state,
                        action=action,
                        new_term=self.discount_factor * next_val,
                    )

            if status == FINISH_TOKEN:
                break


class QLearning(BasePolicy):
    def __init__(self, policy, max_iter, actor, discount_factor, learning_rate=0.1):
        super().__init__(
            policy=policy,
            max_iter=max_iter,
            actor=actor,
            discount_factor=discount_factor,
            stat_collection_freq=1000,
        )
        self.learning_rate = learning_rate

    def single_iteration(self):
        status = MOVING_TOKEN
        n_iter = 0
        self.actor.start_track()

        while status != FINISH_TOKEN:
            n_iter += 1
            eps = max(1 / n_iter, 0.2)
            state = self.actor.get_current_state()

            action = self.policy.get_optimal_action_eps_greedy(state=state, eps=eps)

            status, state_prime = self.actor.get_next_state(state=state, action=action)

            sprime_qstar = self.policy.get_optimal_action_value(state=state_prime)

            s_qstar = self.policy.policy[state][action]

            self.policy.update_policy(
                state=state,
                action=action,
                new_term=self.learning_rate
                * (1 + self.discount_factor * sprime_qstar - s_qstar),
            )


class SARSA(BasePolicy):
    def __init__(self, policy, max_iter, actor, discount_factor, learning_rate=0.1):
        super().__init__(
            policy=policy,
            max_iter=max_iter,
            actor=actor,
            discount_factor=discount_factor,
            stat_collection_freq=1000,
        )
        self.learning_rate = learning_rate

    def single_iteration(self):
        status = MOVING_TOKEN
        n_iter = 0
        self.actor.start_track()

        while status != FINISH_TOKEN:
            n_iter += 1
            eps = max(1 / n_iter, 0.2)
            state = self.actor.get_current_state()
            action = self.policy.get_optimal_action_eps_greedy(state=state, eps=eps)
            status, state_prime = self.actor.get_next_state(state=state, action=action)

            action_prime = self.policy.get_optimal_action_eps_greedy(
                state=state_prime, eps=eps
            )
            sprime_aprime_q = self.policy.policy[state_prime][action_prime]

            s_qstar = self.policy.policy[state][action]

            self.policy.update_policy(
                state=state,
                action=action,
                new_term=self.learning_rate
                * (1 + self.discount_factor * sprime_aprime_q - s_qstar),
            )
