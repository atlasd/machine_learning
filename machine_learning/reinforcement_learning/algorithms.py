import tqdm
import logging
import json
from toolz import dicttoolz
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FINISH_TOKEN = "FINISHED"
MOVING_TOKEN = "MOVING"


class BasePolicy:
    def __init__(
        self,
        policy,
        max_iter,
        actor,
        discount_factor,
        stat_collection_freq=100,
        fname=None,
        satisfactory_performance=None,
        max_exploit=1000,
    ):
        self.policy = policy
        self.actor = actor
        self.max_iter = max_iter
        self.discount_factor = discount_factor
        self.actor.start_track()
        self.stats = {}
        self.stat_collection_freq = stat_collection_freq
        self.fname = fname
        self.satisfactory_performance = satisfactory_performance
        self.max_exploit = max_exploit

    def single_iteration(self):
        raise NotImplementedError

    def explore(self):
        for iter_num in tqdm.tqdm(range(self.max_iter)):
            if (iter_num + 1) % self.stat_collection_freq == 0:
                logger.info("Collecting stats...")
                self.collect_stats(iter_num)

            self.single_iteration()

        self.collect_stats(iter_num)
        logger.info(msg=self.stats)
        if self.fname:
            json.dump(
                {
                    "policy": dicttoolz.keymap(
                        str,
                        dicttoolz.valmap(
                            lambda d: dicttoolz.keymap(str, d), self.policy.policy
                        ),
                    ),
                    "stats": self.stats,
                },
                open(self.fname, "w"),
            )

    def load_from_file(self, filename):
        from_file = json.load(open(filename))
        self.policy.policy = dicttoolz.valmap(
            lambda d: dicttoolz.keymap(eval, d),
            dicttoolz.keymap(eval, from_file.get("policy")),
        )
        self.stats = from_file.get("stats")

    def exploit(self, starting_point=None):
        self.actor.start_track(starting_point=starting_point)
        is_finished = "MOVED"
        n_steps = 0
        path = []
        while is_finished != FINISH_TOKEN and n_steps < self.max_exploit:
            current_state = self.actor.get_current_state()
            path.append(current_state)
            action = self.policy.get_optimal_action(current_state)
            self.actor.agent.set_acceleration(*action)
            self.actor.agent.take_step()
            is_finished = self.actor.check_agent_location()
            n_steps += 1
        return (path, n_steps)

    def collect_stats(self, iter_number):
        self.stats[iter_number] = [
            self.exploit(start)[1]
            for _ in range(10)
            for start in np.arange(len(self.actor.starting_points))
        ]


class ValueIteration(BasePolicy):
    def __init__(
        self,
        policy,
        max_iter,
        actor,
        discount_factor=0.1,
        stat_collection_freq=20,
        **kwargs,
    ):
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
        for state in tqdm.tqdm(self.policy.states[::-1]):
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
                        new_term=1 + self.discount_factor * next_val,
                    )


class QLearning(BasePolicy):
    def __init__(
        self, policy, max_iter, actor, discount_factor, learning_rate=0.1, **kwargs
    ):
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
        status = MOVING_TOKEN
        n_iter = 0
        self.actor.start_track()

        while status != FINISH_TOKEN:
            n_iter += 1
            eps = max(1 / n_iter, 0.05)
            state = self.actor.get_current_state()

            action = self.policy.get_optimal_action_eps_greedy(state=state, eps=eps)

            status, state_prime = self.actor.get_next_state(state=state, action=action)

            sprime_qstar = self.policy.get_optimal_action_value(state=state_prime)

            s_qstar = self.policy.policy[state][action]

            self.policy.update_policy(
                state=state,
                action=action,
                new_term=(
                    self.learning_rate
                    * (1 + self.discount_factor * sprime_qstar - s_qstar)
                ),
            )


class SARSA(BasePolicy):
    def __init__(
        self, policy, max_iter, actor, discount_factor, learning_rate=0.1, **kwargs
    ):
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
        return "SARSA"

    def single_iteration(self):
        status = MOVING_TOKEN
        n_iter = 0
        self.actor.start_track()

        while status != FINISH_TOKEN:
            n_iter += 1
            eps = max(1 / n_iter, 0.05)
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


class Tuner:
    def __init__(self, algo, param_grid):
        self.param_grid = param_grid
        self.algo = algo

    def fit(self):
        self.mean_steps = {}
        for param_set in self.param_grid:
            algo_obj = self.algo(**param_set)
            algo_obj.explore()
            self.mean_steps[param_set] = np.mean(
                [algo_obj.exploit() for _ in range(10)]
            )
        return self.mean_steps


if __name__ == "__main__":
    DIR = "/Users/home/Documents/JHU/machine_learning/course_projects/project6"
    from machine_learning.reinforcement_learning import actors
    from machine_learning.reinforcement_learning import policy

    track = actors.Track(f"{DIR}/R-track.txt", harsh_crash_variant=True)
    track.start_track()
    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    tn = Tuner(
        algo=lambda **kwargs: ValueIteration(policy=pi, actor=track, **kwargs),
        param_grid=[
            {
                "discount_factor": dr,
                "max_exploit": 10000,
                "max_iter": 15,
                "stat_collection_freq": 100000,
            }
            for dr in np.arange(0.4, 1, 0.2)
        ],
    )
    res = tn.fit()
    import ipdb

    ipdb.set_trace()
    res
