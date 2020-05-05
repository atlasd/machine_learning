from machine_learning.reinforcement_learning import actors, policy
from machine_learning.reinforcement_learning.algorithms import *
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from itertools import product
from joblib import Parallel, delayed

DIR = "/Users/home/Documents/JHU/machine_learning/course_projects/project6"

TRACKS = ["L-track.txt", "O-track.txt", "R-track.txt"]
ALGOS = [ValueIteration, QLearning, SARSA]
HARSH_CRASH = [True, False]
ALGO_KWARGS = {
    ("L-track.txt", ValueIteration, True): dict(discount_factor=0.7, max_iter=250,),
    ("L-track.txt", ValueIteration, False): dict(discount_factor=0.8, max_iter=100,),
    ("O-track.txt", ValueIteration, True): dict(discount_factor=0.7, max_iter=1000,),
    ("O-track.txt", ValueIteration, False): dict(discount_factor=0.8, max_iter=500,),
    ("L-track.txt", QLearning, True): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=100000,
    ),
    ("L-track.txt", QLearning, False): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=40000,
    ),
    ("O-track.txt", QLearning, True): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=175000,
    ),
    ("O-track.txt", QLearning, False): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=75000,
    ),
    ("L-track.txt", SARSA, True): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=125000,
    ),
    ("L-track.txt", SARSA, False): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=4000,
    ),
    ("O-track.txt", SARSA, True): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=200000,
    ),
    ("O-track.txt", SARSA, False): dict(
        discount_factor=0.5, learning_rate=0.2, max_iter=75000
    ),
    ("R-track.txt", ValueIteration, True): dict(max_iter=100, discount_factor=0.1),
    ("R-track.txt", ValueIteration, False): dict(max_iter=100, discount_factor=0.1),
    ("R-track.txt", QLearning, True): dict(
        discount_factor=0.3, learning_rate=0.1, max_iter=5000,
    ),
    ("R-track.txt", QLearning, False): dict(
        discount_factor=0.3, learning_rate=0.1, max_iter=5000,
    ),
    ("R-track.txt", SARSA, True): dict(
        discount_factor=0.3, learning_rate=0.1, max_iter=5000
    ),
    ("R-track.txt", SARSA, False): dict(
        discount_factor=0.3, learning_rate=0.1, max_iter=5000,
    ),
}


for track, algo, crash_variant in product(TRACKS, ALGOS, HARSH_CRASH):
    ALGO_KWARGS[(track, algo, crash_variant)].update(
        {"fname": f"{track}_{algo}_{crash_variant}"}
    )


def run_experiment(track, algo, crash_variant, kwargs):
    np.random.seed(73)
    track_obj = actors.Track(f"{DIR}/{track}", harsh_crash_variant=crash_variant)
    track_obj.start_track()
    logger.info(f"Initializing track {track} with harsh_variant={crash_variant}")

    ALGO_KWARGS.update(
        {"fname": f"{track}_{algo}_{crash_variant}_{kwargs.get('discount_factor')}"}
    )

    pi = policy.Policy(states=track_obj.get_states(), actions=track_obj.get_actions())

    algo_object = algo(policy=pi, actor=track_obj, **kwargs)

    logger.info(f"Updating policy via {algo}...")
    algo_object.explore()

    results = []
    for _ in range(20):
        path, steps = algo_object.exploit()
        results.append(steps)

    logger.info(kwargs)
    logger.info(f"Avg Steps to Solve: {np.mean(results)}")


if __name__ == "__main__":
    results = Parallel(n_jobs=2)(
        delayed(run_experiment)(
            track=track,
            algo=algo,
            crash_variant=crash_variant,
            kwargs=dict(
                ALGO_KWARGS.get((track, algo, crash_variant)), max_exploit=1000
            ),
        )
        for track, algo, crash_variant in product(TRACKS, ALGOS, HARSH_CRASH)
        if track[0] == "R" and algo != ValueIteration
    )
