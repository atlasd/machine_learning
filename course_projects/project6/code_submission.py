from machine_learning.reinforcement_learning import actors, policy
from machine_learning.reinforcement_learning.algorithms import *
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from toolz import dicttoolz


if __name__ != "__main__":
    logger.info(
        "Running Value Iteration on L-Track Experiment with restart after crash..."
    )
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/L-track.txt", harsh_crash_variant=True
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    vi = ValueIteration(policy=pi, actor=track, discount_factor=0.7, max_iter=200)

    logger.info("Updating policy via Value Iteration...")
    vi.explore()
    vi.policy.set_exploit_mode()

    value_iteration_results = []
    for _ in range(20):
        path, steps = vi.exploit()
        value_iteration_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(value_iteration_results)}")

    logger.info(
        "Running Value Iteration on L-Track Experiment with no restart after crash..."
    )
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/L-track.txt", harsh_crash_variant=False
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    vi = ValueIteration(policy=pi, actor=track, discount_factor=0.8, max_iter=100)

    logger.info("Updating policy via Value Iteration...")
    vi.explore()
    vi.policy.set_exploit_mode()

    value_iteration_results = []
    for _ in range(20):
        path, steps = vi.exploit()
        value_iteration_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(value_iteration_results)}")


if __name__ != "__main__":
    logger.info(
        "Running Q-Learning on L-Track Experiment with no restart after crash..."
    )
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/L-track.txt", harsh_crash_variant=False
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    q_learner = QLearning(
        policy=pi, actor=track, discount_factor=0.5, learning_rate=0.2, max_iter=100000
    )

    logger.info("Updating policy via Q-Learning...")
    q_learner.explore()

    q_learning_results = []
    for _ in range(20):
        path, steps = q_learner.exploit()
        q_learning_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(q_learning_results)}")

    logger.info("Running SARSA on L-Track Experiment with no restart after crash...")
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/L-track.txt", harsh_crash_variant=False
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    sarsa = SARSA(
        policy=pi, actor=track, discount_factor=0.5, learning_rate=0.2, max_iter=100000
    )

    logger.info("Updating policy via SARSA...")
    sarsa.explore()

    sarsa_results = []
    for _ in range(20):
        path, steps = sarsa.exploit()
        sarsa_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(sarsa_results)}")

    logger.info("Running Q-Learning on L-Track Experiment with restart after crash...")
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/L-track.txt", harsh_crash_variant=True
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    q_learner = QLearning(
        policy=pi, actor=track, discount_factor=0.5, learning_rate=0.2, max_iter=200000
    )

    logger.info("Updating policy via Q-Learning...")
    q_learner.explore()

    q_learning_results = []
    for _ in range(20):
        path, steps = q_learner.exploit()
        q_learning_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(q_learning_results)}")

    logger.info("Running SARSA on L-Track Experiment with restart after crash...")
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/L-track.txt", harsh_crash_variant=True
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    sarsa = SARSA(
        policy=pi, actor=track, discount_factor=0.5, learning_rate=0.2, max_iter=200000
    )

    logger.info("Updating policy via SARSA...")
    sarsa.explore()

    sarsa_results = []
    for _ in range(20):
        path, steps = sarsa.exploit()
        sarsa_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(sarsa_results)}")


if __name__ == "__main__":

    logger.info(
        "Running Q-Learning on O-Track Experiment with no restart after crash..."
    )
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/O-track.txt", harsh_crash_variant=False
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    q_learner = QLearning(
        policy=pi, actor=track, discount_factor=0.5, learning_rate=0.2, max_iter=150000
    )

    logger.info("Updating policy via Q-Learning...")
    q_learner.explore()

    q_learning_results = []
    for _ in range(20):
        path, steps = q_learner.exploit()
        q_learning_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(q_learning_results)}")

    logger.info("Running SARSA on O-Track Experiment with no restart after crash...")
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/O-track.txt", harsh_crash_variant=False
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    sarsa = SARSA(
        policy=pi, actor=track, discount_factor=0.5, learning_rate=0.2, max_iter=150000
    )

    logger.info("Updating policy via SARSA...")
    sarsa.explore()

    sarsa_results = []
    for _ in range(20):
        path, steps = sarsa.exploit()
        sarsa_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(sarsa_results)}")

    logger.info("Running Q-Learning on O-Track Experiment with restart after crash...")
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/O-track.txt", harsh_crash_variant=True
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    q_learner = QLearning(
        policy=pi, actor=track, discount_factor=0.5, learning_rate=0.2, max_iter=300000
    )

    logger.info("Updating policy via Q-Learning...")
    q_learner.explore()

    q_learning_results = []
    for _ in range(20):
        path, steps = q_learner.exploit()
        q_learning_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(q_learning_results)}")

    logger.info("Running SARSA on O-Track Experiment with restart after crash...")
    np.random.seed(73)
    track = actors.Track(
        "course_projects/project6/O-track.txt", harsh_crash_variant=True
    )
    track.start_track()
    logger.info("Initializing track...")

    pi = policy.Policy(states=track.get_states(), actions=track.get_actions())

    sarsa = SARSA(
        policy=pi, actor=track, discount_factor=0.5, learning_rate=0.2, max_iter=300000
    )

    logger.info("Updating policy via SARSA...")
    sarsa.explore()

    sarsa_results = []
    for _ in range(20):
        path, steps = sarsa.exploit()
        sarsa_results.append(steps)

    logger.info(f"Avg Steps to Solve: {np.mean(sarsa_results)}")
