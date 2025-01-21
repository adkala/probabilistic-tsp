from itertools import permutations
from tqdm.rich import tqdm

from env.graph import PTSPGraph, PTSPEnv, PTSPValEnv
from env.random import PTSPRandomEnv

import numpy as np
import random


def get_random_graph_env(num_nodes=6, min_dist=1, max_dist=100, seed=None, num_val=5):
    if seed is None:
        seed = np.random.randint(1000)
    return PTSPRandomEnv(num_nodes, min_dist, max_dist, seed), [
        PTSPValEnv(create_random_graph(num_nodes, min_dist, max_dist, seed + i + 1))
        for i in range(num_val)
    ]


def create_random_env(**kwargs):
    graph = create_random_graph(**kwargs)
    return PTSPEnv(graph), PTSPValEnv(graph)


def create_random_graph(num_nodes=6, min_dist=1, max_dist=100, seed=None):
    generator = np.random.default_rng(seed=seed)

    null_nodes = min(generator.geometric(p=0.5), num_nodes - 1)
    nodes = generator.random(num_nodes - null_nodes)
    nodes /= sum(nodes)
    nodes = nodes.tolist() + [0] * (null_nodes - 1)
    # generator.shuffle(nodes)
    nodes = [0] + nodes

    points = generator.uniform(
        min_dist, (max_dist - min_dist) / np.sqrt(2) + min_dist, (num_nodes, 2)
    )

    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges.append(np.linalg.norm(points[i] - points[j]))

    return PTSPGraph(nodes, edges)


def get_optimal_path(graph):
    search_set = set(range(graph.num_nodes)) - {0}

    scores = []

    min_perm, min_exp_dist = None, float("inf")
    for perm in tqdm(
        permutations(search_set), total=np.math.factorial(len(search_set))
    ):
        exp_dist = get_expected_dist(graph, [0] + list(perm))
        scores.append(exp_dist)
        if exp_dist < min_exp_dist:
            min_perm, min_exp_dist = perm, exp_dist
    return min_perm, min_exp_dist, scores


def get_expected_dist(graph, path):
    """
    param `path` includes start node
    """
    dist = 0
    exp_dist = 0
    for i in range(len(path) - 1):
        dist += graph.get_dist(path[i], path[i + 1])
        exp_dist += dist * graph.get_prob(path[i + 1])
    return exp_dist


def get_policy_path(policy, env, num_nodes=6):
    state, _ = env.reset()

    path = [state["current_node"]]
    while len(path) < num_nodes:
        action, _ = policy.predict(state, deterministic=True)
        state, _, _, _, _ = env.step(action)
        path.append(state["current_node"])
    env.reset()
    return path


if __name__ == "__main__":
    graph = create_random_graph(num_nodes=6)

    optimal_path, optimal_dist, _ = get_optimal_path(graph)
    print(optimal_path, optimal_dist)
