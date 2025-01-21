from env.graph import PTSPGraph, PTSPEnv, PTSPValEnv

import gymnasium as gym
import numpy as np


# creates random metric env
class PTSPRandomEnv(gym.Env):
    def __init__(self, num_nodes=6, min_dist=1, max_dist=100, seed=None):
        self.num_nodes = num_nodes
        self.min_dist = min_dist
        self.max_dist = max_dist

        self.generator = np.random.default_rng(seed=seed)

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.observation_space = gym.spaces.Dict(
            spaces={
                "current_node": gym.spaces.Discrete(self.num_nodes),
                "visited": gym.spaces.MultiBinary(self.num_nodes),
                "prob": gym.spaces.Box(low=0, high=1, shape=(self.num_nodes,)),
                "dist": gym.spaces.Box(
                    low=1,
                    high=100,
                    shape=(self.num_nodes * (self.num_nodes - 1) // 2,),
                ),
            }
        )

    def reset(self, **kwargs):
        nodes = self.generator.random(self.num_nodes - 1)
        nodes /= sum(nodes)
        nodes = [0] + nodes.tolist()

        points = self.generator.uniform(
            self.min_dist,
            (self.max_dist - self.min_dist) / np.sqrt(2) + self.min_dist,
            (self.num_nodes, 2),
        )

        edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                edges.append(np.linalg.norm(points[i] - points[j]))

        self.graph = PTSPGraph(nodes, edges)
        self.env = PTSPEnv(self.graph)

        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
